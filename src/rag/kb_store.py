"""
知识库切片向量存储（Qdrant）。

职责边界：
- 只负责 collection / payload 索引 / 增删改查，不负责切分与 embedding；
- 所有检索强制带 kb_id + agent_scope + status 过滤，避免跨库、跨 Agent 泄漏；
- point id 由 (kb_id, chunk_id) 稳定生成，同一片重复写入是幂等的。

与旧 `vector_store.py` 并存，互不影响：旧链路继续用 `product_faqs`，
新链路写 `kb_chunks_v1`，读侧可切别名 `kb_chunks`。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from qdrant_client import QdrantClient, models

from src.core.config import settings
from src.core.logging import log

# payload 索引：只有需要过滤的字段才建索引
PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "kb_id": models.PayloadSchemaType.KEYWORD,
    "tenant_id": models.PayloadSchemaType.KEYWORD,
    "agent_scope": models.PayloadSchemaType.KEYWORD,  # 数组字段，用 MatchAny 过滤
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "status": models.PayloadSchemaType.KEYWORD,
    "strategy": models.PayloadSchemaType.KEYWORD,
    "chunk_type": models.PayloadSchemaType.KEYWORD,
    "biz_category": models.PayloadSchemaType.KEYWORD,
    "index_version": models.PayloadSchemaType.INTEGER,
}

# 入库前必须存在的 payload 字段（缺失直接报错，不写半成品）
REQUIRED_PAYLOAD_KEYS = (
    "text",
    "parent_id",
    "kb_id",
    "doc_id",
    "agent_scope",
    "strategy",
    "chunk_type",
    "breadcrumb",
)

VALID_STATUS = {"active", "staging", "archived"}


@dataclass(slots=True)
class KbChunkPoint:
    """待写入的一片知识。

    vector 由调用方（embedding 层）提供，本模块不加载模型。
    """

    chunk_id: str
    kb_id: str
    vector: Sequence[float]
    payload: dict[str, Any] = field(default_factory=dict)
    index_version: int = 1
    status: str = "active"


@dataclass(slots=True)
class KbSearchHit:
    """检索命中结果，屏蔽 qdrant 原生类型，方便上层消费。"""

    chunk_id: str
    parent_id: str
    score: float
    text: str
    payload: dict[str, Any]


def point_id_for(kb_id: str, chunk_id: str, index_version: int | None = None) -> str:
    """稳定 point id：同一库内同版本同片段映射到同一 UUID。

    版本号参与生成，保证新旧版本可以并存（staging 期间旧 active 仍可检索），
    同版本重复写入则覆盖自身，保持幂等。
    """
    scope = f"v{int(index_version)}:" if index_version is not None else ""
    return str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"kb-chunk:{kb_id}:{scope}{chunk_id}")
    )


class KbStore:
    """Qdrant 知识库读写封装。"""

    def __init__(
        self,
        collection_name: str | None = None,
        alias: str | None = None,
        dimension: int | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        self.collection_name = collection_name or settings.kb_collection
        self.alias = alias or settings.kb_collection_alias
        self.dimension = dimension or settings.embedding_dimension
        self.client = client or QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

    # ==================== 初始化 ====================

    def ensure_collection(self, recreate: bool = False) -> bool:
        """保证 collection、payload 索引、别名都存在，返回是否新建。

        维度不一致时直接报错，不自动删库——删数据必须由人显式执行。
        """
        exists = self.client.collection_exists(self.collection_name)

        if recreate and exists:
            log.warning(f"重建 collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            log.info(f"已创建 collection `{self.collection_name}` (dim={self.dimension})")

        self._assert_dimension()
        self._ensure_indexes()
        self._ensure_alias()
        return not exists

    def _assert_dimension(self) -> None:
        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            # 命名向量场景取默认向量
            vectors = next(iter(vectors.values()), None)
        existing = getattr(vectors, "size", None)
        if existing != self.dimension:
            raise RuntimeError(
                f"collection `{self.collection_name}` 维度为 {existing}，"
                f"配置期望 {self.dimension}。请迁移或显式重建，不会自动删库。"
            )

    def _ensure_indexes(self) -> None:
        for key, schema in PAYLOAD_INDEXES.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=key,
                    field_schema=schema,
                    wait=True,
                )
            except Exception as exc:  # 索引已存在时 Qdrant 可能返回错误，保持幂等
                log.debug(f"payload 索引 `{key}` 创建跳过: {exc}")

    def _ensure_alias(self) -> None:
        aliases = {
            item.alias_name: item.collection_name
            for item in self.client.get_aliases().aliases
        }
        current = aliases.get(self.alias)
        if current == self.collection_name:
            return
        if current is not None:
            log.warning(
                f"别名 `{self.alias}` 已指向 `{current}`，不自动覆盖；"
                f"需要切换时请显式更新别名。"
            )
            return
        self.client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=self.collection_name,
                        alias_name=self.alias,
                    )
                )
            ]
        )
        log.info(f"已创建别名 `{self.alias}` -> `{self.collection_name}`")

    # ==================== 写入 ====================

    def upsert(
        self,
        records: Iterable[KbChunkPoint],
        batch_size: int = 128,
        wait: bool = True,
    ) -> int:
        """批量写入切片向量，返回写入条数。"""
        records = list(records)
        if not records:
            return 0

        self.ensure_collection()

        points: list[models.PointStruct] = []
        for record in records:
            payload = self._build_payload(record)
            if len(record.vector) != self.dimension:
                raise ValueError(
                    f"向量维度 {len(record.vector)} != collection 维度 {self.dimension}"
                    f"（chunk={record.chunk_id}）"
                )
            points.append(
                models.PointStruct(
                    id=point_id_for(record.kb_id, record.chunk_id, record.index_version),
                    vector=[float(x) for x in record.vector],
                    payload=payload,
                )
            )

        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=wait,
            )
        log.info(f"写入 {len(points)} 个切片 -> `{self.collection_name}`")
        return len(points)

    def _build_payload(self, record: KbChunkPoint) -> dict[str, Any]:
        if record.status not in VALID_STATUS:
            raise ValueError(f"非法 status: {record.status}")

        payload: dict[str, Any] = dict(record.payload)
        payload["chunk_id"] = record.chunk_id
        payload.setdefault("kb_id", record.kb_id)
        payload["index_version"] = int(record.index_version)
        payload["status"] = record.status

        for key in REQUIRED_PAYLOAD_KEYS:
            value = payload.get(key)
            if value is None or value == "" or value == [] or value == {}:
                raise ValueError(f"payload 缺少必填字段 `{key}`（chunk={record.chunk_id}）")

        if not isinstance(payload["agent_scope"], (list, tuple)):
            raise ValueError("agent_scope 必须是数组，例如 ['customer_service']")
        payload["agent_scope"] = list(payload["agent_scope"])
        payload["breadcrumb"] = list(payload["breadcrumb"])
        return payload

    # ==================== 检索 ====================

    def search(
        self,
        vector: Sequence[float],
        kb_ids: Sequence[str],
        agent_role: str,
        *,
        tenant_id: str | None = None,
        top_k: int = 20,
        score_threshold: float | None = None,
        status: str = "active",
        extra_must: list[models.FieldCondition] | None = None,
    ) -> list[KbSearchHit]:
        """按元数据过滤后做向量检索。

        kb_ids / agent_role 是必填参数，调用方无法绕过隔离条件。
        agent_role 是逻辑 Agent 角色（如 intent_agent），不是人工客服会话 ID。
        """
        if not kb_ids:
            raise ValueError("kb_ids 不能为空")
        if not agent_role:
            raise ValueError("agent_role 不能为空")
        if len(vector) != self.dimension:
            raise ValueError(f"查询向量维度 {len(vector)} != {self.dimension}")

        must: list[Any] = [
            models.FieldCondition(
                key="kb_id",
                match=models.MatchAny(any=list(kb_ids)),
            ),
            models.FieldCondition(
                key="agent_scope",
                match=models.MatchAny(any=[agent_role]),
            ),
            models.FieldCondition(
                key="status",
                match=models.MatchValue(value=status),
            ),
        ]
        if tenant_id:
            must.append(
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id),
                )
            )
        if extra_must:
            must.extend(extra_must)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=[float(x) for x in vector],
            query_filter=models.Filter(must=must),
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )

        hits: list[KbSearchHit] = []
        for point in response.points:
            payload = dict(point.payload or {})
            hits.append(
                KbSearchHit(
                    chunk_id=str(payload.get("chunk_id", point.id)),
                    parent_id=str(payload.get("parent_id", "")),
                    score=float(point.score),
                    text=str(payload.get("text", "")),
                    payload=payload,
                )
            )
        return hits

    # ==================== 版本 / 状态 ====================

    def set_status(
        self,
        doc_id: str,
        status: str,
        *,
        kb_id: str | None = None,
        index_version: int | None = None,
    ) -> None:
        """按文档批量改状态，用于 staging -> active、下线归档。"""
        if status not in VALID_STATUS:
            raise ValueError(f"非法 status: {status}")

        must: list[Any] = [
            models.FieldCondition(
                key="doc_id",
                match=models.MatchValue(value=doc_id),
            )
        ]
        if kb_id:
            must.append(
                models.FieldCondition(
                    key="kb_id",
                    match=models.MatchValue(value=kb_id),
                )
            )
        if index_version is not None:
            must.append(
                models.FieldCondition(
                    key="index_version",
                    match=models.MatchValue(value=int(index_version)),
                )
            )

        self.client.set_payload(
            collection_name=self.collection_name,
            payload={"status": status},
            points=models.Filter(must=must),
            wait=True,
        )
        log.info(
            f"设置文档 `{doc_id}` 状态 -> {status}"
            f"（kb={kb_id or '*'}, version={index_version or '*'}）"
        )

    def delete_doc(self, doc_id: str, kb_id: str | None = None) -> None:
        """物理删除某文档的全部切片。"""
        must: list[Any] = [
            models.FieldCondition(
                key="doc_id",
                match=models.MatchValue(value=doc_id),
            )
        ]
        if kb_id:
            must.append(
                models.FieldCondition(
                    key="kb_id",
                    match=models.MatchValue(value=kb_id),
                )
            )
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=models.Filter(must=must)),
            wait=True,
        )
        log.info(f"删除文档 `{doc_id}` 的全部切片（kb={kb_id or '*'}）")

    def delete_version(self, doc_id: str, index_version: int, kb_id: str | None = None) -> None:
        """删除某文档的指定版本，用于新版本生效后清理旧数据。"""
        must: list[Any] = [
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
            models.FieldCondition(
                key="index_version",
                match=models.MatchValue(value=int(index_version)),
            ),
        ]
        if kb_id:
            must.append(
                models.FieldCondition(key="kb_id", match=models.MatchValue(value=kb_id))
            )
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=models.Filter(must=must)),
            wait=True,
        )
        log.info(f"删除文档 `{doc_id}` 版本 v{index_version} 的切片")

    # ==================== 统计 ====================

    def count(
        self,
        *,
        kb_id: str | None = None,
        doc_id: str | None = None,
        status: str | None = None,
        index_version: int | None = None,
    ) -> int:
        must: list[Any] = []
        if kb_id:
            must.append(
                models.FieldCondition(key="kb_id", match=models.MatchValue(value=kb_id))
            )
        if doc_id:
            must.append(
                models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
            )
        if status:
            must.append(
                models.FieldCondition(key="status", match=models.MatchValue(value=status))
            )
        if index_version is not None:
            must.append(
                models.FieldCondition(
                    key="index_version",
                    match=models.MatchValue(value=int(index_version)),
                )
            )

        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=models.Filter(must=must) if must else None,
            exact=True,
        )
        return int(result.count)

    def versions(
        self,
        doc_id: str,
        *,
        kb_id: str | None = None,
        status: str | None = None,
    ) -> list[int]:
        """列出某文档已写入 collection 的版本号（升序）。

        只读 payload 的 index_version，不参与向量检索，用于入库前的版本校验
        和生效后的旧版本清理。
        """
        must: list[Any] = [
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
        ]
        if kb_id:
            must.append(
                models.FieldCondition(key="kb_id", match=models.MatchValue(value=kb_id))
            )
        if status:
            must.append(
                models.FieldCondition(key="status", match=models.MatchValue(value=status))
            )

        versions: set[int] = set()
        offset: Any = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(must=must),
                limit=1000,
                offset=offset,
                with_payload=["index_version"],
                with_vectors=False,
            )
            for record in records:
                value = (record.payload or {}).get("index_version")
                if value is not None:
                    versions.add(int(value))
            if offset is None:
                break
        return sorted(versions)

    def health(self) -> dict[str, Any]:
        """环境自检：collection 是否存在、维度、切片数、别名指向。"""
        exists = self.client.collection_exists(self.collection_name)
        aliases = {
            item.alias_name: item.collection_name
            for item in self.client.get_aliases().aliases
        }
        return {
            "collection": self.collection_name,
            "exists": exists,
            "dimension": self.dimension,
            "points": self.count() if exists else 0,
            "alias": self.alias,
            "alias_target": aliases.get(self.alias),
        }


_kb_store: KbStore | None = None


def get_kb_store() -> KbStore:
    global _kb_store
    if _kb_store is None:
        _kb_store = KbStore()
    return _kb_store
