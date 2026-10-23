"""
Qdrant 向量库封装 —— 负责 collection 初始化、upsert、search
"""
from __future__ import annotations
import hashlib
import uuid as _uuid
from typing import Any
from qdrant_client import QdrantClient, models

from src.core.config import settings
from src.core.logging import log
from src.rag.embedding import get_embedding


def _str_id_to_uuid(str_id: str) -> _uuid.UUID:
    """把任意字符串转成稳定的 UUID（基于 MD5，保证同字符串总是同 UUID）"""
    md5 = hashlib.md5(str_id.encode("utf-8")).hexdigest()
    return _uuid.UUID(md5)


class VectorStore:
    """Qdrant 向量存储服务"""

    def __init__(self):
        self.collection_name = settings.qdrant_collection
        self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self._ensure_collection()

    def _ensure_collection(self):
        """保证 collection 存在且维度匹配"""
        embedding = get_embedding()
        dim = embedding.dimension

        try:
            collections = self.client.get_collections()
            exists = any(c.name == self.collection_name for c in collections.collections)

            if exists:
                # 检查维度是否一致
                coll_info = self.client.get_collection(self.collection_name)
                existing_dim = coll_info.config.params.vectors.size
                if existing_dim != dim:
                    log.warning(f"⚠️ Qdrant collection 维度不匹配: 存在 {existing_dim}, 期望 {dim}")
                    # 重建（可改成双写迁移，MVP 阶段直接重建）
                    log.warning("🔄 自动删除旧 collection 并重建...")
                    self.client.delete_collection(self.collection_name)
                    self._create_collection(dim)
            else:
                self._create_collection(dim)

            log.info(f"✅ Qdrant collection `{self.collection_name}` ready (dim={dim})")
        except Exception as e:
            log.error(f"❌ Qdrant 初始化失败: {e}")

    def _create_collection(self, dim: int):
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert(self, points: list[dict[str, Any]]):
        """
        批量写入向量
        points: [
            {"id": "faq_001", "text": "支持7天无理由退换货", "metadata": {"source": "policy"}},
            ...
        ]
        """
        if not points:
            return

        texts = [p["text"] for p in points]
        embeddings = get_embedding().encode(texts)

        qdrant_points = []
        for i, p in enumerate(points):
            raw_id = p["id"]
            # Qdrant 只接受整数或 UUID，字符串 ID 转 UUID
            try:
                # 先试能不能直接转 int
                point_id = int(raw_id)
            except (ValueError, TypeError):
                point_id = _str_id_to_uuid(str(raw_id))

            qdrant_points.append(models.PointStruct(
                id=point_id,
                vector=embeddings[i],
                payload={
                    "text": p["text"],
                    "_raw_id": str(raw_id),  # 保留原始字符串 ID 在 payload 里
                    **p.get("metadata", {}),
                },
            ))

        self.client.upsert(collection_name=self.collection_name, points=qdrant_points)
        log.info(f"📥 写入 {len(qdrant_points)} 条向量到 Qdrant")

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """检索 top-k 最相似的向量"""
        query_vec = get_embedding().encode(query)[0]
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=top_k,
            with_payload=True,
        )

        hits = []
        for point in results.points:
            hits.append({
                "id": point.id,
                "score": point.score,
                "text": point.payload.get("text", ""),
                **{k: v for k, v in point.payload.items() if k != "text"},
            })
        return hits

    def count(self) -> int:
        try:
            return self.client.count(collection_name=self.collection_name).count
        except Exception:
            return 0


_vs: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _vs
    if _vs is None:
        _vs = VectorStore()
    return _vs
