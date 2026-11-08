"""
知识检索链路：清洗 → 查询编码 → 硬过滤检索 → 软过滤兜底 → 父块归并 → rerank → 上下文预算。

设计原则：
- 硬过滤只放“违反就不可用”的条件：租户 / Agent / 知识库 / active 状态；
- chunk_type 只做软过滤，命中不足时自动去掉，避免意图误判导致 0 召回；
- 查询侧使用 BGE 指令前缀，入库侧不加，二者由 Embedder 分开编码；
- 检索命中子块，返回父块全文，并附带可跳转的 citation 元数据；
- 无可靠命中时明确返回 no_hit，不把低分片段硬塞给大模型。

用法：
    python -m src.rag.kb_retriever "忘记密码怎么办" \
        --kb kb_faq_v1 --agent-role customer_service --chunk-type faq
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Protocol

from qdrant_client import models

from src.core.config import settings
from src.core.logging import log
from src.rag.kb_embed import Embedder, get_embedder
from src.rag.kb_store import KbSearchHit, KbStore, get_kb_store
from src.tools.kb_chunking import estimate_tokens

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"\s+")


# ==================== 请求 / 结果 ====================


@dataclass(slots=True)
class RetrieveRequest:
    """检索请求参数。

    设计原则：
    - 硬过滤条件（kb_ids / agent_role / tenant_id / active）直接进 store.search；
    - chunk_types 是软过滤，命中不足会自动去掉再搜一次；
    - score_threshold 在启用 rerank 时自动延后到 rerank 之后判断。
    """

    # 用户原始问题，清洗前保留用于诊断
    query: str
    # 调用方 Agent 的逻辑角色（如 customer_service / intent_agent），硬过滤条件
    agent_role: str
    # 要检索的知识库 ID 列表，硬过滤条件，至少一个
    kb_ids: list[str]
    # 租户 ID，用于多租户隔离，硬过滤条件
    tenant_id: str = "default"
    # 软过滤：faq / doc_section / table / text，不是硬约束，命中不足自动回退
    chunk_types: list[str] | None = None
    # 调用方（Agent/LLM）给出的改写或子问题，多路召回与 query 一起编码
    query_variants: list[str] = field(default_factory=list)
    # query_variants 最多保留几个参与检索（含原始 query）
    max_variants: int = 3
    # 查询侧编码指令：None=用 Embedder 默认指令；""=关闭；也可传自定义指令做 A/B
    query_instruction: str | None = None
    # 单次向量检索 top_k 候选数（多路时每路都用这个值）
    top_k: int = 40
    # rerank 后最终保留的父块数量上限
    rerank_top_n: int = 6
    # 返回上下文的 token 总预算（包含所有父块的 header + body + 分隔符）
    token_budget: int = 2400
    # 软过滤下的最少候选阈值，低于此数触发 chunk_type 兜底
    min_candidates: int = 5
    # 是否启用 CrossEncoder rerank；关闭时直接按向量分数排序
    enable_rerank: bool = True
    # 软过滤命中不足时，是否允许去掉 chunk_type 再搜一次
    allow_filter_fallback: bool = True
    # 向量余弦分数阈值；启用 rerank 时会延后到 rerank 之后判断
    score_threshold: float | None = None
    # rerank 分数阈值；仅在 rerank 启用时生效，低于此值视为不可靠命中
    rerank_threshold: float | None = None


@dataclass(slots=True)
class Citation:
    """单个父块的引用元数据，用于生成可溯源的 citation。"""

    # 在最终上下文列表中的序号（从 1 开始）
    index: int
    # 原始文档 ID（入库时的唯一标识）
    doc_id: str
    # 原始文档名称（文件名 / 标题）
    doc_name: str
    # 命中子块中分数最高的那个 chunk_id（用于跳转子定位）
    chunk_id: str
    # 父块 ID，整个 RetrievedContext 对应的完整内容块
    parent_id: str
    # 文档内层级路径，如 ["FAQ", "账户"]
    breadcrumb: list[str]
    # 文档内位置信息，如 {page: 3, row: 15}，用于 PDF/表格跳转
    loc: dict[str, Any]
    # 原始文件路径或来源标识
    source_path: str
    # 向量余弦分数（取该父块下所有子块的最大值）
    score: float
    # rerank 分数（CrossEncoder 输出，可能为 None 表示未 rerank 或降级）
    rerank_score: float | None = None
    # 业务分类 ID，用于前端按类别筛选
    biz_category: str = ""
    # 业务分类中文名，用于展示
    biz_category_name: str = ""


@dataclass(slots=True)
class RetrievedContext:
    """组装好的单条上下文：header + 截断后的父块正文 + citation。"""

    # 父块唯一标识，用于去重和关联 citation
    parent_id: str
    # 展示给 LLM 的头部信息（含文档名、章节、位置），不参与截断
    header: str
    # 父块正文（可能被 token 预算截断），传给 LLM 的主要内容
    text: str
    # 引用元数据，供前端溯源和 LLM 输出 citation 时使用
    citation: Citation
    # 本父块下被检索命中的所有子块 chunk_id 列表（已去重排序）
    matched_chunks: list[str]
    # 向量余弦分数（取该父块下命中子块的最大值）
    score: float
    # rerank 分数（CrossEncoder 输出，None 表示未 rerank 或降级）
    rerank_score: float | None
    # header + body 的 token 估算值，用于后续块累加预算
    tokens: int
    # text 是否被 token 预算截断；False 表示正文完整
    truncated: bool = False


@dataclass(slots=True)
class RetrieveResult:
    """检索整体结果，包含上下文列表、命中状态和诊断信息。"""

    # 用户原始问题（与 RetrieveRequest.query 相同，方便日志对齐）
    query: str
    # 清洗后的查询（归一化空白和控制字符）
    cleaned_query: str
    # 最终参与检索的 query 变体列表（含 cleaned_query + 改写子问题，受 max_variants 限制）
    variants: list[str]
    # 组装好的上下文列表，按最终排序（rerank 或向量分数降序）
    contexts: list[RetrievedContext]
    # 是否无可靠命中（True 时 contexts 一定为空）
    no_hit: bool
    # 无命中原因：empty_query / empty_kb_ids / no_candidates / below_score_threshold / below_rerank_threshold / empty_context
    no_hit_reason: str | None
    # 诊断信息：过滤条件、时间打点、软过滤是否兜底、最佳分数等
    diagnostics: dict[str, Any]

    def context_text(self) -> str:
        return "\n\n".join(f"{ctx.header}\n{ctx.text}" for ctx in self.contexts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "cleaned_query": self.cleaned_query,
            "variants": self.variants,
            "no_hit": self.no_hit,
            "no_hit_reason": self.no_hit_reason,
            "contexts": [asdict(ctx) for ctx in self.contexts],
            "context_text": self.context_text(),
            "diagnostics": self.diagnostics,
        }


# ==================== 清洗 / 归并 ====================


def clean_query(query: str) -> str:
    """只归一化空白与控制字符，保留订单号、型号、符号等有效信息。"""
    return _WS_RE.sub(" ", _CONTROL_RE.sub("", query or "")).strip()


def _build_variants(request: RetrieveRequest, cleaned: str) -> list[str]:
    variants = [cleaned] if cleaned else []
    for item in request.query_variants:
        value = clean_query(item)
        if value and value not in variants:
            variants.append(value)
    limit = max(1, request.max_variants)
    return variants[:limit]


def _rrf_fuse(results: list[list[KbSearchHit]], k: int = 60) -> list[KbSearchHit]:
    """多路召回按 RRF 排序，但保留原始余弦分数用于阈值判断。

    RRF 分数只适合排序（量级约 1/60），不能直接和 score_threshold 比较；
    因此 score 取各路中的最大余弦值。
    """
    merged: dict[str, tuple[float, float, KbSearchHit]] = {}
    for hits in results:
        for rank, hit in enumerate(hits, 1):
            contribution = 1.0 / (k + rank)
            cosine = float(hit.score)
            current = merged.get(hit.chunk_id)
            if current is None:
                merged[hit.chunk_id] = (contribution, cosine, hit)
            else:
                merged[hit.chunk_id] = (
                    current[0] + contribution,
                    max(current[1], cosine),
                    current[2],
                )

    ranked = sorted(merged.values(), key=lambda item: item[0], reverse=True)
    return [
        KbSearchHit(
            chunk_id=hit.chunk_id,
            parent_id=hit.parent_id,
            score=max_cosine,
            text=hit.text,
            payload=hit.payload,
        )
        for _, max_cosine, hit in ranked
    ]


@dataclass(slots=True)
class _Candidate:
    """_group_by_parent 的中间结构：同一父块聚合后的候选。"""

    # 父块唯一标识
    parent_id: str
    # 该父块对应子块里分数最高那条的 payload（包含 parent_text / doc_name / breadcrumb 等）
    payload: dict[str, Any]
    # 命中子块中分数最高的 chunk_id（用于 citation 跳转）
    best_chunk_id: str
    # 向量分数（取该父块下所有命中子块的最大值）
    score: float
    # 命中的子块 chunk_id 列表（未去重，后续组装时会排序去重）
    matched_chunks: list[str] = field(default_factory=list)
    # CrossEncoder rerank 分数；None 表示尚未 rerank 或 rerank 降级
    rerank_score: float | None = None


def _group_by_parent(hits: Iterable[KbSearchHit]) -> list[_Candidate]:
    grouped: dict[str, _Candidate] = {}
    for hit in hits:
        parent_id = str(hit.parent_id or hit.payload.get("parent_id") or "")
        if not parent_id:
            continue
        candidate = grouped.get(parent_id)
        if candidate is None:
            grouped[parent_id] = _Candidate(
                parent_id=parent_id,
                payload=dict(hit.payload),
                best_chunk_id=hit.chunk_id,
                score=float(hit.score),
                matched_chunks=[hit.chunk_id],
            )
            continue
        candidate.matched_chunks.append(hit.chunk_id)
        if hit.score > candidate.score:
            candidate.score = float(hit.score)
            candidate.best_chunk_id = hit.chunk_id
            candidate.payload = dict(hit.payload)
    return sorted(grouped.values(), key=lambda item: item.score, reverse=True)


# ==================== Rerank ====================


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class RerankError(RuntimeError):
    """Rerank 模型不可用或输出非法。"""


class CrossEncoderReranker:
    """CrossEncoder 重排，懒加载；失败时由调用方降级到向量分数。

    两种后端：
    - local：本机加载 sentence-transformers CrossEncoder 模型推理；
    - http：调用服务器上的 rerank HTTP API（类似 Embedding 的 http 后端）。
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_length: int = 512,
        batch_size: int = 16,
        device: str | None = None,
        *,
        backend: str | None = None,
        api_url: str | None = None,
        api_path: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """懒加载 CrossEncoder 重排器。

        Args:
            model_name: 模型名或本地路径，None 时取 settings.rerank_model_path。
            max_length: CrossEncoder 最大输入长度，默认 512。
            batch_size: 推理批大小，最小为 1。
            device: 推理设备（如 "cuda" / "cpu"），None 时由 sentence-transformers 自选。
            backend: 后端类型，None 时取 settings.rerank_backend（"local" / "http"）。
            api_url: http 后端地址（如 "http://localhost:8092"），None 时取 settings.rerank_api_url。
            api_path: http 后端路径（默认 "/rerank"）。
            api_key: http 后端 API key，可选。
        """
        self.backend = backend or settings.rerank_backend
        self.model_name = model_name or settings.rerank_model_path
        self.max_length = max_length
        self.batch_size = max(1, batch_size)
        self.device = device
        # http 后端连接参数
        self.api_url = (api_url if api_url is not None else settings.rerank_api_url).rstrip("/")
        self.api_path = (api_path if api_path is not None else settings.rerank_api_path) or "/rerank"
        self.api_key = api_key if api_key is not None else settings.rerank_api_key

        self._model: Any | None = None
        self._http_ready: bool = False
        self._lock = threading.RLock()

    def _ensure_model(self) -> None:
        """local 后端：加载本地 CrossEncoder 模型；http 后端：探测 API 可达性。"""
        with self._lock:
            if self.backend == "http":
                if self._http_ready:
                    return
                self._ensure_http_backend()
                return
            if self._model is not None:
                return
            log.info(f"加载 Rerank 模型: {self.model_name}")
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self.model_name,
                    max_length=self.max_length,
                    device=self.device,
                )
            except Exception as exc:
                raise RerankError(
                    f"Rerank 模型加载失败（{self.model_name}）：{type(exc).__name__}: {exc}"
                ) from exc

    def _ensure_http_backend(self) -> None:
        """http 后端：不加载本地模型，只探测 API 可达性（fail-fast）。"""
        if not self.api_url:
            raise RerankError(
                "rerank_backend=http 但未配置 rerank_api_url，"
                "请在 .env 里填服务器 rerank 服务地址"
            )
        endpoint = f"{self.api_url}{self.api_path}"
        log.info(f"检查 HTTP Rerank 服务: {endpoint}")
        from urllib.error import HTTPError, URLError
        from urllib import request as _req

        probe = _req.Request(endpoint, method="GET")
        try:
            with _req.urlopen(probe, timeout=8) as resp:
                resp.read(256)
        except HTTPError as exc:
            if not (400 <= exc.code <= 499):
                raise RerankError(
                    f"HTTP Rerank 服务异常（{endpoint}）：HTTP {exc.code}"
                ) from exc
        except (URLError, OSError) as exc:
            raise RerankError(
                f"HTTP Rerank 服务不可达（{endpoint}）："
                f"{type(exc).__name__}: {exc}，请检查 SSH 隧道是否已转发该端口"
            ) from exc
        self._http_ready = True
        log.info(f"HTTP Rerank 服务就绪（{endpoint}）")

    def _http_score(self, query: str, texts: list[str]) -> list[float]:
        """POST {api_url}{api_path}，请求体 {"query": "...", "documents": [...]}，
        返回服务器排序后的 items 列表，按原始 index 还原成 scores 列表。"""
        import json as _json

        from urllib.error import URLError, HTTPError
        from urllib import request as _req

        endpoint = f"{self.api_url}{self.api_path}"
        # 服务器 schema: {"query": str, "documents": str[], "top_n": int?}
        payload = _json.dumps(
            {"query": query, "documents": list(texts)}, ensure_ascii=False
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        req = _req.Request(endpoint, data=payload, headers=headers, method="POST")
        try:
            with _req.urlopen(req, timeout=60) as resp:
                raw = resp.read()
        except (URLError, OSError, HTTPError) as exc:
            raise RerankError(
                f"HTTP Rerank 调用失败（{endpoint}）：{type(exc).__name__}: {exc}"
            ) from exc

        try:
            data = _json.loads(raw)
        except Exception as exc:
            raise RerankError(
                f"HTTP Rerank 响应解析失败（{endpoint}）：{type(exc).__name__}"
            ) from exc

        # 服务器返回格式:
        # {"items": [{"index": 0, "score": 3.89, "text": "..."}], "model": "...", "total": 3}
        # 注意：items 是按 score 降序排的，不是按原始 index 顺序。
        # 需要还原成和 texts 同长度、同下标的 scores 列表。
        if isinstance(data, dict) and "items" in data:
            scores_by_index: dict[int, float] = {
                int(item["index"]): float(item["score"])
                for item in data["items"]
            }
            scores = [scores_by_index.get(i, 0.0) for i in range(len(texts))]
        elif isinstance(data, dict) and "scores" in data:
            scores = data["scores"]
        elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            scores = [item.get("score", 0.0) for item in data["data"]]
        elif isinstance(data, list):
            scores = data
        else:
            raise RerankError(
                f"HTTP Rerank 响应缺少 items/scores 字段：{raw[:120]!r}"
            )

        if len(scores) != len(texts):
            raise RerankError(
                f"HTTP Rerank 返回 {len(scores)} 条，期望 {len(texts)} 条"
            )
        # 远程 rerank API 返回的也是 raw logit，统一过 sigmoid 变成 0~1 概率
        import math
        return [1.0 / (1.0 + math.exp(-float(s))) for s in scores]

    def score(self, query: str, texts: list[str]) -> list[float]:
        """对 query 与 texts 做 CrossEncoder 重排打分。

        两种后端：local 用本地模型推理，http 调服务器 API。
        """
        if not texts:
            return []
        self._ensure_model()

        if self.backend == "http":
            return self._http_score(query, texts)

        try:
            raw = self._model.predict(
                [[query, text] for text in texts],
                batch_size=self.batch_size,
                show_progress_bar=False,
                apply_sigmoid=True,  # 输出 0~1 概率而非 raw logit
            )
            return [float(value) for value in raw]
        except Exception as exc:
            raise RerankError(f"Rerank 打分失败：{type(exc).__name__}: {exc}") from exc


_reranker: CrossEncoderReranker | None = None
_reranker_lock = threading.Lock()


def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = CrossEncoderReranker()
    return _reranker


# ==================== 检索 ====================


class Retriever:
    """知识库检索主链路：清洗 → 编码 → 硬过滤检索 → 软过滤兜底 → 父块归并 → rerank → 上下文组装。"""

    def __init__(
        self,
        store: KbStore | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        """注入检索依赖。

        Args:
            store: 向量存储，None 时自动调用 get_kb_store() 初始化 Qdrant 连接。
            embedder: 查询编码器，None 时自动调用 get_embedder() 初始化 BGE。
            reranker: CrossEncoder 重排器，None 时 retrieve 内部不会 rerank。
        """
        self.store = store or get_kb_store()
        self.embedder = embedder or get_embedder()
        self.reranker = reranker

    @staticmethod
    def _chunk_type_condition(chunk_types: list[str]) -> list[models.FieldCondition]:
        """把 chunk_type 字符串列表转成 Qdrant 的 MatchAny 软过滤条件。

        Args:
            chunk_types: ["faq", "doc_section"] 之类的类型列表。
        Returns:
            传给 KbStore.search 的 extra_must 字段条件列表。
        """
        return [
            models.FieldCondition(
                key="chunk_type",
                match=models.MatchAny(any=list(chunk_types)),
            )
        ]

    def _search(
        self,
        vector: list[float],
        request: RetrieveRequest,
        vector_threshold: float | None,
    ) -> tuple[list[KbSearchHit], bool]:
        """单次向量检索 + 软过滤兜底。

        Args:
            vector: 查询的 embedding 向量。
            request: 用户的 RetrieveRequest，用到 kb_ids / agent_role / tenant_id / top_k 等。
            vector_threshold: 向量分数阈值；None 时不限。

        Returns:
            (hits, used_fallback) — used_fallback=True 表示本次检索触发了 chunk_type 兜底，
            即先按 chunk_types 过滤命中不足，又去掉 chunk_types 重新搜了一次。
        """
        extra = (
            self._chunk_type_condition(request.chunk_types)
            if request.chunk_types
            else None
        )
        hits = self.store.search(
            vector,
            request.kb_ids,
            request.agent_role,
            tenant_id=request.tenant_id,
            top_k=request.top_k,
            score_threshold=vector_threshold,
            extra_must=extra,
        )
        if not extra or not request.allow_filter_fallback:
            return hits, False
        if len(hits) >= max(1, request.min_candidates):
            return hits, False

        # 软过滤把召回压得太狠：去掉 chunk_type 重试，宁可多召回再 rerank
        fallback = self.store.search(
            vector,
            request.kb_ids,
            request.agent_role,
            tenant_id=request.tenant_id,
            top_k=request.top_k,
            score_threshold=vector_threshold,
        )
        log.info(
            f"chunk_type 软过滤仅命中 {len(hits)} 条，已回退到全类型检索（{len(fallback)} 条）"
        )
        return fallback, True

    def retrieve(self, request: RetrieveRequest) -> RetrieveResult:
        """知识库检索主入口。

        完整链路：
        1. query 清洗 + 变体构建；
        2. 多路编码（query + query_variants）→ 每路向量检索；
        3. 多路 RRF 融合 → 父块归并；
        4. 可选 rerank（CrossEncoder，失败降级）；
        5. 阈值过滤（rerank_threshold 或 score_threshold）；
        6. 父块组装 + token 预算截断 + citation 生成。

        Args:
            request: RetrieveRequest，包含 query、kb_ids、chunk_types、阈值等全部控制参数。

        Returns:
            RetrieveResult：包含组装好的 contexts、命中状态、诊断信息。
            当无可靠命中时，contexts 为空，no_hit=True，no_hit_reason 给出具体原因。
        """
        started = time.perf_counter()
        cleaned = clean_query(request.query)
        variants = _build_variants(request, cleaned)
        diagnostics: dict[str, Any] = {
            "hard_filters": {
                "kb_ids": list(request.kb_ids),
                "agent_role": request.agent_role,
                "tenant_id": request.tenant_id,
                "status": "active",
            },
            "chunk_types": list(request.chunk_types or []),
            "filter_fallback": False,
            "variant_count": len(variants),
            "candidates": 0,
            "parents": 0,
            "rerank": "disabled",
            "timings_ms": {},
        }

        if not cleaned:
            return RetrieveResult(
                query=request.query,
                cleaned_query="",
                variants=[],
                contexts=[],
                no_hit=True,
                no_hit_reason="empty_query",
                diagnostics=diagnostics,
            )
        if not request.kb_ids:
            return RetrieveResult(
                query=request.query,
                cleaned_query=cleaned,
                variants=variants,
                contexts=[],
                no_hit=True,
                no_hit_reason="empty_kb_ids",
                diagnostics=diagnostics,
            )

        rerank_active = request.enable_rerank and self.reranker is not None
        vector_threshold = None if rerank_active else request.score_threshold

        # 1. 查询编码（BGE 指令前缀只在 query 侧，可按请求 A/B）
        embed_started = time.perf_counter()
        if request.query_instruction is None:
            encoded = self.embedder.encode_query(variants)
            diagnostics["query_instruction"] = (
                getattr(self.embedder, "query_instruction", "") or "(disabled)"
            )
        else:
            encoded = self.embedder.encode(
                variants, instruction=request.query_instruction
            )
            diagnostics["query_instruction"] = request.query_instruction or "(disabled)"
        diagnostics["timings_ms"]["embed"] = round(
            (time.perf_counter() - embed_started) * 1000, 1
        )

        # 2. 多路检索 + 软过滤兜底
        search_started = time.perf_counter()
        search_results: list[list[KbSearchHit]] = []
        fallback_used = False
        for vector in encoded.vectors:
            hits, used_fallback = self._search(vector, request, vector_threshold)
            search_results.append(hits)
            fallback_used = fallback_used or used_fallback
        diagnostics["timings_ms"]["search"] = round(
            (time.perf_counter() - search_started) * 1000, 1
        )
        diagnostics["filter_fallback"] = fallback_used

        hits = (
            search_results[0]
            if len(search_results) == 1
            else _rrf_fuse(search_results)
        )
        diagnostics["candidates"] = len(hits)

        # 3. 按父块归并，去掉同一父块的重复子片
        candidates = _group_by_parent(hits)
        diagnostics["parents"] = len(candidates)
        if not candidates:
            diagnostics["timings_ms"]["total"] = round(
                (time.perf_counter() - started) * 1000, 1
            )
            return RetrieveResult(
                query=request.query,
                cleaned_query=cleaned,
                variants=variants,
                contexts=[],
                no_hit=True,
                no_hit_reason="no_candidates",
                diagnostics=diagnostics,
            )

        # 4. rerank（可选，失败降级）
        if rerank_active:
            rerank_started = time.perf_counter()
            rerank_texts = [self._rerank_text(item.payload) for item in candidates]
            try:
                scores = self.reranker.score(cleaned, rerank_texts)
                if len(scores) != len(candidates):
                    raise RerankError("rerank 分数数量与候选数量不一致")
                for candidate, score in zip(candidates, scores):
                    candidate.rerank_score = float(score)
                candidates.sort(
                    key=lambda item: item.rerank_score or 0.0, reverse=True
                )
                diagnostics["rerank"] = "applied"
            except Exception as exc:
                log.warning(f"rerank 失败，降级为向量分数：{exc}")
                diagnostics["rerank"] = f"failed: {type(exc).__name__}"
            diagnostics["timings_ms"]["rerank"] = round(
                (time.perf_counter() - rerank_started) * 1000, 1
            )

        best_candidate = candidates[0]
        if request.rerank_threshold is not None and best_candidate.rerank_score is not None:
            if best_candidate.rerank_score < request.rerank_threshold:
                diagnostics["timings_ms"]["total"] = round(
                    (time.perf_counter() - started) * 1000, 1
                )
                return RetrieveResult(
                    query=request.query,
                    cleaned_query=cleaned,
                    variants=variants,
                    contexts=[],
                    no_hit=True,
                    no_hit_reason="below_rerank_threshold",
                    diagnostics=diagnostics,
                )
        elif request.score_threshold is not None and not rerank_active:
            if best_candidate.score < request.score_threshold:
                diagnostics["timings_ms"]["total"] = round(
                    (time.perf_counter() - started) * 1000, 1
                )
                return RetrieveResult(
                    query=request.query,
                    cleaned_query=cleaned,
                    variants=variants,
                    contexts=[],
                    no_hit=True,
                    no_hit_reason="below_score_threshold",
                    diagnostics=diagnostics,
                )

        # 5. 父块组装 + token 预算 + 引用
        contexts = self._assemble(candidates, request)
        diagnostics["timings_ms"]["total"] = round(
            (time.perf_counter() - started) * 1000, 1
        )
        diagnostics["best_score"] = round(best_candidate.score, 4)
        if best_candidate.rerank_score is not None:
            diagnostics["best_rerank_score"] = round(best_candidate.rerank_score, 4)

        return RetrieveResult(
            query=request.query,
            cleaned_query=cleaned,
            variants=variants,
            contexts=contexts,
            no_hit=not contexts,
            no_hit_reason=None if contexts else "empty_context",
            diagnostics=diagnostics,
        )

    @staticmethod
    def _rerank_text(payload: dict[str, Any]) -> str:
        """从 payload 中抽取用于 CrossEncoder 重排的文本。

        FAQ 类型用 "问题 + 答案摘要"（补回向量阶段丢掉的 answer 信息）；
        文档类型用 parent_text（或兜底 text），截断到 1200 字符以内，避免拖慢 rerank。

        Args:
            payload: 检索命中子块的 payload 字典，包含 chunk_type / answer / parent_text 等。
        Returns:
            用于 rerank 的纯文本，不含 JSON 结构，方便 CrossEncoder 做相关性打分。
        """
        if payload.get("chunk_type") == "faq":
            answer = str(payload.get("answer") or "")
            return f"{payload.get('text', '')}\n{answer[:300]}"
        text = str(payload.get("parent_text") or payload.get("text") or "")
        return text[:1200]

    def _assemble(
        self,
        candidates: list[_Candidate],
        request: RetrieveRequest,
    ) -> list[RetrievedContext]:
        """把排序后的候选父块组装成 RetrievedContext 列表。

        组装过程：
        - 最多取 rerank_top_n 个候选；
        - 依次生成 header（文档名 + 章节 + 位置）；
        - 按 token_budget 从前往后累加，剩余空间不足时提前停止；
        - 正文按剩余 token 空间二分截断，不整块丢弃；
        - 同时生成 Citation 元数据。

        Args:
            candidates: 已排序的 _Candidate 列表（rerank 或向量分数降序）。
            request: RetrieveRequest，用到 rerank_top_n / token_budget。

        Returns:
            组装好的 RetrievedContext 列表，长度 ≤ rerank_top_n，累计 token ≤ token_budget。
        """
        contexts: list[RetrievedContext] = []
        used = 0
        for candidate in candidates:
            if len(contexts) >= max(1, request.rerank_top_n):
                break
            payload = candidate.payload
            parent_text = str(
                payload.get("parent_text") or payload.get("text") or ""
            ).strip()
            if not parent_text:
                continue

            index = len(contexts) + 1
            breadcrumb = [str(x) for x in payload.get("breadcrumb", [])]
            loc = payload.get("loc") or {}
            header = self._header(index, payload, breadcrumb, loc)
            header_tokens = estimate_tokens(header)
            remaining = request.token_budget - used - header_tokens - 2
            if remaining <= 0:
                break

            body, truncated = _fit_text(parent_text, remaining)
            if not body:
                break

            contexts.append(
                RetrievedContext(
                    parent_id=candidate.parent_id,
                    header=header,
                    text=body,
                    citation=Citation(
                        index=index,
                        doc_id=str(payload.get("doc_id", "")),
                        doc_name=str(payload.get("doc_name", "")),
                        biz_category=str(payload.get("biz_category", "")),
                        biz_category_name=str(payload.get("biz_category_name", "")),
                        chunk_id=candidate.best_chunk_id,
                        parent_id=candidate.parent_id,
                        breadcrumb=breadcrumb,
                        loc=dict(loc),
                        source_path=str(payload.get("source_path", "")),
                        score=round(float(candidate.score), 6),
                        rerank_score=(
                            round(float(candidate.rerank_score), 6)
                            if candidate.rerank_score is not None
                            else None
                        ),
                    ),
                    matched_chunks=sorted(set(candidate.matched_chunks)),
                    score=float(candidate.score),
                    rerank_score=candidate.rerank_score,
                    tokens=header_tokens + estimate_tokens(body),
                    truncated=truncated,
                )
            )
            used += contexts[-1].tokens + 2
        return contexts

    @staticmethod
    def _header(
        index: int,
        payload: dict[str, Any],
        breadcrumb: list[str],
        loc: dict[str, Any],
    ) -> str:
        """为单条上下文生成头部信息（文档名 / 章节路径 / 位置）。

        格式："[1] 文档：xxx｜章节：FAQ / 账户｜位置：page=3, row=15"。
        只拼非空字段，避免噪音。

        Args:
            index: 该上下文在列表中的序号（从 1 开始）。
            payload: 命中子块的 payload，从中读 doc_name / doc_id。
            breadcrumb: 文档内层级路径，如 ["FAQ", "账户"]。
            loc: 文档内位置，如 {page: 3, row: 15}。

        Returns:
            纯文本 header，不含 markdown 格式，直接拼在正文前面。
        """
        parts = [f"[{index}] 文档：{payload.get('doc_name') or payload.get('doc_id') or '-'}"]
        if breadcrumb:
            parts.append("章节：" + " / ".join(breadcrumb))
        loc_text = _loc_text(loc)
        if loc_text:
            parts.append("位置：" + loc_text)
        return "｜".join(parts)


def _loc_text(loc: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("page", "sheet", "row"):
        if key in loc and loc[key] is not None:
            parts.append(f"{key}={loc[key]}")
    return ", ".join(parts)


def _fit_text(text: str, budget: int) -> tuple[str, bool]:
    """按 token 预算二分截断，保证不整块丢弃。"""
    if budget <= 0:
        return "", True
    if estimate_tokens(text) <= budget:
        return text, False
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if estimate_tokens(text[:mid]) <= budget:
            low = mid
        else:
            high = mid - 1
    return text[: max(1, low)].rstrip() + "…", True


# ==================== CLI ====================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="知识库检索（过滤 + 父块 + 引用）")
    parser.add_argument("query", help="用户原始问题")
    parser.add_argument("--kb", action="append", required=True, help="知识库 ID，可重复")
    parser.add_argument(
        "--agent-role",
        required=True,
        help="逻辑 Agent 角色（如 intent_agent），不是人工客服会话 ID",
    )
    parser.add_argument("--tenant", default="default", help="租户 ID")
    parser.add_argument(
        "--chunk-type",
        action="append",
        default=None,
        help="软过滤类型：faq / doc_section / table / text，可重复",
    )
    parser.add_argument("--variant", action="append", default=None, help="改写/子问题")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--rerank-top-n", type=int, default=6)
    parser.add_argument("--token-budget", type=int, default=2400)
    parser.add_argument("--min-candidates", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--rerank-threshold", type=float, default=None)
    instruction = parser.add_mutually_exclusive_group()
    instruction.add_argument(
        "--query-instruction", default=None, help="自定义查询指令（A/B 用）"
    )
    instruction.add_argument(
        "--no-query-instruction",
        action="store_true",
        help="关闭 BGE 查询前缀（FAQ 问题对问题场景可对比）",
    )
    parser.add_argument("--rerank", action="store_true", help="启用本地 CrossEncoder 重排")
    parser.add_argument("--json", action="store_true", help="输出完整 JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    request = RetrieveRequest(
        query=args.query,
        agent_role=args.agent_role,
        kb_ids=args.kb,
        tenant_id=args.tenant,
        chunk_types=args.chunk_type,
        query_variants=args.variant or [],
        query_instruction="" if args.no_query_instruction else args.query_instruction,
        top_k=args.top_k,
        rerank_top_n=args.rerank_top_n,
        token_budget=args.token_budget,
        min_candidates=args.min_candidates,
        enable_rerank=args.rerank,
        score_threshold=args.score_threshold,
        rerank_threshold=args.rerank_threshold,
    )
    retriever = Retriever(reranker=get_reranker() if args.rerank else None)
    result = retriever.retrieve(request)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    elif result.no_hit:
        print(f"无可靠命中：{result.no_hit_reason}")
    else:
        print(result.context_text())
        print("\n--- diagnostics ---")
        print(json.dumps(result.diagnostics, ensure_ascii=False, indent=2))
    return 0 if result.contexts else 1


if __name__ == "__main__":
    raise SystemExit(main())
