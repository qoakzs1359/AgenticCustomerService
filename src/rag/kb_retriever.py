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
    query: str
    agent_role: str
    kb_ids: list[str]
    tenant_id: str = "default"
    # 软过滤：faq / doc_section / table / text，不是硬约束
    chunk_types: list[str] | None = None
    # 调用方（Agent/LLM）给出的改写或子问题，原 query 永远保留
    query_variants: list[str] = field(default_factory=list)
    max_variants: int = 3
    # None=用 Embedder 默认指令；""=关闭；也可传自定义指令做 A/B
    query_instruction: str | None = None
    top_k: int = 40
    rerank_top_n: int = 6
    token_budget: int = 2400
    min_candidates: int = 5
    enable_rerank: bool = True
    allow_filter_fallback: bool = True
    score_threshold: float | None = None
    rerank_threshold: float | None = None


@dataclass(slots=True)
class Citation:
    index: int
    doc_id: str
    doc_name: str
    chunk_id: str
    parent_id: str
    breadcrumb: list[str]
    loc: dict[str, Any]
    source_path: str
    score: float
    rerank_score: float | None = None
    biz_category: str = ""
    biz_category_name: str = ""


@dataclass(slots=True)
class RetrievedContext:
    parent_id: str 
    header: str
    text: str
    citation: Citation
    matched_chunks: list[str]
    score: float
    rerank_score: float | None
    tokens: int
    truncated: bool


@dataclass(slots=True)
class RetrieveResult:
    query: str
    cleaned_query: str
    variants: list[str]
    contexts: list[RetrievedContext]
    no_hit: bool
    no_hit_reason: str | None
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
    parent_id: str
    payload: dict[str, Any]
    best_chunk_id: str
    score: float
    matched_chunks: list[str] = field(default_factory=list)
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
    """CrossEncoder 重排，懒加载；失败时由调用方降级到向量分数。"""

    def __init__(
        self,
        model_name: str | None = None,
        max_length: int = 512,
        batch_size: int = 16,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.rerank_model_path
        self.max_length = max_length
        self.batch_size = max(1, batch_size)
        self.device = device
        self._model: Any | None = None
        self._lock = threading.RLock()

    def _ensure_model(self) -> None:
        with self._lock:
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

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        self._ensure_model()
        try:
            raw = self._model.predict(
                [[query, text] for text in texts],
                batch_size=self.batch_size,
                show_progress_bar=False,
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
    def __init__(
        self,
        store: KbStore | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.store = store or get_kb_store()
        self.embedder = embedder or get_embedder()
        self.reranker = reranker

    @staticmethod
    def _chunk_type_condition(chunk_types: list[str]) -> list[models.FieldCondition]:
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
        """FAQ 用“问题+答案摘要”，文档用父块正文，补回向量阶段丢掉的信息。"""
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
