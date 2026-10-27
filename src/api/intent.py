"""
意图识别 API —— 对外暴露意图判定服务

路由：
  POST /api/intent/detect  完整链路（向量检索 → LLM 判定）
  POST /api/intent/classify 仅 LLM 判定（传入已检索好的 evidence）
  GET  /api/intent/health  健康检查（检查 DeepSeek / embedding / qdrant）
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any

router = APIRouter(prefix="/api/intent", tags=["意图识别"])


# ====== 请求/响应模型 ======


class DetectRequest(BaseModel):
    user_text: str = Field(..., min_length=1, description="用户原始问题")


class ClassifyRequest(BaseModel):
    user_text: str = Field(..., min_length=1, description="用户原始问题")
    evidence: list[dict[str, Any]] | None = Field(
        default=None,
        description="外部已检索好的证据（传入则跳过向量检索）",
    )


class DetectResponse(BaseModel):
    l1: str
    l2: str
    l3: str = ""
    confidence: float
    confidence_breakdown: dict[str, float] = {}
    reasoning: str = ""
    evidence_refs: list[int] = []
    need_clarify: bool = False
    clarify_hint: str = ""
    multi: list[str] = []
    source: str
    evidence_count: int = 0
    server_time: str = ""  # 北京时间


class RetrieveDebugRequest(BaseModel):
    """检索链路调试 —— 直接透传 RetrieveRequest 所有可调参数,返回完整 diagnostics。"""

    query: str = Field(..., min_length=1, description="用户原始问题")
    kb_ids: list[str] = Field(default_factory=lambda: ["kb_intent_v1"])
    agent_role: str = "intent_agent"
    tenant_id: str = "default"
    chunk_types: list[str] | None = Field(default=None, description="软过滤类型,如 ['intent_sample'] 或 ['faq']")
    query_variants: list[str] = Field(default_factory=list, description="改写/子问题,多路召回")
    max_variants: int = 3
    # None=用 Embedder 默认指令;""=关闭;传自定义字符串做 A/B
    query_instruction: str | None = Field(default=None, description="BGE 查询指令前缀,空串关闭,None 用默认")
    top_k: int = 40
    rerank_top_n: int = 6
    token_budget: int = 2400
    min_candidates: int = 5
    enable_rerank: bool = True
    allow_filter_fallback: bool = True
    score_threshold: float | None = Field(default=None, description="向量相似度阈值")
    rerank_threshold: float | None = Field(default=None, description="Rerank 分数阈值")


# ====== 路由 ======


@router.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    """完整链路：向量检索知识库 → LLM 意图判定"""
    from src.agent.intent_agent import get_intent_agent
    from src.core.timezone import now_str

    agent = get_intent_agent()
    result = await agent.detect(req.user_text)
    return DetectResponse(
        l1=result.l1,
        l2=result.l2,
        l3=result.l3,
        confidence=result.confidence,
        confidence_breakdown=result.confidence_breakdown,
        reasoning=result.reasoning,
        evidence_refs=result.evidence_refs,
        need_clarify=result.need_clarify,
        clarify_hint=result.clarify_hint,
        multi=result.multi,
        source=result.source,
        evidence_count=len(result.evidence or []),
        server_time=now_str(),
    )


@router.post("/classify", response_model=DetectResponse)
async def classify(req: ClassifyRequest):
    """仅 LLM 判定（跳过向量检索，evidence 由外部传入）"""
    from src.agent.intent_agent import get_intent_agent
    from src.core.timezone import now_str

    agent = get_intent_agent()
    result = await agent.classify(req.user_text, evidence=req.evidence)
    return DetectResponse(
        l1=result.l1,
        l2=result.l2,
        l3=result.l3,
        confidence=result.confidence,
        confidence_breakdown=result.confidence_breakdown,
        reasoning=result.reasoning,
        evidence_refs=result.evidence_refs,
        need_clarify=result.need_clarify,
        clarify_hint=result.clarify_hint,
        multi=result.multi,
        source=result.source,
        evidence_count=len(result.evidence or []),
        server_time=now_str(),
    )


@router.post("/retrieve")
async def retrieve(req: RetrieveDebugRequest):
    """检索链路调试 —— 直接调 Retriever,返回完整 diagnostics。

    不经过 LLM 判定,纯看向量检索 + rerank + 阈值过滤的原始效果,
    方便迭代调 top_k / threshold / chunk_type / 指令前缀 等参数。
    """
    from src.rag.kb_retriever import RetrieveRequest, Retriever, get_reranker

    retriever = Retriever(reranker=get_reranker() if req.enable_rerank else None)
    # query_instruction: None 透传给 Embedder 用默认;"" 关闭;自定义字符串透传
    request = RetrieveRequest(
        query=req.query,
        kb_ids=req.kb_ids,
        agent_role=req.agent_role,
        tenant_id=req.tenant_id,
        chunk_types=req.chunk_types,
        query_variants=req.query_variants,
        max_variants=req.max_variants,
        query_instruction=req.query_instruction,
        top_k=req.top_k,
        rerank_top_n=req.rerank_top_n,
        token_budget=req.token_budget,
        min_candidates=req.min_candidates,
        enable_rerank=req.enable_rerank,
        allow_filter_fallback=req.allow_filter_fallback,
        score_threshold=req.score_threshold,
        rerank_threshold=req.rerank_threshold,
    )
    result = retriever.retrieve(request)
    return result.to_dict()


@router.get("/health")
async def health():
    """健康检查：DeepSeek Key / Embedding / Qdrant"""
    from src.core.config import settings
    from src.core.timezone import now_str
    import urllib.request

    checks = {
        "deepseek_api_key": bool(settings.deepseek_api_key),
        "embedding": False,
        "qdrant": False,
    }
    # embedding
    try:
        with urllib.request.urlopen(
            f"{settings.embedding_api_url}/", timeout=3
        ) as resp:
            checks["embedding"] = True
    except Exception:
        pass
    # qdrant
    try:
        with urllib.request.urlopen(
            f"http://{settings.qdrant_host}:{settings.qdrant_port}/collections",
            timeout=3,
        ) as resp:
            checks["qdrant"] = True
    except Exception:
        pass
    return {
        "status": "ok" if all(checks.values()) else "degraded",
        "checks": checks,
        "server_time": now_str(),
    }
