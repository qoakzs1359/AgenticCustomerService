"""
证据聚合 —— 检索结果 → LLM 审视材料

职责：
- 调用 kb_retriever 检索意图样本库（kb_intent_v1）；
- 把 RetrieveResult.contexts 按 l2_code 聚类，聚合成「审视材料」：
  每个候选意图给 {l2_code, l2_name, 最多 2 条片段原文}；
- LLM 按判定方法第 2 步「带怀疑地审视」这些片段（含语义相近干扰项），
  判断哪条真正匹配用户诉求，最终结论引用 evidence_refs；
- chunk_ids / samples 完整信息保留在 trace()，供 badcase 追溯。

用法：
    evidence = builder.build("我的快递三天没更新了")
    result = await classifier.classify(user_text, evidence=evidence)
"""
from __future__ import annotations

from collections import defaultdict

from src.agent.intent_llm_prompt import build_context_block
from src.rag.kb_retriever import RetrieveRequest, RetrieveResult, Retriever

# 意图样本库：kb.py 入库时使用的常量
DEFAULT_KB_ID = "kb_intent_v1"
DEFAULT_AGENT_ROLE = "intent_agent"
DEFAULT_CHUNK_TYPE = ["intent_sample"]


class Evidence:
    """一条证据：某 L2 候选在检索中的命中汇总

    to_dict() 输出给 LLM 的审视材料：l2_code + l2_name + 片段原文(带 L2 标签，
    供 LLM 按「判定方法第 2 步」带怀疑地审视，而非盲目采纳)；
    chunk_ids 保留在 trace()，供 badcase 追溯（不进 prompt）。
    """

    __slots__ = ("idx", "l2_code", "l2_name", "hits", "top_score", "samples", "chunk_ids")

    def __init__(self, idx: int, l2_code: str, l2_name: str = ""):
        self.idx = idx
        self.l2_code = l2_code
        self.l2_name = l2_name
        self.hits = 0
        self.top_score = 0.0
        self.samples: list[str] = []
        self.chunk_ids: list[str] = []

    def add(self, text: str, score: float, chunk_id: str) -> None:
        self.hits += 1
        self.top_score = max(self.top_score, score)
        if len(self.samples) < 2:
            self.samples.append(text)
        if len(self.chunk_ids) < 8:
            self.chunk_ids.append(chunk_id)

    def to_dict(self, idx: int, *, include_trace: bool = False) -> dict:
        base = {
            "idx": idx,
            "l2_code": self.l2_code,
            "l2_name": self.l2_name,
            "samples": self.samples,
        }
        if include_trace:
            base.update(self.trace())
        return base

    def trace(self) -> dict:
        """追溯专用完整信息（日志/评估用，不进 prompt）"""
        return {
            "l2_code": self.l2_code,
            "hits": self.hits,
            "top_score": round(self.top_score, 4),
            "samples": self.samples,
            "chunk_ids": self.chunk_ids,
        }


class EvidenceBuilder:
    """检索 → 证据聚合器"""

    def __init__(
        self,
        kb_id: str = DEFAULT_KB_ID,
        agent_role: str = DEFAULT_AGENT_ROLE,
        top_k: int = 40,
        retriever: Retriever | None = None,
    ) -> None:
        self.kb_id = kb_id
        self.agent_role = agent_role
        self.top_k = top_k
        self.retriever = retriever

    # ====== 核心：检索 → 证据列表 ======

    def build(
        self,
        query: str,
        *,
        enable_rerank: bool = False,
        include_trace: bool = False,
    ) -> list[dict]:
        """检索意图样本库并按 L2 聚合，返回证据摘要列表（按命中数排序，最多 4 条）。

        Args:
            query: 用户问题
            enable_rerank: 是否启用 rerank 重排
            include_trace: 是否在返回的 dict 里带上追溯信息（hits/top_score/chunk_ids），默认 False 不进 prompt
        """
        result = self._retrieve(query, enable_rerank=enable_rerank)
        return self.aggregate(result, include_trace=include_trace)

    def retrieve_raw(self, query: str, *, enable_rerank: bool = False) -> RetrieveResult:
        """直接返回原始检索结果（含每个片段的向量 score），供调试/评估用。"""
        return self._retrieve(query, enable_rerank=enable_rerank)

    def _retrieve(self, query: str, *, enable_rerank: bool) -> RetrieveResult:
        """复用新链路 kb_retriever（只查意图样本库）。"""
        if self.retriever is None:
            from src.rag.kb_retriever import Retriever

            self.retriever = Retriever()
        request = RetrieveRequest(
            query=query,
            agent_role=self.agent_role,
            kb_ids=[self.kb_id],
            chunk_types=DEFAULT_CHUNK_TYPE,
            top_k=self.top_k,
            enable_rerank=enable_rerank,
        )
        return self.retriever.retrieve(request)

    def aggregate(self, result: RetrieveResult, *, include_trace: bool = False) -> list[dict]:
        """把 RetrieveResult.contexts 按 L2 编码聚类 → 证据摘要。

        L2 取自 Citation.biz_category_name（kb.py 入库时 = 二级类别编码）。
        no_hit 时返回空列表（LLM 走无证据分支，不编造）。
        """
        if result.no_hit or not result.contexts:
            return []

        buckets: dict[str, Evidence] = {}
        for ctx in result.contexts:
            citation = getattr(ctx, "citation", None)
            if citation is None:
                continue
            # 入库时 payload.biz_category_name = L2 编码（见 src/intent/kb.py）
            l2 = str(citation.biz_category_name or "").strip().upper()
            if not l2:
                l2 = str(citation.biz_category or "unknown").strip().upper()
            bucket = buckets.setdefault(l2, Evidence(idx=0, l2_code=l2, l2_name=l2))
            bucket.add(getattr(ctx, "text", ""), float(ctx.score), citation.chunk_id)

        # 按命中条数排序（并列按最高分），取前 4
        ordered = sorted(buckets.values(), key=lambda b: (b.hits, b.top_score), reverse=True)[:4]
        return [b.to_dict(i + 1, include_trace=include_trace) for i, b in enumerate(ordered)]

    # ====== 便捷输出 ======

    def build_block(self, query: str, *, enable_rerank: bool = False) -> str:
        """直接得到可注入 prompt 的证据文本（调试/CLI 用）。"""
        return build_context_block(self.build(query, enable_rerank=enable_rerank))


# ====== 单例 ======
_builder_instance: EvidenceBuilder | None = None


def get_evidence_builder() -> EvidenceBuilder:
    global _builder_instance
    if _builder_instance is None:
        _builder_instance = EvidenceBuilder()
    return _builder_instance


if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "我的快递三天没更新了，是不是丢件了？"
    builder = get_evidence_builder()
    
    # 1. 原始检索结果（带每个片段的向量 score）
    raw = builder.retrieve_raw(query)
    print(f"\n🔍 原始检索结果（共 {len(raw.contexts)} 个片段）:")
    print(f"   最佳向量 score: {raw.diagnostics.get('best_score', 'N/A')}")
    print(f"   候选数: {raw.diagnostics.get('candidates', 'N/A')}  父块数: {raw.diagnostics.get('parents', 'N/A')}")
    print(f"   耗时(ms): 嵌入={raw.diagnostics.get('timings_ms', {}).get('embed', 'N/A')} 搜索={raw.diagnostics.get('timings_ms', {}).get('search', 'N/A')}")
    for i, ctx in enumerate(raw.contexts[:6], 1):  # 只打印前6个
        l2 = ctx.citation.biz_category_name or "unknown"
        print(f"   [{i}] score={ctx.score:.4f}  L2={l2.upper()}  片段前50字: {ctx.text[:50]}...")
    
    # 2. 按 L2 聚合的证据（带 trace 信息）
    evs = builder.build(query, include_trace=True)
    print(f"\n📊 按 L2 聚合的证据（top {len(evs)} 组）:")
    for e in evs:
        print(f"   [{e['idx']}] L2={e['l2_code']}  命中数={e['hits']}  top_score={e['top_score']:.4f}")
        print(f"       样本片段: {e['samples'][0][:60]}..." if e['samples'] else "       （无样本）")
    
    # 3. 给 LLM 的 prompt 注入文本（不带 trace）
    evs_for_prompt = builder.build(query)
    print(f"\n📝 给 LLM 的 prompt 证据块:")
    print(build_context_block(evs_for_prompt))