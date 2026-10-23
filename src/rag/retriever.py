"""
RAG 检索器 —— 把 Embedding + Qdrant 串起来，返回可直接拼进 prompt 的上下文
"""
from __future__ import annotations
from src.core.logging import log
from src.rag.vector_store import get_vector_store


class RAGRetriever:
    """RAG 检索封装"""

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.3) -> str:
        """
        检索相关知识并格式化为 LLM prompt 可直接使用的上下文文本

        返回空字符串表示没有找到有用的知识
        """
        try:
            vs = get_vector_store()
            total = vs.count()
            if total == 0:
                log.warning("⚠️ Qdrant collection 为空，请先初始化 FAQ 数据")
                return ""

            hits = vs.search(query, top_k=top_k)

            # 过滤低分结果
            good_hits = [h for h in hits if h["score"] >= min_score]
            if not good_hits:
                log.debug(f"🔍 RAG 未找到足够相关的知识（最高分: {hits[0]['score'] if hits else 0}）")
                return ""

            # 格式化
            lines = ["【知识库相关知识】"]
            for h in good_hits:
                source_tag = h.get("source", "faq")
                lines.append(f"- [{source_tag}] {h['text']} (相关度: {h['score']:.2f})")

            context = "\n".join(lines)
            log.info(f"🔍 RAG 检索到 {len(good_hits)} 条知识 (最高分: {good_hits[0]['score']:.2f})")
            return context

        except Exception as e:
            log.error(f"❌ RAG 检索异常: {e}")
            return ""

    def augment_messages(
        self,
        messages: list[dict],
        user_text: str,
        system_prompt: str,
    ) -> list[dict]:
        """
        把 RAG 检索结果注入到 system prompt 里
        """
        context = self.retrieve(user_text)
        if not context:
            return messages

        augmented_prompt = f"{system_prompt}\n\n{context}\n\n（请优先参考以上知识回答用户问题）"
        messages = list(messages)
        messages[0] = {"role": "system", "content": augmented_prompt}
        return messages


_retriever: RAGRetriever | None = None


def get_rag_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever
