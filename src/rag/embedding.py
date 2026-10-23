"""
Embedding 模型封装 —— bge-base-zh-v1.5 本地加载
如果模型加载失败（比如首次联网超时），会返回 Mock Embedding（全零向量），
让整个 RAG 链路不会因为 Embedding 挂掉而完全不可用
"""
from __future__ import annotations
import numpy as np
from src.core.config import settings
from src.core.logging import log


class EmbeddingService:
    """Embedding 服务（延迟加载，首次调用时才真正加载模型）"""

    def __init__(self):
        self._model = None
        self._dim = settings.embedding_dimension
        self._model_path = settings.embedding_model_path

    def _ensure_model(self):
        if self._model is not None:
            return
        log.info(f"📥 加载 Embedding 模型: {self._model_path}")
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_path)
            # 校验维度
            test_vec = self._model.encode("测试", normalize_embeddings=True)
            if len(test_vec) != self._dim:
                log.warning(f"⚠️ Embedding 维度不匹配: 期望 {self._dim}，实际 {len(test_vec)}，已自动修正 config")
                self._dim = len(test_vec)
            log.info(f"✅ Embedding 模型加载成功，维度 = {self._dim}")
        except Exception as e:
            log.error(f"❌ Embedding 模型加载失败: {e}")
            log.warning("⚠️  降级使用 Mock Embedding（全零向量），RAG 检索效果会失效")
            self._model = None

    def encode(self, texts: str | list[str]) -> list[list[float]]:
        """把一段或多段文本转成向量"""
        self._ensure_model()

        if isinstance(texts, str):
            texts = [texts]

        if self._model is None:
            # Mock：返回全零向量
            return [[0.0] * self._dim for _ in texts]

        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        self._ensure_model()
        return self._dim


# 单例
_embedding_svc: EmbeddingService | None = None


def get_embedding() -> EmbeddingService:
    global _embedding_svc
    if _embedding_svc is None:
        _embedding_svc = EmbeddingService()
    return _embedding_svc
