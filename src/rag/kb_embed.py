"""
严格的 Embedding 封装：文本 → 768 维归一化向量。

与旧的 `embedding.py` 的区别：
- 模型加载失败、返回零向量、维度不符时直接抛错，**不做 Mock 降级**；
- 支持 token 级截断（bge 上限 512，默认截到 510）；
- 批内按内容哈希去重，并有进程内 LRU 缓存，重灌同一文件时跳过未变内容。

入库链路必须 fail-fast：写入零向量比报错更难排查。
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from src.core.config import settings
from src.core.logging import log


# ----------------------------- 异常定义 -----------------------------

class EmbeddingError(RuntimeError):
    """Embedding 不可用或产出非法向量（fail-fast 的核心）。"""


# ----------------------------- 数据类 -----------------------------

@dataclass(slots=True)
class EmbeddingResult:
    """encode() 返回的结构化结果。

    slots=True 节省内存，所有字段都是只读语义。
    """
    vectors: list[list[float]]  # 归一化后的向量列表，顺序与输入一致
    truncated: list[bool]        # 对应每条输入是否发生 token 截断
    model: str                   # 模型名（如 BAAI/bge-base-zh-v1.5）
    dim: int                     # 向量维度（必须 768）
    cache_hits: int = 0          # 本次调用缓存命中数


# ----------------------------- 核心封装 -----------------------------

class Embedder:
    """线程安全的 Embedding 服务（懒加载单例）。

    两种后端，由 embedding_backend 切换，对外接口完全一致：
    - local：本机加载 sentence-transformers 模型推理；
    - http：调用服务器上的 embedding HTTP API（SSH 隧道访问）。

    设计要点：
    - 进程内 LRU 缓存（20000 条），重灌同一文件时跳过未变内容；
    - 批内按内容 hash 去重，同一批里重复的文本只算一次；
    - 查询侧（encode_query）自动加 BGE query 指令前缀，入库侧（encode_documents）不加。
    """

    def __init__(
        self,
        model_name: str | None = None,
        dim: int | None = None,
        max_tokens: int = 510,
        batch_size: int = 32,
        cache_size: int = 20_000,
        device: str | None = None,
        model: Any | None = None,
        query_instruction: str | None = None,
        backend: str | None = None,
        api_url: str | None = None,
        api_path: str | None = None,
        api_key: str | None = None,
    ) -> None:
        # ---- 后端选择：local 还是 http，默认从 .env 读 ----
        self.backend = backend or settings.embedding_backend

        # ---- 模型参数 ----
        self.model_name = model_name or settings.embedding_model_path  # BAAI/bge-base-zh-v1.5
        self.dim = dim or settings.embedding_dimension                 # 768
        self.max_tokens = max_tokens                                   # BGE 上限 512，留 2 给特殊 token
        self.batch_size = max(1, batch_size)                            # http 后端每次请求的文本数
        self.cache_size = max(0, cache_size)                            # 0 = 关闭缓存
        self.device = device                                            # local 后端的 torch device（cuda / cpu）

        # ---- BGE query 指令：查询侧自动加，入库侧不加 ----
        # None 表示从配置读；传 "" 显式关闭
        self.query_instruction = (
            settings.embedding_query_instruction
            if query_instruction is None
            else query_instruction
        )

        # ---- http 后端的连接参数（对应 .env 里的 EMBEDDING_API_URL / PATH / KEY）----
        self.api_url = (api_url if api_url is not None else settings.embedding_api_url).rstrip("/")
        self.api_path = (api_path if api_path is not None else settings.embedding_api_path) or "/embed"
        self.api_key = api_key if api_key is not None else settings.embedding_api_key

        # ---- 运行时状态 ----
        self._model = model                  # local 后端的 SentenceTransformer 实例
        self._tokenizer = getattr(model, "tokenizer", None) if model is not None else None
        self._loaded = model is not None     # 是否已探测/加载完毕

        # ---- LRU 缓存：key=文本 SHA1，value=(向量, 是否截断) ----
        # OrderedDict 同时用于记录 LRU 顺序
        self._cache: OrderedDict[str, tuple[list[float], bool]] = OrderedDict()
        self._lock = threading.RLock()       # 可重入锁，保证多线程安全

    # ==================== 模型加载 ====================

    def _ensure_model(self) -> None:
        """第一次调用 encode 时触发，http 后端只做 API 可达性探测。

        local 后端真正加载模型（~2GB 内存）。
        http 后端只 GET 一下端点，能连通就算就绪（不加载本地模型）。
        """
        with self._lock:
            if self._loaded:
                return  # 已经就绪，跳过
            if self.backend == "http":
                self._ensure_http_backend()
                return
            # ---- local 后端：真正加载 sentence-transformers ----
            log.info(f"加载 Embedding 模型: {self.model_name}")
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device=self.device)
                self._tokenizer = self._model.tokenizer  # token 截断用
                # 验证模型实际维度和配置一致，不一致直接报错
                getter = getattr(self._model, "get_embedding_dimension", None)
                if getter is None:
                    getter = self._model.get_sentence_embedding_dimension
                actual = int(getter())
                if actual != self.dim:
                    raise EmbeddingError(
                        f"模型维度 {actual} 与配置 {self.dim} 不一致，"
                        f"请修正 embedding_dimension 或更换模型"
                    )
            except EmbeddingError:
                raise  # 自己抛的直接透传
            except Exception as exc:
                raise EmbeddingError(
                    f"Embedding 模型加载失败（{self.model_name}）：{type(exc).__name__}: {exc}"
                ) from exc

            self._loaded = True
            log.info(f"Embedding 模型就绪，维度 = {self.dim}")

    # ==================== HTTP 后端（服务器 API） ====================

    def _ensure_http_backend(self) -> None:
        """http 后端：不做本地加载，只探测 API 可达性（fail-fast）。

        GET 请求端点，4xx 也算可达（因为 embedding 接口通常只接受 POST）。
        连不上直接抛 EmbeddingError，携带 SSH 隧道提示。
        """
        if not self.api_url:
            raise EmbeddingError(
                "embedding_backend=http 但未配置 embedding_api_url，"
                "请在 .env 里填服务器 embedding 服务地址"
            )
        endpoint = f"{self.api_url}{self.api_path}"
        log.info(f"检查 HTTP Embedding 服务: {endpoint}")
        from urllib.error import HTTPError, URLError
        from urllib import request as _req

        request = _req.Request(endpoint, method="GET")
        try:
            with _req.urlopen(request, timeout=8) as resp:
                resp.read(512)  # 只取前 512 字节，不关心内容
        except HTTPError as exc:
            # 服务可达即可（4xx 通常是接口只支持 POST）
            if not (400 <= exc.code <= 499):
                raise EmbeddingError(
                    f"HTTP Embedding 服务异常（{endpoint}）：HTTP {exc.code}"
                ) from exc
        except (URLError, OSError) as exc:
            raise EmbeddingError(
                f"HTTP Embedding 服务不可达（{endpoint}）："
                f"{type(exc).__name__}: {exc}，请检查 SSH 隧道是否已转发该端口"
            ) from exc
        self._loaded = True
        log.info(f"HTTP Embedding 服务就绪（{endpoint}），维度 = {self.dim}")

    def _http_encode(self, texts: Sequence[str]) -> list[list[float]]:
        """POST {api_url}{api_path}，请求体 {"texts": [...]}。

        兼容三种返回格式：
          {"vectors": [...]}  — 服务器自定义格式
          [...]               — 直接向量列表
          {"data": [{"embedding": [...]}, ...]}  — OpenAI 兼容格式

        请求头带 X-API-Key（不是 Authorization: Bearer，是服务器自定义头）。
        """
        import json as _json

        from urllib.error import URLError, HTTPError
        from urllib import request as _req

        endpoint = f"{self.api_url}{self.api_path}"
        # ensure_ascii=False 防止中文被转义，服务器需要原始中文字符
        payload = _json.dumps({"texts": list(texts)}, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key  # 服务器要求的是自定义头，不是标准 Bearer
        request = _req.Request(endpoint, data=payload, headers=headers, method="POST")
        try:
            with _req.urlopen(request, timeout=60) as resp:
                raw = resp.read()
        except (URLError, OSError, HTTPError) as exc:
            raise EmbeddingError(
                f"HTTP Embedding 调用失败（{endpoint}）：{type(exc).__name__}: {exc}"
            ) from exc

        try:
            data = _json.loads(raw)
        except Exception as exc:
            raise EmbeddingError(
                f"HTTP Embedding 响应解析失败（{endpoint}）：{type(exc).__name__}"
            ) from exc

        # ---- 兼容多种返回格式 ----
        if isinstance(data, dict) and "vectors" in data:
            vectors = data["vectors"]
        elif isinstance(data, list):
            vectors = data
        elif isinstance(data, dict) and "data" in data:  # OpenAI 兼容格式
            vectors = [item["embedding"] for item in data["data"]]
        else:
            raise EmbeddingError(
                f"HTTP Embedding 响应缺少 vectors 字段：{raw[:120]!r}"
            )

        # ---- 返回条数校验 ----
        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"HTTP Embedding 返回 {len(vectors)} 条，期望 {len(texts)} 条"
            )
        return [[float(x) for x in row] for row in vectors]

    # ==================== 编码 ====================

    def encode(
        self, texts: str | Iterable[str], instruction: str = ""
    ) -> EmbeddingResult:
        """把一批文本转成归一化向量，返回顺序与输入一致。

        instruction 只用于查询侧（BGE 非对称检索）；入库/文档侧传空字符串。

        完整流程：
          1. 输入校验（非空）→ 拼接指令前缀
          2. 模型就绪检查（懒加载）
          3. 批内按内容 hash 去重
          4. LRU 缓存命中直接复用
          5. 未命中的送 _encode_batch（local 或 http）
          6. 缓存新结果
          7. 按原始顺序还原（去重过要保证输出顺序和输入一致）
          8. 维度/零向量校验 → EmbeddingResult
        """
        # ---- 1. 归一化成列表 ----
        if isinstance(texts, str):
            items = [texts]
        else:
            items = [str(t) for t in texts]

        if not items:
            return EmbeddingResult([], [], self.model_name, self.dim)

        # ---- 2. 空文本校验（防止算出零向量写入 Qdrant）----
        for idx, text in enumerate(items):
            if not text or not text.strip():
                raise EmbeddingError(f"第 {idx + 1} 条文本为空，拒绝向量化")

        instruction = instruction or ""
        # instruction 拼在文本前面，BGE 就是这么设计的
        # query 侧： "为这个句子生成表示以用于检索相关文章：快递到哪了"
        # doc 侧：   "女装连衣裙的快递到哪了，怎么两天没更新？"（无前缀）
        prepared = [f"{instruction}{text}" if instruction else text for text in items]

        # ---- 3. 懒加载模型 ----
        self._ensure_model()

        # ---- 4. 批内去重 + LRU 缓存 ----
        # 前缀已拼进 prepared，所以 query/doc 天然分开缓存 key
        unique_texts: dict[str, str] = {}       # key → 待编码文本
        truncated_by_hash: dict[str, bool] = {} # key → 是否截断
        vectors: dict[str, list[float]] = {}    # key → 向量（这次调用内的）
        cache_hits = 0

        for text in prepared:
            key = self._hash(text)
            if key in vectors:
                continue  # 本次调用内已编码过
            cached = self._cache.get(key)
            if cached is not None:
                # LRU 命中：直接从缓存拿，不跑模型
                vectors[key], truncated_by_hash[key] = cached
                self._cache.move_to_end(key)  # 刷新 LRU 位置
                cache_hits += 1
                continue
            # 缓存未命中 → 截断（local 后端需要 tokenizer 才能截）
            truncated_text, was_truncated = self._truncate(text)
            unique_texts[key] = truncated_text
            truncated_by_hash[key] = was_truncated

        # ---- 5. 对未命中的做批量编码 ----
        pending = list(unique_texts.items())
        if pending:
            encoded = self._encode_batch([text for _, text in pending])
            for (key, _), vector in zip(pending, encoded):
                vectors[key] = vector
                self._put_cache(key, vector, truncated_by_hash[key])

        # ---- 6. 按原始输入顺序还原输出 ----
        out_vectors: list[list[float]] = []
        out_flags: list[bool] = []
        with self._lock:
            for text in prepared:
                key = self._hash(text)
                vector = vectors.get(key)
                if vector is None:
                    raise EmbeddingError("向量映射失败，输入与输出不一致")
                # ---- 7. 维度和零向量校验 ----
                if len(vector) != self.dim:
                    raise EmbeddingError(
                        f"向量维度 {len(vector)} != 期望 {self.dim}（text={text[:20]!r}）"
                    )
                if not any(abs(x) > 1e-12 for x in vector):
                    raise EmbeddingError(f"模型返回全零向量（text={text[:20]!r}）")
                out_vectors.append(vector)
                out_flags.append(truncated_by_hash.get(key, False))

        return EmbeddingResult(
            vectors=out_vectors,
            truncated=out_flags,
            model=self.model_name,
            dim=self.dim,
            cache_hits=cache_hits,
        )

    def encode_documents(self, texts: str | Iterable[str]) -> EmbeddingResult:
        """入库/文档侧编码：不加任何指令前缀。

        BGE 非对称设计：query 和 document 用不同指令，
        document 侧保持原文，让语义更纯粹。
        """
        return self.encode(texts, instruction="")

    def encode_query(self, texts: str | Iterable[str]) -> EmbeddingResult:
        """查询侧编码：按 BGE 要求加查询指令，和入库侧保持非对称。

        默认指令来自 .env 里的 EMBEDDING_QUERY_INSTRUCTION，
        典型值："为这个句子生成表示以用于检索相关文章："
        """
        return self.encode(texts, instruction=self.query_instruction)

    def encode_one(self, text: str) -> list[float]:
        """单条文档编码（兼容旧调用）。"""
        return self.encode_documents([text]).vectors[0]

    def encode_query_one(self, text: str) -> list[float]:
        """单条查询编码（兼容旧调用）。"""
        return self.encode_query([text]).vectors[0]

    def _encode_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """真正的批量编码：根据后端类型分发。

        http 后端：按 batch_size 分批调用 _http_encode，避免超出服务器单次请求上限。
        local 后端：直接让 SentenceTransformer.encode，显存/内存不足时自动减半 batch_size 重试。
        """
        if self.backend == "http":
            results: list[list[float]] = []
            bs = self.batch_size
            with self._lock:
                for i in range(0, len(texts), bs):
                    results.extend(self._http_encode(texts[i : i + bs]))
            return results

        # ---- local 后端 ----
        batch_size = self.batch_size
        with self._lock:
            while True:
                try:
                    matrix = self._model.encode(
                        list(texts),
                        batch_size=batch_size,
                        normalize_embeddings=True,  # L2 归一化，让余弦相似度直接可算
                        show_progress_bar=False,
                        convert_to_numpy=True,
                    )
                    return [[float(x) for x in row] for row in matrix]
                except (RuntimeError, MemoryError) as exc:
                    # 显存/内存爆了 → 减半 batch_size 重试（最少减到 1）
                    if batch_size <= 1:
                        raise EmbeddingError(f"向量化失败：{type(exc).__name__}: {exc}") from exc
                    batch_size = max(1, batch_size // 2)
                    log.warning(f"向量化显存/内存不足，batch_size 降为 {batch_size} 后重试")

    def _truncate(self, text: str) -> tuple[str, bool]:
        """按模型 tokenizer 截断到 max_tokens。

        local 后端有 tokenizer 能精确截；http 后端没 tokenizer 时直接返回原文。
        返回 (截断后的文本, 是否截断了)。
        """
        tokenizer = self._tokenizer
        if tokenizer is None:
            return text, False  # http 后端没 tokenizer，跳过节流
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) <= self.max_tokens:
                return text, False  # 没超，原样返回
            # 超过 → 只取前 max_tokens 个 token 再 decode 回文本
            short = tokenizer.decode(ids[: self.max_tokens], skip_special_tokens=True)
            return short, True
        except Exception as exc:
            log.warning(f"token 截断失败，按原文送入模型：{exc}")
            return text, False

    def _put_cache(self, key: str, vector: list[float], truncated: bool) -> None:
        """写入 LRU 缓存，超过 cache_size 时淘汰最久未用的。"""
        if self.cache_size <= 0:
            return  # cache_size=0 表示关闭缓存
        with self._lock:
            self._cache[key] = (vector, truncated)
            self._cache.move_to_end(key)  # 刚写的放队尾
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)  # 队头最老的淘汰

    @staticmethod
    def _hash(text: str) -> str:
        """文本 → SHA1，用作缓存 key。

        不直接用字符串做 dict key 是因为 prepared 带了 instruction 前缀，
        哈希能让 key 短一点、快一点。
        """
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    # ==================== 自检 ====================

    def health(self) -> dict[str, Any]:
        """不触发模型加载，只看当前状态（给 /health 接口用）。"""
        return {
            "backend": self.backend,
            "api_url": f"{self.api_url}{self.api_path}" if self.backend == "http" else "",
            "model": self.model_name,
            "dim": self.dim,
            "max_tokens": self.max_tokens,
            "batch_size": self.batch_size,
            "query_instruction": self.query_instruction or "(disabled)",
            "device": self.device or "auto",
            "loaded": self._loaded,
            "cache_size": len(self._cache),
        }


# ----------------------------- 单例 -----------------------------

_embedder: Embedder | None = None
_embedder_lock = threading.Lock()


def get_embedder() -> Embedder:
    """全局单例获取。双检锁模式，保证多线程安全。"""
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = Embedder()
    return _embedder
