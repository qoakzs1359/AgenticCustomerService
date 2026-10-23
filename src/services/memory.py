"""
Redis 中期记忆 —— 会话对话历史的快速读写缓存
设计：
    key: cs:{session_id}:history → List[JSON]
    过期时间: 24小时（一天内没对话就清掉）
"""
from __future__ import annotations
import json
from typing import Optional
from src.core.config import settings
from src.core.logging import log


KEY_PREFIX = "cs"
TTL_SECONDS = 24 * 3600  # 24h


def _make_key(session_id: str) -> str:
    return f"{KEY_PREFIX}:{session_id}:history"


class RedisMemory:
    """Redis 会话记忆"""

    def __init__(self):
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            import redis.asyncio as aioredis
            self._client = aioredis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                username=settings.redis_user or None,
                password=settings.redis_password or None,
                db=settings.redis_db,
                decode_responses=True,
            )
            log.info("✅ Redis 异步客户端初始化成功")
        except Exception as e:
            log.warning(f"⚠️  Redis 初始化失败（将用内存兜底）: {e}")
            self._client = None
        return self._client

    async def append(self, session_id: str, role: str, content: str) -> None:
        """追加一条消息"""
        client = self._ensure_client()
        if client is None:
            return
        try:
            await client.rpush(_make_key(session_id), json.dumps({"role": role, "content": content}))
            await client.expire(_make_key(session_id), TTL_SECONDS)
        except Exception as e:
            log.debug(f"Redis append 失败: {e}")

    async def get_recent(self, session_id: str, limit: int = 20) -> list[dict]:
        """取最近 N 条消息"""
        client = self._ensure_client()
        if client is None:
            return []
        try:
            # LRANGE 取尾部 limit 条 = 最新的 limit 条
            raw = await client.lrange(_make_key(session_id), -limit, -1)
            return [json.loads(r) for r in raw]
        except Exception as e:
            log.debug(f"Redis get_recent 失败: {e}")
            return []

    async def clear(self, session_id: str) -> None:
        """清空某会话的缓存"""
        client = self._ensure_client()
        if client is None:
            return
        try:
            await client.delete(_make_key(session_id))
        except Exception:
            pass


_memory_svc: Optional[RedisMemory] = None


def get_memory() -> RedisMemory:
    global _memory_svc
    if _memory_svc is None:
        _memory_svc = RedisMemory()
    return _memory_svc
