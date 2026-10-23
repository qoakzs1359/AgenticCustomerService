"""
MySQL 持久化 —— SQLAlchemy 异步

模型设计：
    session     : 会话表（会话状态：active / transferred / closed）
    message     : 消息表（每条消息：user / assistant / human）

MVP 阶段先把模型 + 连接 + 初始化写好，WebSocket 网关里暂时不主动落库，
验证通过后再把落库逻辑接进 websocket.py
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, String, Integer, Text, DateTime, Enum, create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from src.core.config import settings
from src.core.logging import log


Base = declarative_base()


class SessionStatus(str):
    ACTIVE = "active"
    TRANSFERRED = "transferred"
    CLOSED = "closed"


class Session(Base):
    """会话表"""
    __tablename__ = "session"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, nullable=False, index=True, comment="前端生成的会话ID")
    status = Column(String(16), nullable=False, default="active", comment="active/transferred/closed")
    agent_id = Column(String(64), nullable=True, comment="接管的人工客服ID")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class Message(Base):
    """消息表"""
    __tablename__ = "message"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, index=True)
    role = Column(String(16), nullable=False, comment="user / assistant / human")
    content = Column(Text, nullable=False)
    tool_called = Column(String(256), nullable=True, comment="本轮调用过的 tool 名（逗号分隔）")
    created_at = Column(DateTime, default=datetime.now)


# ====== 异步引擎 ======
_engine = None
_async_session_factory: Optional[async_sessionmaker] = None


async def get_session_factory() -> async_sessionmaker:
    global _engine, _async_session_factory
    if _engine is None:
        try:
            _engine = create_async_engine(
                settings.mysql_dsn,
                echo=False,
                pool_pre_ping=True,
            )
            _async_session_factory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
            log.info("✅ MySQL 异步引擎初始化成功")
        except Exception as e:
            log.error(f"❌ MySQL 连接失败（服务可能未启动）: {e}")
            _engine = None
            raise
    return _async_session_factory


async def init_db():
    """首次启动时建表 —— 用同步引擎简单粗暴地建"""
    try:
        # 用同步引擎建表（AsyncEngine 的 create_all 在某些版本有坑）
        from sqlalchemy import create_engine as _create_engine
        sync_dsn = settings.mysql_dsn.replace("mysql+aiomysql://", "mysql+pymysql://")
        sync_engine = _create_engine(sync_dsn, echo=False)
        Base.metadata.create_all(sync_engine, checkfirst=True)
        log.info("✅ MySQL 表初始化完成")
    except Exception as e:
        log.warning(f"⚠️  MySQL 建表跳过（可忽略）: {e}")


# ====== 便捷 CRUD ======

async def save_session(session_id: str, status: str = "active") -> None:
    """创建或更新会话"""
    try:
        factory = await get_session_factory()
        async with factory() as session:
            existing = await session.get(Session, session_id)  # type: ignore[arg-type]
            if existing:
                existing.status = status
                existing.updated_at = datetime.now()
            else:
                session.add(Session(session_id=session_id, status=status))
            await session.commit()
    except Exception as e:
        log.warning(f"⚠️  save_session 失败: {e}")


async def save_message(
    session_id: str,
    role: str,
    content: str,
    tool_called: list[str] | None = None,
) -> None:
    """存一条消息"""
    try:
        factory = await get_session_factory()
        async with factory() as session:
            tool_str = ",".join(tool_called) if tool_called else None
            session.add(Message(
                session_id=session_id,
                role=role,
                content=content,
                tool_called=tool_str,
            ))
            await session.commit()
    except Exception as e:
        log.warning(f"⚠️  save_message 失败: {e}")


async def get_recent_messages(session_id: str, limit: int = 20) -> list[dict]:
    """取某会话最近 N 条消息（返回 LLM messages 格式）"""
    try:
        factory = await get_session_factory()
        async with factory() as session:
            rows = await session.execute(
                Message.__table__.select()  # type: ignore[attr-defined]
                .where(Message.session_id == session_id)
                .order_by(Message.created_at.desc())  # type: ignore[attr-defined]
                .limit(limit)
            )
            # 用 select 的方式兼容
            from sqlalchemy import select
            stmt = select(Message).where(
                Message.session_id == session_id
            ).order_by(Message.created_at.desc()).limit(limit)  # type: ignore[attr-defined]
            result = await session.execute(stmt)
            msgs = result.scalars().all()
            return [{"role": m.role, "content": m.content} for m in reversed(msgs)]
    except Exception as e:
        log.warning(f"⚠️  get_recent_messages 失败: {e}")
        return []
