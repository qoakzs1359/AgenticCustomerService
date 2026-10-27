"""
WebSocket 网关 —— 管理用户 / 人工客服 的实时连接

核心设计原则：
  1. 回复用户优先 —— ws_send(reply) 必须在持久化之前，绝不能因为 Redis/MySQL 阻塞而让用户等不到回复
  2. 持久化降级 —— Redis/MySQL 全部用 create_task 后台跑，永不阻塞主流程
  3. 异常隔离 —— 任何持久化异常只 log，不影响对话主链路
"""
from __future__ import annotations
import asyncio
import json
import uuid
from typing import Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from src.core.logging import log
from src.agent.customer_service_agent import get_agent

router = APIRouter()
PERSIST_TIMEOUT = 5  # 秒

# ====== 网关层转人工关键词（在进入任何 agent 之前拦截）======
_HUMAN_HANDOFF_KEYWORDS = ["人工", "转人工", "人工客服", "客服人工", "我要人工", "真人客服"]


def _contains_human_keyword(text: str) -> bool:
    """网关层硬拦截：用户消息里有没有转人工关键词"""
    return any(kw in text.strip() for kw in _HUMAN_HANDOFF_KEYWORDS)


# ==================== 连接管理 ====================

user_connections: dict[str, WebSocket] = {}
agent_connections: dict[str, WebSocket] = {}
session_agent_map: dict[str, str] = {}          # session_id → agent_id
agent_active_sessions: dict[str, list[str]] = {}  # agent_id → [session_ids] 正向映射
agent_queue: list[str] = []                     # 待接入队列
conversation_cache: dict[str, list[dict]] = {}  # session_id → 历史消息


def make_session_id() -> str:
    return f"sess_{uuid.uuid4().hex[:12]}"


def make_agent_id() -> str:
    return f"agent_{uuid.uuid4().hex[:8]}"


async def ws_send(ws: WebSocket, msg_type: str, payload: Any):
    """统一的 WebSocket 消息发送（加超时，防止客户端网络问题阻塞）"""
    try:
        await asyncio.wait_for(
            ws.send_text(json.dumps({"type": msg_type, "data": payload}, ensure_ascii=False)),
            timeout=3,
        )
    except asyncio.TimeoutError:
        log.warning(f"⚠️  ws_send 超时: {msg_type}")
    except Exception as e:
        log.warning(f"⚠️  ws_send 失败: {e}")


# ==================== 持久化（后台异步，永不阻塞主流程）====================

def _fire_and_forget(coro, desc: str = ""):
    """把协程扔后台跑，异常只 log 不向上传播"""
    async def wrapper():
        try:
            await asyncio.wait_for(coro, timeout=PERSIST_TIMEOUT)
        except asyncio.TimeoutError:
            log.debug(f"⏱️  持久化超时: {desc}")
        except Exception as e:
            log.debug(f"⚠️  持久化跳过: {desc} — {e}")
    try:
        asyncio.create_task(wrapper())
    except RuntimeError:
        # 没有 event loop（比如脚本测试场景），直接忽略
        pass


async def _load_history(session_id: str) -> list[dict]:
    """取会话历史 —— Redis 优先，降级内存"""
    try:
        from src.services.memory import get_memory
        redis_hist = await asyncio.wait_for(
            get_memory().get_recent(session_id, limit=20),
            timeout=PERSIST_TIMEOUT,
        )
        if redis_hist:
            return redis_hist
    except Exception as e:
        log.debug(f"Redis get_recent 降级: {e}")
    return conversation_cache.get(session_id, [])


def _bg_save_messages(
    session_id: str,
    entries: list[tuple[str, str]],
    tool_called: list[str] | None = None,
):
    """存消息 —— 内存 + Redis + MySQL（后台异步）"""
    # 内存同步写（必须立即生效，后面可能还要读）
    conversation_cache.setdefault(session_id, [])
    for role, content in entries:
        conversation_cache[session_id].append({"role": role, "content": content})

    # Redis 后台异步
    async def _redis():
        from src.services.memory import get_memory
        for role, content in entries:
            await get_memory().append(session_id, role, content)
    _fire_and_forget(_redis(), f"Redis 写入 {session_id}")

    # MySQL 后台异步
    async def _mysql():
        from src.services.mysql import save_message
        for role, content in entries:
            await save_message(session_id, role, content, tool_called if role == "assistant" else None)
    _fire_and_forget(_mysql(), f"MySQL save_message {session_id}")


def _bg_save_session_status(session_id: str, status: str):
    async def _do():
        from src.services.mysql import save_session
        await save_session(session_id, status)
    _fire_and_forget(_do(), f"MySQL save_session {session_id} → {status}")


# ==================== 用户端 ====================

@router.websocket("/ws/user")
async def user_websocket(ws: WebSocket):
    """
    用户聊天 WebSocket
    接收消息: {"type": "chat", "text": "订单到哪了"}
    """
    await ws.accept()

    session_id = make_session_id()
    user_connections[session_id] = ws
    conversation_cache[session_id] = []
    _bg_save_session_status(session_id, "active")

    await ws_send(ws, "session_start", {"session_id": session_id})
    log.info(f"🟢 用户会话启动: {session_id}")

    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await ws_send(ws, "error", "消息格式错误")
                continue

            msg_type = data.get("type")

            if msg_type == "chat":
                user_text = data.get("text", "").strip()
                if not user_text:
                    continue
                log.info(f"👤 [{session_id}] 用户: {user_text}")

                # 被人工接管 → 直接转发
                if session_id in session_agent_map:
                    agent_id = session_agent_map[session_id]
                    agent_ws = agent_connections.get(agent_id)
                    if agent_ws:
                        await ws_send(agent_ws, "user_message", {"session_id": session_id, "text": user_text})
                        await ws_send(ws, "agent_typing", {"session_id": session_id})
                        _bg_save_messages(session_id, [("user", user_text)])
                    else:
                        await ws_send(ws, "error", "客服连接异常")
                    continue

                # 🆕 网关层硬拦截：转人工关键词（不进任何 agent）
                if _contains_human_keyword(user_text):
                    log.info(f"🎯 网关层拦截转人工: {session_id}")
                    agent_queue.append(session_id)
                    _bg_save_session_status(session_id, "transferred")
                    _bg_save_messages(session_id, [("user", user_text)])
                    await ws_send(ws, "reply", {
                        "reply": "好的，正在为您转接人工客服，请稍候...",
                        "tool_called": [],
                    })
                    await ws_send(ws, "handoff", {"session_id": session_id})
                    log.info(f"🔁 会话 {session_id} 排队待人工接入，队列: {len(agent_queue)}")
                    for agent_id, agent_ws in agent_connections.items():
                        await ws_send(agent_ws, "new_handoff", {
                            "session_id": session_id,
                            "user_text": user_text,
                        })
                        await ws_send(agent_ws, "queue_update", agent_queue[:])
                    continue  # 跳过后面的 agent.chat()

                # 走智能体（intent_level_1/2 后续由 IntentAgent 接入后传入）
                agent = get_agent()
                history = await _load_history(session_id)
                result = await agent.chat(user_text, history)

                # ⚡ 先回复用户（不能等持久化）
                await ws_send(ws, "reply", {
                    "reply": result["reply"],
                    "tool_called": result["tool_called"],
                })

                # 💾 持久化放后台（绝不阻塞回复）
                _bg_save_messages(
                    session_id,
                    [("user", user_text), ("assistant", result["reply"])],
                    tool_called=result["tool_called"],
                )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error(f"❌ 用户 WebSocket 异常: {e}")
    finally:
        user_connections.pop(session_id, None)
        conversation_cache.pop(session_id, None)
        _bg_save_session_status(session_id, "closed")
        if session_id in session_agent_map:
            agent_id = session_agent_map.pop(session_id)
            agent_ws = agent_connections.get(agent_id)
            if agent_ws:
                await ws_send(agent_ws, "session_closed", {"session_id": session_id})
        log.info(f"🔴 用户会话结束: {session_id}")


# ==================== 客服端 ====================

@router.websocket("/ws/agent")
async def agent_websocket(ws: WebSocket):
    """人工客服后台 WebSocket"""
    await ws.accept()

    agent_id = make_agent_id()
    agent_connections[agent_id] = ws
    agent_active_sessions[agent_id] = []

    await ws_send(ws, "session_start", {"agent_id": agent_id})
    await ws_send(ws, "queue_update", agent_queue[:])
    # 客服上线时，把该客服已接入的会话 + 每条的历史发过去
    active = agent_active_sessions.get(agent_id, [])
    active_with_history = []
    for sid in active:
        active_with_history.append({
            "session_id": sid,
            "history": conversation_cache.get(sid, []),
        })
    await ws_send(ws, "agent_sessions", active_with_history)
    log.info(f"🛡️  人工客服上线: {agent_id}")

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type")

            if msg_type == "take_session":
                session_id = data.get("session_id")
                if session_id and session_id in agent_queue:
                    agent_queue.remove(session_id)
                    session_agent_map[session_id] = agent_id
                    # 加进该客服的 active 列表
                    if session_id not in agent_active_sessions[agent_id]:
                        agent_active_sessions[agent_id].append(session_id)
                    _bg_save_session_status(session_id, "transferred")

                    # 发给客服端：会话接管成功 + 历史消息（让客服看到机器人之前聊了啥）
                    history = conversation_cache.get(session_id, [])
                    await ws_send(ws, "session_taken", {
                        "session_id": session_id,
                        "history": history,
                    })

                    # 通知用户端：人工客服已接入
                    user_ws = user_connections.get(session_id)
                    if user_ws:
                        await ws_send(user_ws, "agent_connected", {
                            "session_id": session_id,
                            "agent_id": agent_id,
                        })
                        # 🆕 自动欢迎语 —— 接入后系统替人工客服说第一句话
                        welcome = "您好！我是人工客服小助手，很高兴为您服务~请问有什么可以帮您？"
                        await ws_send(user_ws, "agent_message", {"text": welcome})
                        _bg_save_messages(session_id, [("assistant", welcome)])

                    log.info(f"🧑‍💼 客服 {agent_id} 接管会话 {session_id}，历史 {len(history)} 条")
                    for aw in agent_connections.values():
                        await ws_send(aw, "queue_update", agent_queue[:])

            elif msg_type == "send":
                session_id = data.get("session_id")
                text = data.get("text", "").strip()
                if session_id and text:
                    user_ws = user_connections.get(session_id)
                    if user_ws:
                        await ws_send(user_ws, "agent_message", {"text": text})
                        _bg_save_messages(session_id, [("assistant", text)])

            elif msg_type == "close_session":
                session_id = data.get("session_id")
                if session_id:
                    session_agent_map.pop(session_id, None)
                    # 从客服的 active 列表移除
                    if agent_id in agent_active_sessions:
                        agent_active_sessions[agent_id] = [
                            s for s in agent_active_sessions[agent_id] if s != session_id
                        ]
                    _bg_save_session_status(session_id, "closed")
                    user_ws = user_connections.get(session_id)
                    if user_ws:
                        await ws_send(user_ws, "session_closed", {"session_id": session_id})
                    # 客服端自己也需要刷新 active 列表
                    await ws_send(ws, "agent_sessions", [
                        {"session_id": s, "history": conversation_cache.get(s, [])}
                        for s in agent_active_sessions[agent_id]
                    ])
                    log.info(f"❌ 会话 {session_id} 被客服 {agent_id} 关闭")

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error(f"❌ 客服 WebSocket 异常: {e}")
    finally:
        agent_connections.pop(agent_id, None)
        # 清理该客服所有 session 映射
        for sid in list(session_agent_map.keys()):
            if session_agent_map[sid] == agent_id:
                del session_agent_map[sid]
        # 清理 active 列表
        agent_active_sessions.pop(agent_id, None)
        log.info(f"🛑 人工客服下线: {agent_id}")
