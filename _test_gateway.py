"""
网关层转人工拦截验证脚本
验证点:
  1. "转人工" → 网关层拦截（不走 agent），收到 reply + handoff
  2. "你好" → 走 agent.chat()（不被网关拦截）
  3. "我要人工客服" → 包含 "人工" → 触发网关层拦截
"""
from __future__ import annotations
import asyncio
import json
import sys

import websockets

WS_URL = "ws://localhost:8000/ws/user"


async def run_one_case(case_name: str, user_text: str, expected_intercept: bool):
    """连接一次，发一条消息，收集所有响应"""
    print(f"\n{'='*60}")
    print(f"🧪 测试用例: {case_name}")
    print(f"   用户发送: {user_text!r}")
    print(f"   预期网关拦截: {expected_intercept}")
    print(f"{'='*60}")

    try:
        async with websockets.connect(WS_URL, max_size=10 * 1024 * 1024) as ws:
            # 先收 session_start
            first = await asyncio.wait_for(ws.recv(), timeout=5)
            print(f"  📥 session_start: {first[:120]}")

            # 发送 chat 消息
            payload = json.dumps({"type": "chat", "text": user_text}, ensure_ascii=False)
            await ws.send(payload)

            # 收集响应（真实 LLM 可能慢，等 60 秒）
            responses = []
            deadline = asyncio.get_event_loop().time() + 60
            try:
                while asyncio.get_event_loop().time() < deadline:
                    remaining = deadline - asyncio.get_event_loop().time()
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                    msg = json.loads(raw)
                    responses.append(msg)
                    mtype = msg.get("type", "?")
                    print(f"  📥 type={mtype}  data={json.dumps(msg.get('data'), ensure_ascii=False)[:150]}")
            except asyncio.TimeoutError:
                pass

            # 分析
            reply_types = [r["type"] for r in responses]
            has_reply = "reply" in reply_types
            has_handoff = "handoff" in reply_types

            print(f"\n  📊 收到的消息类型: {reply_types}")
            print(f"  ✅ 有 reply: {has_reply}")
            print(f"  ✅ 有 handoff: {has_handoff}")

            # 校验
            ok = True
            if expected_intercept:
                if not has_handoff:
                    print(f"  ❌ FAIL: 预期网关拦截（应收到 handoff），但没收到")
                    ok = False
                if not has_reply:
                    print(f"  ❌ FAIL: 预期有 reply")
                    ok = False
            else:
                if has_handoff:
                    print(f"  ❌ FAIL: 预期不拦截，却收到 handoff")
                    ok = False
                if not has_reply:
                    print(f"  ❌ FAIL: 预期有 reply（来自 agent）")
                    ok = False

            if ok:
                print(f"  🎉 PASS")

    except Exception as e:
        print(f"  ❌ 连接异常: {e}")


async def main():
    cases = [
        ("CASE 1: '转人工' → 网关拦截", "转人工", True),
        ("CASE 2: '你好' → 走 agent", "你好", False),
        ("CASE 3: '我要人工客服' → 网关拦截（含'人工'）", "我要人工客服", True),
    ]

    for name, text, expected in cases:
        await run_one_case(name, text, expected)
        await asyncio.sleep(1)  # 间隔一下

    print(f"\n{'='*60}")
    print("🏁 全部测试用例跑完")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
