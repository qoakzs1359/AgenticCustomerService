"""
电商客服智能体核心 —— ReAct 循环 + DeepSeek Tool Calling

流程:
  用户消息 → 转人工检查 → LLM 推理(带 tools) → 有 tool_call? → 执行 tool → 结果喂回 LLM → 最终回复
                                                              → 无 tool_call → 直接回复
"""
from __future__ import annotations
import json
from typing import Any
from openai import AsyncOpenAI

from src.core.config import settings
from src.core.logging import log
from src.agent.prompt_loader import PromptLoader, AgentNames
from src.tools.mock_tools import get_tool_schemas, execute_tool


class CustomerServiceAgent:
    """电商客服智能体"""

    MAX_TOOL_ROUNDS = 5  # 防止 LLM 死循环调用 tool

    def __init__(self):
        if not settings.deepseek_api_key:
            log.warning("DEEPSEEK_API_KEY 未配置，Agent 只能用 Mock 回复")
            self.client = None
        else:
            self.client = AsyncOpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
            )
        self.model = settings.deepseek_model
        self.agent_name = AgentNames.CUSTOMER_SERVICE
        # system prompt 不再预加载 → 每次 chat 根据 intent 动态注入 category_context
        self.tool_schemas = get_tool_schemas()

    async def chat(
        self,
        user_text: str,
        conversation_history: list[dict[str, str]] | None = None,
        intent_level_1: str | None = None,
        intent_level_2: str | None = None,
    ) -> dict[str, Any]:
        """
        处理一条用户消息，返回:
        {
            "reply": str,              # 最终回复文本
            "handoff_human": bool,     # 是否触发转人工
            "tool_called": list[str],  # 本轮调用过的 tool 名称
        }

        :param user_text:              用户原始输入
        :param conversation_history:   历史对话（LLM messages 格式）
        :param intent_level_1:         意图识别输出的一级问题类别名（如 "商品咨询"）
        :param intent_level_2:         意图识别输出的二级问题编码（如 "GOODS_SPEC"）
                                       留空或传 None 时走 _default.md 通用能力描述
        """
        # 1. 未配置 API Key —— 返回 Mock
        if not self.client:
            return self._mock_reply(user_text)

        # 2. RAG 检索商品知识（失败不阻断，降级为空上下文）
        rag_context = ""
        try:
            from src.rag.retriever import get_rag_retriever
            rag_context = get_rag_retriever().retrieve(user_text, top_k=3)
        except Exception as e:
            log.warning(f"⚠️ RAG 检索失败（不影响对话）: {e}")

        # 3. 构建 messages —— 根据意图识别结果动态加载能力描述注入 system prompt
        sys_content = PromptLoader.load_for_intent(
            self.agent_name,
            intent_level_1=intent_level_1,
            intent_level_2=intent_level_2,
        )
        log.info(
            f"📝 动态 Prompt 加载完成: "
            f"intent_l1={intent_level_1 or '(未识别)'}, "
            f"intent_l2={intent_level_2 or '(未识别)'}, "
            f"prompt_len={len(sys_content)} chars"
        )

        if rag_context:
            sys_content = f"{sys_content}\n\n{rag_context}\n\n（请优先参考以上知识库内容回答）"

        messages: list[dict] = [
            {"role": "system", "content": sys_content},
        ]
        if conversation_history:
            messages.extend(conversation_history[-10:])  # 只带最近 10 条
        messages.append({"role": "user", "content": user_text})

        # 3. ReAct 循环
        called_tools: list[str] = []
        for round_idx in range(self.MAX_TOOL_ROUNDS):
            log.debug(f"🔄 Agent 第 {round_idx+1} 轮推理")

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tool_schemas,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            # 3a. LLM 没有要调工具 —— 这就是最终回复
            if not msg.tool_calls:
                final_reply = msg.content or "抱歉，我没太听懂您的意思，可以再说一遍吗？"
                log.info(f"💬 Agent 最终回复: {final_reply[:50]}...")
                return {
                    "reply": final_reply,
                    "handoff_human": False,
                    "tool_called": called_tools,
                }

            # 3b. LLM 要调工具 —— 执行后把结果喂回
            messages.append(msg)  # 先把 assistant 的 tool_call 消息加进去

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments or "{}")
                log.info(f"🛠️  调用工具: {tool_name}({tool_args})")
                called_tools.append(tool_name)

                result = await execute_tool(tool_name, tool_args)
                log.info(f"✅ 工具返回: {json.dumps(result, ensure_ascii=False)[:100]}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        # 4. 超过最大轮次 —— 兜底
        log.warning("⚠️  Agent 超过最大 tool 调用轮次，兜底回复")
        return {
            "reply": "抱歉，我正在帮您查询中，请稍候... 如果等待较久，可以联系人工客服。",
            "handoff_human": False,
            "tool_called": called_tools,
        }

    # ====== Mock 模式（未配置 API Key 时的兜底） ======

    def _mock_reply(self, user_text: str) -> dict[str, Any]:
        """没有 API Key 时的简单关键词匹配 Mock 回复"""
        text = user_text.lower()

        if any(k in text for k in ["订单", "发货", "状态"]):
            return {
                "reply": "[Mock回复] 订单 DD20261022001 当前状态是【已发货】，预计 10/23 送达~",
                "handoff_human": False,
                "tool_called": ["query_order"],
            }
        if any(k in text for k in ["快递", "物流", "到哪"]):
            return {
                "reply": "[Mock回复] SF1234567890 包裹目前正在【上海浦东分拨中心】，快递员张师傅正在派送中。",
                "handoff_human": False,
                "tool_called": ["query_logistics"],
            }
        if any(k in text for k in ["货", "价格", "多少钱", "库存"]):
            return {
                "reply": "[Mock回复] 秋冬保暖羽绒服 女款 当前库存 128 件，售价 ¥499，48 小时内发货哦~",
                "handoff_human": False,
                "tool_called": ["query_product"],
            }

        return {
            "reply": "[Mock回复] 您好，我是智能客服助手。如果需要查询订单、物流或商品信息，可以告诉我哦~",
            "handoff_human": False,
            "tool_called": [],
        }


# ====== 单例 ======
_agent_instance: CustomerServiceAgent | None = None


def get_agent() -> CustomerServiceAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = CustomerServiceAgent()
    return _agent_instance
