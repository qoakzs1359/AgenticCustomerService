"""
意图识别智能体 —— 完整链路编排：用户问题 → 知识库向量检索 → LLM 意图判定

分工（与 src/agent/customer_service_agent.py 对齐）：
  intent_llm_prompt.py   意图识别 LLM 的提示词工程设计（角色/规则/模糊处理/置信度公式/输出契约）
  src/intent/evidence.py 知识库检索与证据聚合（动态变量 {context} 的来源）
  本文件                 智能体编排（检索调度 / 调 LLM / 注入动态变量 / 校验 / 重试 / 兜底）

链路：
  用户问题 → 向量检索知识库（kb_intent_v1）→ 证据聚合（带标签片段）
          → LLM 判定（{context} + {user_question} 注入）→ IntentResult（结构化输出）

降级策略（任一环节不可用不阻塞链路）：
  检索服务不可用 / 无命中 → 无证据判定（LLM 仅凭 {user_question} 判断，可输出 unknown）
  LLM 完全不可用（未配 key / 连续失败）→ 关键词规则兜底

路由说明：
  IntentResult → 子 Agent 路由为预留能力（route()），暂不实现。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from src.core.config import settings
from src.core.logging import log
from src.agent.intent_llm_prompt import system_prompt

# LLM 完全不可用时的关键词兜底：最少的关键词 → 意图（不做完整覆盖）
_RULE_FALLBACK: list[tuple[list[str], str, str, str]] = [
    (["转人工", "人工客服"], "订单&售后退款", "ORDER_QUERY", "转人工/找人工客服"),
    (["退款", "退货", "拒收"], "订单&售后退款", "AFTER_REFUND", "退款退货相关"),
    (["订单", "下单", "取消"], "订单&售后退款", "ORDER_QUERY", "订单相关"),
    (["快递", "物流", "发货", "签收", "单号"], "物流咨询", "LOG_TRACK", "物流相关"),
    (["尺码", "型号", "参数"], "商品咨询", "GOODS_SPEC", "商品规格参数"),
    (["价格", "多少钱", "优惠", "券", "满减"], "活动优惠咨询", "PROMO_RULE", "价格优惠相关"),
    (["价保", "保价", "补差"], "活动优惠咨询", "PROMO_PRICE", "价保补差"),
]


@dataclass
class IntentResult:
    """意图判定结果（LLM 输出的规范化对象）"""

    l1: str = "unknown"
    l2: str = "unknown"
    l3: str = ""
    confidence: float = 0.0
    confidence_breakdown: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    evidence_refs: list[int] = field(default_factory=list)
    need_clarify: bool = False
    clarify_hint: str = ""
    multi: list[str] = field(default_factory=list)
    source: str = "llm"  # llm / fallback
    evidence: list[dict] = field(default_factory=list)  # 本次判定使用的证据（追溯用）

    @property
    def is_unknown(self) -> bool:
        return self.l2 == "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "l1": self.l1,
            "l2": self.l2,
            "l3": self.l3,
            "confidence": self.confidence,
            "confidence_breakdown": self.confidence_breakdown,
            "reasoning": self.reasoning,
            "evidence_refs": self.evidence_refs,
            "need_clarify": self.need_clarify,
            "clarify_hint": self.clarify_hint,
            "multi": self.multi,
            "source": self.source,
            "evidence": self.evidence,
        }


class IntentAgent:
    """意图识别智能体：用户问题 → 知识库向量检索 → LLM 判定 → 校验/重试 → 兜底

    用法：
        agent = get_intent_agent()
        result = await agent.detect("我的快递三天没更新了")   # 完整链路（检索+判定）
        result = await agent.detect("...", evidence=evs)      # 外部已检索好，直接判定
    """

    MAX_RETRY = 1  # LLM 输出非法 JSON / 非法标签时重试次数

    def __init__(self, evidence_builder=None):
        if not settings.deepseek_api_key:
            log.warning("DEEPSEEK_API_KEY 未配置，意图识别只能走规则兜底")
            self.client = None
        else:
            self.client = AsyncOpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
            )
        self.model = settings.deepseek_model
        self.system_prompt = system_prompt
        self._evidence_builder = evidence_builder  # 惰性初始化，见 _retrieve_evidence

    # ====== 对外入口 ======

    async def detect(
        self, user_text: str, evidence: list[dict] | None = None
    ) -> IntentResult:
        """意图识别主入口：默认完整链路（向量检索 → LLM 判定）。

        参数:
            user_text: 用户原始消息
            evidence:  外部已检索好的证据（传入则跳过检索步骤）
        """
        text = user_text.strip()
        if not text:
            return IntentResult(l2="unknown", confidence=0.0,
                                reasoning="空消息", source="fallback")

        if evidence is None:
            evidence = self._retrieve_evidence(text)
        return await self.classify(text, evidence=evidence)

    async def classify(
        self, user_text: str, evidence: list[dict] | None = None
    ) -> IntentResult:
        """LLM 意图判定（证据驱动，不负责检索）。"""
        text = user_text.strip()
        if not text:
            return IntentResult(l2="unknown", confidence=0.0,
                                reasoning="空消息", source="fallback")

        # 未配置 API Key —— 规则兜底
        if not self.client:
            return self._rule_fallback(text)

        # 从 evidence 动态提取「合法标签集合」（无证据 → 空集，走 unknown）
        valid_tags = self._extract_valid_tags(evidence)

        user_content = build_user_content(text, evidence)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        for attempt in range(self.MAX_RETRY + 1):
            try:
                result = await self._call_llm(messages, valid_tags)
                if result is not None:
                    result.evidence = evidence or []
                    return result
                log.warning(f"⚠️ 意图判定输出非法，第 {attempt + 1} 次重试")
            except Exception as e:
                log.warning(f"⚠️ 意图判定请求异常: {e}，第 {attempt + 1} 次重试")

        # 全挂 —— 规则兜底
        log.warning("⚠️ 意图判定 LLM 调用失败，降级为规则兜底")
        return self._rule_fallback(text)

    # ====== 子 Agent 路由（预留） ======

    def route(self, result: IntentResult):
        """预留：按判定结果路由到对应子 Agent（暂不实现）。

        规划：
            - result.is_unknown            → 兜底回复智能体
            - result.need_clarify          → 澄清流程（用 result.clarify_hint 反问）
            - 其他                          → 按 result.l2 路由到对应业务子 Agent
        返回 None 表示路由尚未接入。
        """
        # TODO: 接入子 Agent 路由表后实现
        return None

    # ====== 内部实现 ======

    def _retrieve_evidence(self, query: str) -> list[dict] | None:
        """向量检索知识库并聚合证据；检索不可用时降级为 None（不阻塞判定）。"""
        try:
            if self._evidence_builder is None:
                from src.intent.evidence import get_evidence_builder

                self._evidence_builder = get_evidence_builder()
            evidence = self._evidence_builder.build(query)
            if evidence:
                log.info(f"🔍 检索到 {len(evidence)} 组候选证据: "
                         f"{[e['l2_code'] for e in evidence]}")
            else:
                log.info("🔍 知识库无命中，走无证据判定")
            return evidence
        except Exception as e:
            log.warning(f"⚠️ 知识库检索不可用，降级为无证据判定: "
                        f"{type(e).__name__}: {e}")
            return None

    @staticmethod
    def _extract_valid_tags(evidence: list[dict] | None) -> set[str]:
        """从 evidence 动态提取合法标签集合（{context} 中出现的标签）"""
        if not evidence:
            return set()
        return {str(e.get("l2_code", "")).strip().upper() for e in evidence if e.get("l2_code")}

    async def _call_llm(
        self, messages: list[dict], valid_tags: set[str]
    ) -> IntentResult | None:
        """调一次 LLM，返回规范化结果；解析/校验失败返回 None

        valid_tags: 本次 {context} 中出现的标签集合；LLM 输出的 l2 必须 ∈ valid_tags ∪ {unknown}
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,  # 分类任务要稳定，不要随机
        )
        raw = (response.choices[0].message.content or "").strip()
        data = json.loads(raw)

        l2_raw = str(data.get("l2", "")).strip()
        l2 = l2_raw.upper()
        # unknown 大小写归一化后统一为小写 "unknown"，其余标签保持大写编码
        if l2 == "UNKNOWN":
            l2 = "unknown"
        if l2 != "unknown" and l2 not in valid_tags:
            log.warning(
                f"⚠️ LLM 输出了不在检索片段中的标签: {l2}（合法集合={valid_tags}）"
            )
            return None

        l1 = str(data.get("l1", "")).strip()
        l3 = str(data.get("l3", "")).strip()
        reasoning = str(data.get("reasoning", "")).strip()

        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        # confidence_breakdown: LLM 逐个因子的估算（0~1）
        breakdown_raw = data.get("confidence_breakdown", {}) or {}
        confidence_breakdown: dict[str, float] = {}
        if isinstance(breakdown_raw, dict):
            for key in ("evidence_alignment", "intent_clarity", "category_specificity"):
                try:
                    confidence_breakdown[key] = max(
                        0.0, min(1.0, float(breakdown_raw.get(key, 0.0)))
                    )
                except (TypeError, ValueError):
                    confidence_breakdown[key] = 0.0

        multi = data.get("multi", []) or []
        if isinstance(multi, list):
            multi = [str(m).strip().upper() for m in multi if str(m).strip()]
            multi = [
                ("unknown" if m == "UNKNOWN" else m)
                for m in multi
                if m in valid_tags or m == "UNKNOWN"
            ][:2]

        # evidence_refs: LLM 引用的证据编号（int 列表）；非法值忽略
        evidence_refs = data.get("evidence_refs", []) or []
        if isinstance(evidence_refs, list):
            evidence_refs = [
                int(r) for r in evidence_refs
                if str(r).strip().lstrip("-").isdigit()
            ][:8]
        else:
            evidence_refs = []

        need_clarify = bool(data.get("need_clarify", False))
        clarify_hint = str(data.get("clarify_hint", "")).strip()

        log.info(
            f"🎯 意图判定: L1={l1} L2={l2} 置信度={confidence:.2f} "
            f"澄清={need_clarify} 证据={evidence_refs}"
        )
        return IntentResult(
            l1=l1,
            l2=l2,
            l3=l3,
            confidence=confidence,
            confidence_breakdown=confidence_breakdown,
            reasoning=reasoning,
            evidence_refs=evidence_refs,
            need_clarify=need_clarify,
            clarify_hint=clarify_hint,
            multi=multi,
            source="llm",
        )

    def _rule_fallback(self, text: str) -> IntentResult:
        """LLM 不可用时的关键词兜底"""
        for keywords, l1, l2, reason in _RULE_FALLBACK:
            if any(kw in text for kw in keywords):
                return IntentResult(
                    l1=l1, l2=l2, l3=reason,
                    confidence=0.5, reasoning=f"规则兜底: {reason}",
                    source="fallback",
                )
        return IntentResult(
            l2="unknown", confidence=0.5,
            reasoning="规则兜底无法识别", source="fallback",
        )


# ====== 运行时变量渲染（编排层，不属于提示词文件） ======


def build_context_block(evidence: list[dict] | None) -> str:
    """把 evidence 列表渲染成 LLM user message 中的 {context} 文本块。

    evidence 格式（与 src/intent/evidence.py 对齐）：
      [{"idx": 1, "l2_code": "LOG_TRACK", "samples": ["片段原文", ...]}, ...]
    """
    if not evidence:
        return "检索片段：（无）"
    lines = [
        "以下片段由知识库检索得到，可能存在语义相近但标签不同的干扰项；"
        "请按规则第 2 条审视每条是否真正匹配用户诉求，勿盲目采信。"
    ]
    for e in evidence:
        tag = e.get("l2_code", "")
        for s in (e.get("samples") or []):
            lines.append(f"[{e['idx']}] ({tag}) {s}")
    lines.append("（片段括号内为知识库标注的意图标签，也是你唯一可输出的标签来源）")
    return "\n".join(lines)


def build_user_content(user_text: str, evidence: list[dict] | None) -> str:
    """组装完整的 LLM user message：{context} 块 + 用户原文。"""
    return f"{build_context_block(evidence)}\n\n{user_text}"


# ====== 单例 ======
_agent_instance: IntentAgent | None = None


def get_intent_agent() -> IntentAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = IntentAgent()
    return _agent_instance


async def detect_intent(user_text: str) -> IntentResult:
    """便捷函数：单条消息意图识别（完整链路）"""
    return await get_intent_agent().detect(user_text)
