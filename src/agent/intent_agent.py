"""
意图识别智能体 —— 完整链路编排：用户问题 → 知识库向量检索 → LLM 意图判定

分工：
  intent_llm_prompt.py   意图识别 LLM 的提示词（角色/分类规则/taxonomy/输出契约）
  src/intent/evidence.py 知识库检索与证据聚合（rerank_score 透传）
  本文件                 编排（检索 → LLM 判定 → 证据填充 → 兜底）

输出 6 字段（IntentResult）：
  LLM 输出 4 字段：l1 / l2_code / l2_name / confidence_level
  代码层填充 2 字段：evidence_source（top1 l2_code）/ evidence_score（CrossEncoder rerank）

链路：
  用户问题 → 向量检索（多路召回 + RRF + CrossEncoder rerank）
          → 证据聚合（按 l2_code 聚类，保留 top_rerank_score）
          → LLM 判定（{taxonomy} + {context} + {user_question}）
          → 代码层填 evidence_source / evidence_score
          → IntentResult

降级：检索不可用 → LLM 无证据判定；LLM 不可用 → 关键词规则兜底
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from src.core.config import settings
from src.core.logging import log
from src.agent.intent_llm_prompt import system_prompt as _PROMPT_TEMPLATE, TAXONOMY

# 渲染一次 taxonomy，后续只替换动态的 {context} 和 {user_question}
_SYSTEM_PROMPT_BASE = _PROMPT_TEMPLATE.replace("{taxonomy}", TAXONOMY)

# LLM 完全不可用时的关键词兜底
_RULE_FALLBACK: list[tuple[list[str], str, str, str]] = [
    (["转人工", "人工客服"], "订单&售后退款", "ORDER_QUERY", "查询订单状态/找不到订单"),
    (["退款", "退货", "拒收"], "订单&售后退款", "AFTER_REFUND", "仅退款流程/退款到账时效"),
    (["换货"], "订单&售后退款", "AFTER_EXCHANGE", "换货/补发（少发漏发）申请"),
    (["订单", "下单", "取消"], "订单&售后退款", "ORDER_QUERY", "查询订单状态/找不到订单"),
    (["快递", "物流", "发货", "签收", "单号", "催发"], "物流咨询", "LOG_TRACK", "物流查询/停滞不更新"),
    (["尺码", "型号", "参数"], "商品咨询", "GOODS_SPEC", "规格参数/尺码选择咨询"),
    (["价格", "多少钱", "优惠", "券", "满减"], "活动优惠咨询", "PROMO_RULE", "活动规则/满减/到手价计算"),
    (["价保", "保价", "补差"], "活动优惠咨询", "PROMO_PRICE", "价保补差咨询"),
    (["运费", "运费险"], "物流咨询", "LOG_FEE", "运费/运费险相关咨询"),
    (["库存", "现货", "补货"], "商品咨询", "GOODS_STOCK", "库存/补货/发货时效咨询"),
]

_CONFIDENCE_LEVELS = {"HIGH", "MID", "LOW", "NONE"}

# ====== 意图漂移检测专用规则 ======
# 设计原则：
#   1. 每条漂移规则只关心 (keywords → l1)，不需要精确到 l2_code
#      —— 漂移检测的目标是"判断一级类别有没有变"，不需要做精准分类
#   2. 每条规则的关键词都是该 l1 最标志性的词，保证命中后 l1 判断可靠
#   3. 规则表独立于 _RULE_FALLBACK —— _RULE_FALLBACK 服务于 LLM 兜底识别，
#      这里服务于"快速判断新消息的一级类别归属"，两者目标不同

# l2_code → l1 的完整映射表（从 TAXONOMY 硬编码，和 taxonomy 同步维护）
_L1_BY_L2: dict[str, str] = {
    # 商品咨询
    "GOODS_SPEC": "商品咨询",
    "GOODS_USAGE": "商品咨询",
    "GOODS_STOCK": "商品咨询",
    "GOODS_GIFT": "商品咨询",
    "GOODS_QUALITY": "商品咨询",
    # 物流咨询
    "LOG_SHIP": "物流咨询",
    "LOG_TRACK": "物流咨询",
    "LOG_SIGN": "物流咨询",
    "LOG_FEE": "物流咨询",
    "LOG_ADDR": "物流咨询",
    # 订单&售后退款
    "ORDER_QUERY": "订单&售后退款",
    "ORDER_CANCEL": "订单&售后退款",
    "AFTER_REFUND": "订单&售后退款",
    "AFTER_RETURN": "订单&售后退款",
    "AFTER_EXCHANGE": "订单&售后退款",
    # 活动优惠咨询
    "PROMO_COUPON": "活动优惠咨询",
    "PROMO_RULE": "活动优惠咨询",
    "PROMO_PRICE": "活动优惠咨询",
    "PROMO_DEPOSIT": "活动优惠咨询",
}

# 漂移检测规则：(关键词列表, 归属 l1)
# 同一关键词出现在多条规则中时，先匹配到的生效（列表顺序 = 优先级）
_SHIFT_RULES: list[tuple[list[str], str]] = [
    # 订单&售后退款 —— 优先级最高，因为"退款"也可能和"运费险退款"等混淆
    (["退款", "退货", "拒收", "仅退", "退钱"], "订单&售后退款"),
    (["换货", "补发", "少发", "漏发"], "订单&售后退款"),
    (["订单", "下单", "取消订单", "取消", "找不到订单"], "订单&售后退款"),

    # 物流咨询
    (["快递", "物流", "发货", "签收", "单号", "催发", "到哪", "到哪了"], "物流咨询"),
    (["运费", "运费险"], "物流咨询"),
    (["拦截", "改地址", "改收货", "送到哪"], "物流咨询"),

    # 商品咨询
    (["尺码", "型号", "参数", "规格", "颜色", "多大"], "商品咨询"),
    (["库存", "现货", "补货", "还有货", "没货"], "商品咨询"),
    (["用法", "怎么用", "怎么装", "安装", "使用方法"], "商品咨询"),
    (["正品", "真的假的", "材质", "功效"], "商品咨询"),
    (["赠品", "送什么", "套餐内容"], "商品咨询"),

    # 活动优惠咨询
    (["价格", "多少钱", "优惠", "满减", "到手价"], "活动优惠咨询"),
    (["券", "优惠券", "领券", "券失效"], "活动优惠咨询"),
    (["价保", "保价", "补差", "降价"], "活动优惠咨询"),
    (["定金", "尾款", "预售"], "活动优惠咨询"),
]

# 方案 A：用户主动表达"换个话题"的标记词 —— 命中必漂移
_SHIFT_MARKERS: list[str] = [
    "换个", "换一换", "对了", "还有", "另外", "顺便", "话说回来",
    "不说这个了", "回到正题", "问个别的", "还有个问题", "另外问下",
    "顺便问下", "还有一件事",
]

# 漂移检测结果枚举
class ShiftResult:
    STAY = "stay"                    # 无漂移
    SHIFT_BY_MARKER = "shift_marker" # 方案 A：标记词触发
    SHIFT_BY_RULE = "shift_rule"     # 方案 C：规则冲突触发


def check_intent_shift(user_text: str, cached_l1: str | None) -> str:
    """检测用户新消息是否已从缓存意图漂移到新话题（方案 A + C 组合）

    :param user_text:     用户最新消息
    :param cached_l1:     当前缓存的一级类别名（如 "物流咨询"），None 表示无缓存
    :return: ShiftResult 中的一个字符串常量 —— 调用方据此决定是否重跑意图识别
    """
    if not cached_l1:
        return ShiftResult.STAY  # 没缓存谈不上漂移

    text = user_text.strip()

    # ── 方案 A：用户主动说"换个话题" → 必漂 ──
    for marker in _SHIFT_MARKERS:
        if marker in text:
            log.info(f"🔀 漂移检出[方案A-标记词]: '{marker}'")
            return ShiftResult.SHIFT_BY_MARKER

    # ── 方案 C：taxonomy 规则冲突 → 疑似漂 ──
    for keywords, rule_l1 in _SHIFT_RULES:
        if any(kw in text for kw in keywords):
            if rule_l1 != cached_l1:
                kw_hit = [kw for kw in keywords if kw in text][:3]
                log.info(
                    f"🔀 漂移检出[方案C-规则冲突]: "
                    f"缓存l1={cached_l1}, 新消息命中l1={rule_l1}, 关键词={kw_hit}"
                )
                return ShiftResult.SHIFT_BY_RULE
            # 命中且和缓存 l1 一致 → 不漂移，直接返回
            return ShiftResult.STAY

    # 新消息没命中任何规则 → 不触发漂移（保守策略，避免过度重识别）
    return ShiftResult.STAY


@dataclass
class IntentResult:
    """意图判定结果（v3.2）

    LLM 输出 5 字段，代码层派生 3 字段。
    """

    # === LLM 输出 ===
    l1: str = "未知"              # 一级类别名称
    l2_code: str = "unknown"      # 二级类别编码
    l2_name: str = "unknown"      # 二级类别名称
    confidence_level: str = "NONE"  # HIGH / MID / LOW / NONE
    evidence_refs: list[int] = field(default_factory=list)  # LLM 参考的检索片段编号

    # === 代码层派生（从 evidence_refs + evidence 查得） ===
    evidence_source: str = ""     # refs 第一条对应的 l2_code
    evidence_score: float = 0.0   # refs 第一条对应的 top_rerank_score
    evidence_texts: list[str] = field(default_factory=list)  # refs 对应的片段原文

    # === 系统元信息 ===
    source: str = "llm"           # llm / fallback
    evidence: list[dict] = field(default_factory=list)  # 追溯用

    @property
    def is_unknown(self) -> bool:
        return self.l2_code == "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "l1": self.l1,
            "l2_code": self.l2_code,
            "l2_name": self.l2_name,
            "confidence_level": self.confidence_level,
            "evidence_refs": self.evidence_refs,
            "evidence_source": self.evidence_source,
            "evidence_score": round(self.evidence_score, 4),
            "evidence_texts": self.evidence_texts,
            "source": self.source,
            "evidence_count": len(self.evidence),
        }


class IntentAgent:
    """意图识别智能体。

    用法：
        agent = get_intent_agent()
        result = await agent.detect("我的快递三天没更新了")
        result = await agent.detect("...", evidence=evs)  # 跳过检索
    """

    MAX_RETRY = 1

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
        self._evidence_builder = evidence_builder

    # ====== 对外入口 ======

    async def detect(
        self, user_text: str, evidence: list[dict] | None = None
    ) -> IntentResult:
        """完整链路：向量检索 → LLM 判定 → 证据填充"""
        text = user_text.strip()
        if not text:
            return self._empty_result()

        if evidence is None:
            evidence = self._retrieve_evidence(text)
        return await self.classify(text, evidence=evidence)

    async def classify(
        self, user_text: str, evidence: list[dict] | None = None
    ) -> IntentResult:
        """仅 LLM 判定（evidence 由外部传入）"""
        text = user_text.strip()
        if not text:
            return self._empty_result()

        # 规则兜底
        if not self.client:
            result = self._rule_fallback(text)
            self._fill_evidence(result, evidence)
            return result

        # 组装 system prompt（taxonomy 已预渲染，这里替换动态的 context + user_question）
        system_content = _SYSTEM_PROMPT_BASE.replace(
            "{context}", build_context_block(evidence)
        ).replace(
            "{user_question}", text
        )
        messages = [
            {"role": "system", "content": system_content},
        ]

        for attempt in range(self.MAX_RETRY + 1):
            try:
                result = await self._call_llm(messages)
                if result is not None:
                    result.evidence = evidence or []
                    self._fill_evidence(result, evidence)
                    return result
                log.warning(f"⚠️ 意图判定输出非法，第 {attempt + 1} 次重试")
            except Exception as e:
                log.warning(f"⚠️ 意图判定请求异常: {e}，第 {attempt + 1} 次重试")

        # 全挂 → 规则兜底
        log.warning("⚠️ 意图判定 LLM 调用失败，降级为规则兜底")
        result = self._rule_fallback(text)
        self._fill_evidence(result, evidence)
        return result

    # ====== 内部实现 ======

    def _empty_result(self) -> IntentResult:
        return IntentResult(source="fallback")

    def _retrieve_evidence(self, query: str) -> list[dict] | None:
        try:
            if self._evidence_builder is None:
                from src.intent.evidence import get_evidence_builder
                self._evidence_builder = get_evidence_builder()
            evidence = self._evidence_builder.build(query, enable_rerank=True)
            if evidence:
                log.info(f"🔍 检索到 {len(evidence)} 组候选: {[e['l2_code'] for e in evidence]}")
            else:
                log.info("🔍 知识库无命中，走无证据判定")
            return evidence
        except Exception as e:
            log.warning(f"⚠️ 检索不可用，降级为无证据判定: {type(e).__name__}: {e}")
            return None

    @staticmethod
    def _fill_evidence(result: IntentResult, evidence: list[dict] | None) -> None:
        """代码层从 LLM 输出的 evidence_refs 派生 evidence_source / evidence_score / evidence_texts。

        查找顺序：refs 里第一个有效 idx → 对应 evidence 条目 → 取 l2_code + top_rerank_score + samples 原文。
        refs 为空、idx 无效、或 evidence 无匹配时，都留空/留 0。
        """
        if not evidence or not result.evidence_refs:
            return
        evidence_by_idx = {e.get("idx"): e for e in evidence}
        texts: list[str] = []
        first_match = None
        for ref in result.evidence_refs:
            ev = evidence_by_idx.get(ref)
            if not ev:
                continue
            if first_match is None:
                first_match = ev
            # 每条 evidence 可能有多个 samples，全部收进 texts（去重）
            for s in (ev.get("samples") or []):
                if s and s not in texts:
                    texts.append(s)

        if first_match is None:
            # refs 全部无效 → fallback 到 evidence[0]
            first_match = evidence[0]
        result.evidence_source = str(first_match.get("l2_code", "") or "")
        rerank = first_match.get("top_rerank_score", 0.0) or 0.0
        vector = first_match.get("top_score", 0.0) or 0.0
        result.evidence_score = float(rerank or vector)
        result.evidence_texts = texts

    async def _call_llm(self, messages: list[dict]) -> IntentResult | None:
        """调一次 LLM，解析 5 字段输出，失败返回 None"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = (response.choices[0].message.content or "").strip()
        data = json.loads(raw)

        # l1
        l1 = str(data.get("l1", "")).strip() or "未知"

        # l2_code：大写，空或 UNKNOWN 统一为 "unknown"
        l2_code = str(data.get("l2_code", "")).strip().upper()
        if not l2_code or l2_code == "UNKNOWN":
            l2_code = "unknown"

        # l2_name
        l2_name = str(data.get("l2_name", "")).strip()
        if not l2_name or l2_name.lower() == "unknown":
            l2_name = "unknown"

        # confidence_level：校验四档
        cl = str(data.get("confidence_level", "")).strip().upper()
        if cl not in _CONFIDENCE_LEVELS:
            cl = "NONE"

        # evidence_refs：int 数组，最多 2 个，非法值忽略
        refs_raw = data.get("evidence_refs", []) or []
        refs: list[int] = []
        if isinstance(refs_raw, list):
            refs = [
                int(r) for r in refs_raw
                if str(r).strip().lstrip("-").isdigit()
            ][:2]

        # unknown 时强制一致
        if l2_code == "unknown":
            l1 = "未知"
            l2_name = "unknown"
            cl = "NONE"
            refs = []

        log.info(f"🎯 意图: L1={l1} L2={l2_code}({l2_name}) CL={cl} refs={refs}")
        return IntentResult(
            l1=l1, l2_code=l2_code, l2_name=l2_name, confidence_level=cl,
            evidence_refs=refs,
            source="llm",
        )

    def _rule_fallback(self, text: str) -> IntentResult:
        for keywords, l1, l2_code, l2_name in _RULE_FALLBACK:
            if any(kw in text for kw in keywords):
                return IntentResult(
                    l1=l1, l2_code=l2_code, l2_name=l2_name,
                    confidence_level="MID",  # 规则兜底给 MID
                    source="fallback",
                )
        return IntentResult(
            confidence_level="NONE",
            source="fallback",
        )


# ====== 运行时工具函数 ======


def build_context_block(evidence: list[dict] | None) -> str:
    """把 evidence 列表渲染成 system prompt 中的 {context} 块。"""
    if not evidence:
        return "（无检索片段）"
    lines = [
        "以下是知识库检索到的相似问题片段（仅供参考，可能存在语义相近但标签不同的干扰项，",
        "以你对用户真实诉求的理解为准）：",
    ]
    for e in evidence:
        code = e.get("l2_code", "")
        name = e.get("l2_name", "")
        rerank = e.get("top_rerank_score", 0.0)
        score_str = f"rerank={rerank:.2f}" if rerank else ""
        tag = f"{code}({name})" if name else code
        for s in (e.get("samples") or []):
            suffix = f" {score_str}" if score_str else ""
            lines.append(f"[{e['idx']}] {tag}{suffix}: {s}")
    return "\n".join(lines)


# ====== 单例 ======
_agent_instance: IntentAgent | None = None


def get_intent_agent() -> IntentAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = IntentAgent()
    return _agent_instance


async def detect_intent(user_text: str) -> IntentResult:
    return await get_intent_agent().detect(user_text)
