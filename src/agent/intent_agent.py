"""
意图识别 Agent —— 骨架代码（后续实现）

职责:
  接收用户输入，输出结构化的意图分类结果:
    {
        "intent_level_1": str | None,   # 一级问题类别名（如 "商品咨询"）
        "intent_level_2": str | None,   # 二级问题编码（如 "GOODS_SPEC"）
        "confidence": float             # 置信度 0-1
    }

输出映射表（来自 cs_question_category 数据库）:
  ┌────────── 一级 ──────────┬────────────── 二级编码 ─────────────┐
  │ 商品咨询                 │ GOODS_SPEC / GOODS_USAGE / GOODS_STOCK / GOODS_GIFT / GOODS_QUALITY │
  │ 物流咨询                 │ LOG_SHIP / LOG_TRACK / LOG_SIGN / LOG_FEE / LOG_ADDR │
  │ 订单&售后退款             │ ORDER_QUERY / ORDER_CANCEL / AFTER_REFUND / AFTER_RETURN / AFTER_EXCHANGE │
  │ 活动优惠咨询              │ PROMO_COUPON / PROMO_RULE / PROMO_PRICE / PROMO_DEPOSIT │
  └─────────────────────────┴──────────────────────────────────────┘

当前阶段:
  先用关键词 Mock 实现（开发测试用），后续替换为 LLM 结构化输出。
  真实实现时，PromptLoader 已准备好 prompts/intent/{system,user}.md。
"""
from __future__ import annotations
import re
from typing import Any
from src.core.logging import log


# ====== Mock 关键词规则（开发阶段用，后续替换为 LLM 分类） ======
# 格式: (category_code, level1_name, [关键词列表])
_KEYWORD_RULES: list[tuple[str, str, list[str]]] = [
    # 商品咨询
    ("GOODS_SPEC", "商品咨询", ["尺码", "规格", "参数", "型号", "多大", "颜色", "选哪个"]),
    ("GOODS_USAGE", "商品咨询", ["怎么用", "使用方法", "安装", "教程", "步骤"]),
    ("GOODS_STOCK", "商品咨询", ["现货", "库存", "补货", "发货时间", "预售", "多久发", "什么时候发"]),
    ("GOODS_GIFT", "商品咨询", ["赠品", "套餐", "包含", "有什么送", "礼包"]),
    ("GOODS_QUALITY", "商品咨询", ["正品", "真假", "材质", "成分", "功效", "有没有证明"]),
    # 物流咨询
    ("LOG_SHIP", "物流咨询", ["催发货", "还没发", "为什么不发", "快点发", "发货没"]),
    ("LOG_TRACK", "物流咨询", ["快递到哪", "物流", "到哪了", "在哪", "不动", "轨迹"]),
    ("LOG_SIGN", "物流咨询", ["显示签收", "没收到", "没拿到", "快递柜", "驿站"]),
    ("LOG_FEE", "物流咨询", ["运费", "运费险", "谁出运费", "包邮", "补运费"]),
    ("LOG_ADDR", "物流咨询", ["改地址", "拦截", "退回", "改收货", "换地址"]),
    # 订单&售后
    ("ORDER_QUERY", "订单&售后退款", ["订单号", "订单在哪", "找订单", "订单状态", "查订单"]),
    ("ORDER_CANCEL", "订单&售后退款", ["取消订单", "撤单", "不要了", "能取消"]),
    ("AFTER_REFUND", "订单&售后退款", ["退款", "仅退款", "退钱", "退款到账"]),
    ("AFTER_RETURN", "订单&售后退款", ["退货", "退货退款", "退回去", "退货地址"]),
    ("AFTER_EXCHANGE", "订单&售后退款", ["换货", "补发", "少发", "漏发"]),
    # 活动优惠
    ("PROMO_COUPON", "活动优惠咨询", ["优惠券", "领券", "券怎么用", "券失效", "券不能用"]),
    ("PROMO_RULE", "活动优惠咨询", ["满减", "活动规则", "到手价", "拼团", "秒杀"]),
    ("PROMO_PRICE", "活动优惠咨询", ["价保", "补差", "退差价", "降价了"]),
    ("PROMO_DEPOSIT", "活动优惠咨询", ["定金", "尾款", "预售定金"]),
]


class IntentAgent:
    """意图识别 Agent（当前为关键词 Mock，后续替换为 LLM 结构化输出）"""

    def classify(self, user_text: str) -> dict[str, Any]:
        """
        分类用户意图，输出结构化结果。

        :return: {"intent_level_1": str|None, "intent_level_2": str|None, "confidence": float}
        """
        text = user_text.strip().lower()

        for code, level1, keywords in _KEYWORD_RULES:
            for kw in keywords:
                if kw in text:
                    log.info(
                        f"🎯 意图识别命中: "
                        f"L1={level1}, L2={code}, keyword={kw}"
                    )
                    return {
                        "intent_level_1": level1,
                        "intent_level_2": code,
                        "confidence": 0.7,  # 关键词匹配置信度保守给 0.7
                    }

        log.info("🎯 意图识别未命中任何规则，返回默认")
        return {
            "intent_level_1": None,
            "intent_level_2": None,
            "confidence": 0.0,
        }


# ====== 单例 ======
_intent_instance: IntentAgent | None = None


def get_intent_agent() -> IntentAgent:
    global _intent_instance
    if _intent_instance is None:
        _intent_instance = IntentAgent()
    return _intent_instance
