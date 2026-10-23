"""
三个 Mock Tool 的实现
每个 Tool 暴露:
    - schema: dict  —— 给 LLM 的 tool definition（JSON Schema 格式）
    - async run(**kwargs) —— 真正执行逻辑的函数
"""
from __future__ import annotations
from typing import Any
import random
from datetime import datetime, timedelta


# ==================== Mock 数据源 ====================

MOCK_ORDERS = {
    "DD20261022001": {
        "order_id": "DD20261022001",
        "product_name": "秋冬保暖羽绒服 女款",
        "product_id": "P10086",
        "amount": 499.00,
        "quantity": 1,
        "status": "已发货",
        "create_time": "2026-10-20 14:32:00",
        "express_company": "顺丰速运",
        "express_no": "SF1234567890",
    },
    "DD20261022002": {
        "order_id": "DD20261022002",
        "product_name": "无线蓝牙耳机 Pro",
        "product_id": "P2048",
        "amount": 299.00,
        "quantity": 2,
        "status": "待发货",
        "create_time": "2026-10-22 09:15:00",
        "express_company": None,
        "express_no": None,
    },
    "DD20261021003": {
        "order_id": "DD20261021003",
        "product_name": "家用智能扫地机器人",
        "product_id": "P3072",
        "amount": 1299.00,
        "quantity": 1,
        "status": "已签收",
        "create_time": "2026-10-18 16:48:00",
        "express_company": "京东物流",
        "express_no": "JD9876543210",
    },
}

MOCK_LOGISTICS = {
    "SF1234567890": [
        {"time": "2026-10-21 08:00:00", "status": "包裹已到达【北京转运中心】"},
        {"time": "2026-10-21 22:15:00", "status": "包裹已发出，下一站【上海浦东分拨中心】"},
        {"time": "2026-10-22 06:30:00", "status": "包裹已到达【上海浦东分拨中心】"},
        {"time": "2026-10-22 10:00:00", "status": "快递员【张师傅 138****1234】正在派送中"},
    ],
    "JD9876543210": [
        {"time": "2026-10-20 10:00:00", "status": "订单已出库"},
        {"time": "2026-10-21 14:00:00", "status": "包裹已到达【广州分拣中心】"},
        {"time": "2026-10-21 18:30:00", "status": "包裹已签收，签收人：本人"},
    ],
}

MOCK_PRODUCTS = {
    "P10086": {
        "product_id": "P10086",
        "name": "秋冬保暖羽绒服 女款",
        "price": 499.00,
        "stock": 128,
        "colors": ["黑色", "米白色", "卡其色"],
        "sizes": ["S", "M", "L", "XL"],
        "material": "90%白鸭绒 + 10%羽毛",
        "weight": "约 1.2kg",
        "ship_eta": "48小时内发货",
        "warranty": "支持7天无理由退换",
    },
    "P2048": {
        "product_id": "P2048",
        "name": "无线蓝牙耳机 Pro",
        "price": 299.00,
        "stock": 56,
        "colors": ["曜石黑", "珍珠白"],
        "battery": "单次续航8小时，充电盒32小时",
        "bluetooth": "Bluetooth 5.3",
        "ship_eta": "24小时内发货",
        "warranty": "支持7天无理由退换，一年质保",
    },
    "P3072": {
        "product_id": "P3072",
        "name": "家用智能扫地机器人",
        "price": 1299.00,
        "stock": 12,
        "colors": ["高级黑", "月光银"],
        "features": ["激光导航", "自动回充", "App遥控", "扫拖一体"],
        "ship_eta": "48小时内发货",
        "warranty": "支持7天无理由退换，两年质保",
    },
}


# ==================== Tool 定义 ====================

class QueryOrderTool:
    """查询订单详情"""

    schema = {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": "根据订单号查询订单详情，包括商品、金额、状态、物流信息。当用户询问订单状态、购买商品、是否已发货时调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单编号，例如 DD20261022001",
                    }
                },
                "required": ["order_id"],
            },
        },
    }

    @staticmethod
    async def run(order_id: str) -> dict[str, Any]:
        order = MOCK_ORDERS.get(order_id.strip())
        if not order:
            return {"success": False, "message": f"未找到订单 {order_id}"}
        return {"success": True, "data": order}


class QueryLogisticsTool:
    """查询物流轨迹"""

    schema = {
        "type": "function",
        "function": {
            "name": "query_logistics",
            "description": "根据快递单号查询物流轨迹。当用户询问快递到哪了、为什么还没到、快递状态时调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "express_no": {
                        "type": "string",
                        "description": "快递单号，例如 SF1234567890",
                    }
                },
                "required": ["express_no"],
            },
        },
    }

    @staticmethod
    async def run(express_no: str) -> dict[str, Any]:
        tracks = MOCK_LOGISTICS.get(express_no.strip())
        if not tracks:
            return {"success": False, "message": f"未找到快递单号 {express_no} 的物流信息"}
        return {"success": True, "express_no": express_no, "tracks": tracks}


class QueryProductTool:
    """查询商品信息"""

    schema = {
        "type": "function",
        "function": {
            "name": "query_product",
            "description": "根据商品ID查询商品详情，包括价格、库存、规格、发货时效、售后政策。当用户询问商品有没有货、多少钱、参数规格时调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "商品编号，例如 P10086",
                    }
                },
                "required": ["product_id"],
            },
        },
    }

    @staticmethod
    async def run(product_id: str) -> dict[str, Any]:
        product = MOCK_PRODUCTS.get(product_id.strip())
        if not product:
            return {"success": False, "message": f"未找到商品 {product_id}"}
        return {"success": True, "data": product}


# ==================== Tool 注册表 ====================

ALL_TOOLS: list = [QueryOrderTool, QueryLogisticsTool, QueryProductTool]


def get_tool_schemas() -> list[dict]:
    """给 LLM 用的 tool definitions"""
    return [t.schema for t in ALL_TOOLS]


async def execute_tool(name: str, arguments: dict) -> dict:
    """按名称执行某个 Tool"""
    for tool_cls in ALL_TOOLS:
        if tool_cls.schema["function"]["name"] == name:
            return await tool_cls.run(**arguments)
    return {"success": False, "message": f"未知的工具: {name}"}
