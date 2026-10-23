"""
平台 Mock REST API —— 模拟真实电商平台的订单/物流/商品接口
目的：
  1. 前端/Postman 可以直接调用验证数据
  2. Tool 层后续可以改为调真实 HTTP 接口，保持接口形状不变
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.tools.mock_tools import MOCK_ORDERS, MOCK_LOGISTICS, MOCK_PRODUCTS

router = APIRouter()


# ==================== Orders ====================

@router.get("/orders/{order_id}")
async def get_order(order_id: str):
    order = MOCK_ORDERS.get(order_id.strip())
    if not order:
        raise HTTPException(status_code=404, detail=f"订单 {order_id} 不存在")
    return {"code": 0, "data": order}


class OrderCreate(BaseModel):
    product_id: str
    quantity: int = 1


@router.post("/orders")
async def create_order(payload: OrderCreate):
    """模拟下单接口"""
    product = MOCK_PRODUCTS.get(payload.product_id)
    if not product:
        raise HTTPException(status_code=404, detail="商品不存在")

    order_id = f"DD{datetime.now().strftime('%Y%m%d')}{len(MOCK_ORDERS)+1:03d}"
    from datetime import datetime
    new_order = {
        "order_id": order_id,
        "product_name": product["name"],
        "product_id": payload.product_id,
        "amount": product["price"] * payload.quantity,
        "quantity": payload.quantity,
        "status": "待发货",
        "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "express_company": None,
        "express_no": None,
    }
    MOCK_ORDERS[order_id] = new_order
    return {"code": 0, "data": new_order}


# ==================== Logistics ====================

@router.get("/logistics/{express_no}")
async def get_logistics(express_no: str):
    tracks = MOCK_LOGISTICS.get(express_no.strip())
    if not tracks:
        raise HTTPException(status_code=404, detail=f"快递单号 {express_no} 不存在")
    return {"code": 0, "data": {"express_no": express_no, "tracks": tracks}}


# ==================== Products ====================

@router.get("/products/{product_id}")
async def get_product(product_id: str):
    product = MOCK_PRODUCTS.get(product_id.strip())
    if not product:
        raise HTTPException(status_code=404, detail=f"商品 {product_id} 不存在")
    return {"code": 0, "data": product}


@router.get("/products")
async def list_products(keyword: Optional[str] = None):
    """商品列表（支持关键词过滤）"""
    items = list(MOCK_PRODUCTS.values())
    if keyword:
        items = [p for p in items if keyword in p["name"]]
    return {"code": 0, "data": items}
