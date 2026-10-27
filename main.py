"""
FastAPI 入口
启动: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
import os
import sys
from contextlib import asynccontextmanager

# 把工作区本地包目录加入 sys.path（pip install --target 装的包放这里）
_LOCAL_PYLIBS = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pylibs")
if os.path.isdir(_LOCAL_PYLIBS) and _LOCAL_PYLIBS not in sys.path:
    sys.path.insert(0, _LOCAL_PYLIBS)
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.core.config import settings
from src.core.logging import log

FRONTEND_DIST = "web/dist"
FALLBACK_INDEX = "web/user.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期 —— 启动时预热依赖，关闭时释放资源"""
    log.info("🚀 电商客服智能体启动中...")

    # 检查 DeepSeek API Key 是否配置
    if not settings.deepseek_api_key:
        log.warning("⚠️  DEEPSEEK_API_KEY 未配置，智能体对话功能不可用，请在 .env 中填写")
    else:
        log.info("✅ DeepSeek API Key 已配置")

    # 初始化数据库（MySQL 建表）
    try:
        from src.services.mysql import init_db
        await init_db()
    except Exception as e:
        log.warning(f"⚠️  MySQL 初始化跳过：{e}")

    # 检查前端产物
    if os.path.exists(os.path.join(FRONTEND_DIST, "index.html")):
        log.info("✅ Vue3 前端产物已就绪（web/dist）")
    else:
        log.warning("⚠️  未找到 Vue 前端产物，使用原生 HTML 兜底。运行 'cd web-vue && npm run build' 构建")

    log.info(f"✅ 监听地址 {settings.server_host}:{settings.server_port}")
    yield
    log.info("👋 服务关闭")


app = FastAPI(
    title="电商客服智能体",
    description="MVP — 售前/售后/物流一站式 AI 客服，转人工兜底",
    version="0.1.0",
    lifespan=lifespan,
)


# ====== 后端自有路由（必须在 spa_fallback 之前注册）======

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "version": "0.2.0"}


from src.api.mock import router as mock_router
from src.api.websocket import router as ws_router
app.include_router(mock_router, prefix="/api/mock", tags=["Mock"])
app.include_router(ws_router, tags=["WebSocket"])


# ====== SPA history 模式 fallback（必须放最后，否则会吞掉 /api /ws 等）======

@app.get("/", include_in_schema=False)
@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str | None = None):
    """
    所有非 API 路径 → 返回 Vue SPA index.html，让前端 createWebHistory() 路由接管。

    顺序说明：
      FastAPI 里具体路由（/health, /api/mock/*, /ws/* 等）会先被匹配到，
      本函数是 catch-all 兜底，只处理那些后端不认识的路径（/user, /agent, /favicon.svg 等）。
    """
    dist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), FRONTEND_DIST)

    # 1. 有具体路径且是静态文件 → 正常返回
    if full_path:
        static_file = os.path.join(dist_dir, full_path)
        # 安全：不能跳出 dist 目录
        real_static = os.path.realpath(static_file)
        real_dist = os.path.realpath(dist_dir)
        if real_static.startswith(real_dist) and os.path.isfile(real_static):
            return FileResponse(real_static)

    # 2. 兜底：Vue SPA index.html（让前端路由处理 /user /agent 等）
    index_html = os.path.join(dist_dir, "index.html")
    if os.path.isfile(index_html):
        return FileResponse(index_html)

    # 3. 连前端产物都没有 → 原生 HTML 兜底
    fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)), FALLBACK_INDEX)
    if os.path.isfile(fallback):
        return FileResponse(fallback)

    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail="Not Found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=True,
    )
