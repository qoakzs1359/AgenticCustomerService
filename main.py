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


@app.get("/")
async def root():
    """根路径 —— 优先返回 Vue SPA，降级返回原生 HTML"""
    if os.path.exists(os.path.join(FRONTEND_DIST, "index.html")):
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
    return FileResponse(FALLBACK_INDEX)


@app.get("/agent")
async def agent_page():
    """客服后台路由"""
    if os.path.exists(os.path.join(FRONTEND_DIST, "index.html")):
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
    return FileResponse("web/agent.html")


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "version": "0.2.0"}


# ====== 路由注册 ======
from src.api.mock import router as mock_router
from src.api.websocket import router as ws_router
app.include_router(mock_router, prefix="/api/mock", tags=["Mock"])
app.include_router(ws_router, tags=["WebSocket"])


# ====== 静态文件（放在路由最后，避免 /api /ws 被静态接管）======
if os.path.exists(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="static")
else:
    # 兜底：原生 HTML
    app.mount("/web", StaticFiles(directory="web", html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=True,
    )
