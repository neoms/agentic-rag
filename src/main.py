"""Agentic RAG 应用主入口 - FastAPI 应用"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.router import api_router
from src.models.common import HealthResponse, ErrorResponse
from src.config.settings import settings
from src.store.vector_store import vector_store
from src.services.document_service import document_service

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("Agentic RAG 服务启动中...")
    logger.info("LLM 模型: %s", settings.llm_model)
    logger.info("Embedding 模型: %s", settings.embedding_model)
    logger.info("ChromaDB 持久化路径: %s", settings.chroma_persist_dir_path)
    try:
        stats = vector_store.get_collection_stats()
        logger.info("向量集合: %s, 文档数: %d", stats["name"], stats["count"])
    except Exception:
        logger.warning("ChromaDB 初始化失败")
    yield
    document_service.shutdown(wait=True)
    logger.info("Agentic RAG 服务关闭")


app = FastAPI(
    title="Agentic RAG API",
    description="基于 LangGraph 的多策略检索增强生成系统，支持自反思 Agent、工具调用、流式输出",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 中间件（白名单来源；不使用通配符 * 与 credentials 组合，避免非法 CORS 配置）
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("未处理的异常: %s", exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(detail="服务器内部错误", error_code="INTERNAL_ERROR").model_dump(),
    )


# 健康检查
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="健康检查",
    tags=["system"],
)
async def health():
    try:
        vector_store.get_collection_stats()
        return {"status": "ok", "version": "0.1.0"}
    except Exception:
        return {"status": "degraded", "version": "0.1.0"}


# 注册 API 路由
app.include_router(api_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
    )
