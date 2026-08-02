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
    # 清理崩溃残留的临时上传文件（启动瞬间无在途任务，清空安全）
    try:
        temp_dir = settings.project_root / "data" / "temp_uploads"
        if temp_dir.exists():
            removed = 0
            for f in temp_dir.iterdir():
                if f.is_file():
                    f.unlink(missing_ok=True)
                    removed += 1
            if removed:
                logger.info("已清理临时上传目录 %d 个残留文件", removed)
    except Exception as e:
        logger.warning("临时文件清理失败: %s", e)
    try:
        stats = vector_store.get_collection_stats()
        logger.info("向量集合: %s, 文档数: %d", stats["name"], stats["count"])
    except Exception:
        logger.warning("ChromaDB 初始化失败")
    yield
    document_service.shutdown(wait=True)
    # 关闭已初始化的 SQLite 连接（懒加载单例未创建则跳过）
    try:
        from src.store import state_store as state_store_mod
        if state_store_mod._store is not None:
            state_store_mod._store.close()
    except Exception as e:
        logger.warning("state_db 关闭异常: %s", e)
    try:
        from src.cache import _service as cache_svc
        if cache_svc is not None:
            cache_svc.close()
    except Exception as e:
        logger.warning("cache_db 关闭异常: %s", e)
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
    summary="健康检查（组件明细）",
    description="逐项检查 ChromaDB / 运行时状态库 / 多级缓存 / 知识图谱 / 配置；"
                "deep=true 时额外做一次真实 Embedding 探针调用验证凭据。",
    tags=["system"],
)
async def health(deep: bool = False):
    checks: dict[str, str] = {}

    # 1) ChromaDB 向量库
    try:
        stats = vector_store.get_collection_stats()
        checks["chroma"] = f"ok (docs={stats.get('count', '?')})"
    except Exception as e:
        checks["chroma"] = f"error: {e}"

    # 2) 运行时状态库（会话历史 / 上传任务）
    try:
        from src.store.state_store import get_runtime_state_store
        get_runtime_state_store()
        checks["state_db"] = "ok"
    except Exception as e:
        checks["state_db"] = f"error: {e}"

    # 3) 多级缓存库
    try:
        from src.cache import get_cache_service
        cache_stats = get_cache_service().stats()
        checks["cache"] = f"ok (entries={cache_stats.get('total', '?')})"
    except Exception as e:
        checks["cache"] = f"error: {e}"

    # 4) 知识图谱（Kuzu）
    try:
        from src.knowledge_graph import get_graph_store
        checks["kg"] = f"ok (nodes={get_graph_store().node_count})"
    except Exception as e:
        checks["kg"] = f"error: {e}"

    # 5) 配置完整性（不发外部请求）
    checks["config"] = (
        "ok"
        if settings.dashscope_api_key
        else "error: DASHSCOPE_API_KEY 未配置"
    )

    # 6) 深度探测：真实 Embedding 调用（仅 deep=true 时，避免轮询花钱）
    if deep:
        try:
            from src.backend.embedding import get_embedding_client
            vec = get_embedding_client().embed_query("health probe")
            checks["embedding"] = f"ok (dim={len(vec)})" if vec else "error: 空向量"
        except Exception as e:
            checks["embedding"] = f"error: {e}"

    status = "ok" if all(v.startswith("ok") for v in checks.values()) else "degraded"
    return {"status": status, "version": "0.1.0", "checks": checks}


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
