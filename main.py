"""应用入口 - 启动 FastAPI 服务"""

from src.main import app

if __name__ == "__main__":
    import uvicorn
    from src.config.settings import settings

    uvicorn.run(
        "src.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
    )
