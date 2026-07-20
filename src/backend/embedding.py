"""百炼 Embedding 客户端工厂 - 基于 langchain_community 官方集成"""

import logging
from langchain_community.embeddings import DashScopeEmbeddings
from src.config.settings import settings

logger = logging.getLogger(__name__)


def create_embedding_client(
    model: str | None = None,
) -> DashScopeEmbeddings:
    """创建百炼 Embedding 客户端

    Args:
        model: 嵌入模型名，默认使用 settings.embedding_model

    Returns:
        DashScopeEmbeddings 实例
    """
    final_model = model or settings.embedding_model
    logger.info("创建 Embedding 客户端: model=%s", final_model)
    return DashScopeEmbeddings(
        model=final_model,
        dashscope_api_key=settings.dashscope_api_key,
    )


# 全局默认 Embedding 客户端（懒加载，避免在导入时就需要 API Key）
_embedding_client: DashScopeEmbeddings | None = None


def get_embedding_client() -> DashScopeEmbeddings:
    """获取全局单例 Embedding 客户端"""
    global _embedding_client
    if _embedding_client is None:
        logger.info("首次初始化全局 Embedding 客户端")
        _embedding_client = create_embedding_client()
    return _embedding_client
