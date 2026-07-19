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
        model: 嵌入模型名，默认使用 settings.embedding_model（text-embedding-v2）

    Returns:
        DashScopeEmbeddings 实例
    """
    return DashScopeEmbeddings(
        model=model or settings.embedding_model,
        dashscope_api_key=settings.dashscope_api_key,
    )


# 全局默认 Embedding 客户端（懒加载，避免在导入时就需要 API Key）
_embedding_client: DashScopeEmbeddings | None = None


def get_embedding_client() -> DashScopeEmbeddings:
    """获取全局单例 Embedding 客户端"""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = create_embedding_client()
    return _embedding_client
