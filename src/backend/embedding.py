"""百炼 Embedding 客户端工厂 - 基于 langchain-openai 的 OpenAIEmbeddings"""

import logging
from langchain_openai import OpenAIEmbeddings
from src.config.settings import settings

logger = logging.getLogger(__name__)


def create_embedding_client(
    model: str | None = None,
) -> OpenAIEmbeddings:
    """创建百炼 Embedding 客户端

    Args:
        model: 嵌入模型名，默认使用 settings.embedding_model（text-embedding-v2）

    Returns:
        OpenAIEmbeddings 实例
    """
    return OpenAIEmbeddings(
        model=model or settings.embedding_model,
        api_key=settings.dashscope_api_key,
        base_url=settings.llm_base_url,
    )


# 全局默认 Embedding 客户端（懒加载，避免在导入时就需要 API Key）
_embedding_client: OpenAIEmbeddings | None = None


def get_embedding_client() -> OpenAIEmbeddings:
    """获取全局单例 Embedding 客户端"""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = create_embedding_client()
    return _embedding_client
