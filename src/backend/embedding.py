"""百炼 Embedding 客户端工厂 - 基于 langchain_community 官方集成

注入合理的请求超时，避免 dashscope SDK 默认 300s 超时导致前端卡死。
"""

import logging
from typing import List

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.embeddings.dashscope import embed_with_retry

from src.config.settings import settings
from src.metrics import embedding_calls_total

logger = logging.getLogger(__name__)

# Embedding API 超时（秒），dashscope SDK 默认 300s 太长
EMBEDDING_REQUEST_TIMEOUT = 60


class TimeoutAwareDashScopeEmbeddings(DashScopeEmbeddings):
    """带可控超时的 DashScopeEmbeddings

    原理：
        dashscope SDK 的 TextEmbedding.call() 支持通过 **kwargs 传递
        request_timeout 参数。但原生 DashScopeEmbeddings 不暴露此参数，
        这里手动注入。
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """注入 request_timeout 到 embedding API 调用"""
        embedding_calls_total.inc()
        embeddings = embed_with_retry(
            self,
            input=texts,
            text_type="document",
            model=self.model,
            request_timeout=EMBEDDING_REQUEST_TIMEOUT,
        )
        return [item["embedding"] for item in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """注入 request_timeout 到 query embedding API 调用"""
        embedding_calls_total.inc()
        embedding = embed_with_retry(
            self,
            input=text,
            text_type="query",
            model=self.model,
            request_timeout=EMBEDDING_REQUEST_TIMEOUT,
        )[0]["embedding"]
        return embedding


def create_embedding_client(
    model: str | None = None,
) -> TimeoutAwareDashScopeEmbeddings:
    """创建百炼 Embedding 客户端（带超时控制）

    Args:
        model: 嵌入模型名，默认使用 settings.embedding_model

    Returns:
        TimeoutAwareDashScopeEmbeddings 实例
    """
    final_model = model or settings.embedding_model
    logger.info(
        "创建 Embedding 客户端: model=%s, timeout=%ds",
        final_model, EMBEDDING_REQUEST_TIMEOUT,
    )
    return TimeoutAwareDashScopeEmbeddings(
        model=final_model,
        dashscope_api_key=settings.dashscope_api_key,
    )


# 全局默认 Embedding 客户端（懒加载，避免在导入时就需要 API Key）
_embedding_client: TimeoutAwareDashScopeEmbeddings | None = None


def get_embedding_client() -> TimeoutAwareDashScopeEmbeddings:
    """获取全局单例 Embedding 客户端"""
    global _embedding_client
    if _embedding_client is None:
        logger.info("首次初始化全局 Embedding 客户端")
        _embedding_client = create_embedding_client()
    return _embedding_client
