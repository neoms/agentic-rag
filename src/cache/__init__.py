"""多级缓存模块 - 精准缓存 + 语义缓存"""

from src.cache.service import CacheService

_service: CacheService | None = None


def get_cache_service() -> CacheService:
    """懒加载全局单例（首次请求时才创建存储与索引）"""
    global _service
    if _service is None:
        _service = CacheService()
    return _service


def reset_cache_service() -> None:
    """重置单例（测试用）"""
    global _service
    _service = None
