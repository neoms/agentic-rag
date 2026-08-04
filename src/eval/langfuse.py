"""Langfuse 观测接入 - 懒加载单例 + 全链路优雅降级

职责：
- 生产追踪：为 rag_service 的每次请求创建 trace，LangGraph 图内 span 经
  CallbackHandler 关联到同一条 trace
- 请求级元数据：以 "request" span 记录 query/answer/检索上下文/耗时，
  供在线评估采样读取
- 打分：反馈（user_feedback）与在线评估分数写回 trace
- 未配置 LANGFUSE_PUBLIC_KEY/SECRET_KEY 时所有函数安全返回 None/False
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from src.config.settings import settings

logger = logging.getLogger(__name__)

_client: Langfuse | None = None
_client_lock = threading.Lock()


def langfuse_enabled() -> bool:
    """Langfuse 是否已配置（公钥/私钥成对）"""
    return bool(
        settings.langfuse_public_key.strip()
        and settings.langfuse_secret_key.strip()
    )


def get_langfuse_client() -> Langfuse | None:
    """获取全局单例 Langfuse 客户端（未配置返回 None）"""
    global _client
    if not langfuse_enabled():
        return None
    if _client is None:
        with _client_lock:
            if _client is None:
                logger.info("初始化 Langfuse 客户端: host=%s", settings.langfuse_host)
                _client = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
    return _client


def create_trace_id() -> str | None:
    """创建 trace id（未配置返回 None）"""
    client = get_langfuse_client()
    return client.create_trace_id() if client else None


def build_graph_callback(trace_id: str | None) -> CallbackHandler | None:
    """构建 LangGraph 图内 span 关联用的 CallbackHandler（未配置返回 None）"""
    if not trace_id or get_langfuse_client() is None:
        return None
    return CallbackHandler(trace_context={"trace_id": trace_id})


def attach_request_span(
    trace_id: str | None,
    *,
    input_data: dict[str, Any],
    output_data: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    """给 trace 附加请求级 span（含 answer/检索上下文，供在线评估读取）

    在线评估从该 span 的 output 中读取：answer / sources / latency_seconds /
    cache_type，从 input 读取 query。
    """
    if not trace_id:
        return
    client = get_langfuse_client()
    if client is None:
        return
    try:
        span = client.start_observation(
            trace_context={"trace_id": trace_id},
            name="request",
            as_type="span",
            input=input_data,
            output=output_data,
            metadata=metadata,
        )
        span.end()
    except Exception:
        logger.warning("Langfuse 请求 span 写入失败 (trace=%s)", trace_id, exc_info=True)


def score_trace(
    trace_id: str,
    name: str,
    value: float,
    comment: str | None = None,
) -> bool:
    """给 trace 写一个分数（如 user_feedback / 在线评估指标），成功返回 True"""
    client = get_langfuse_client()
    if client is None or not trace_id:
        return False
    try:
        client.create_score(
            trace_id=trace_id,
            name=name,
            value=round(float(value), 4),
            comment=comment,
        )
        client.flush()
        return True
    except Exception:
        logger.warning("Langfuse 打分失败: trace=%s name=%s", trace_id, name, exc_info=True)
        return False


def record_feedback(trace_id: str, rating: int, comment: str | None = None) -> bool:
    """用户反馈（👍/👎，1-5 分）写回 Langfuse"""
    return score_trace(trace_id, name="user_feedback", value=rating / 5.0, comment=comment)


def flush() -> None:
    """主动刷新待上报事件（关闭前调用）"""
    client = get_langfuse_client()
    if client is not None:
        try:
            client.flush()
        except Exception:
            logger.debug("Langfuse flush 失败", exc_info=True)


def reset_langfuse() -> None:
    """重置单例（测试用）"""
    global _client
    _client = None
