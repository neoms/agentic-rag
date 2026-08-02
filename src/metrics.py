"""运行指标 - Prometheus 指标定义（prometheus-client）

指标注册到默认 REGISTRY（含进程级指标：内存/CPU/文件描述符），
由 GET /metrics 端点按 Prometheus 文本协议导出。
"""

import logging

from prometheus_client import REGISTRY, PROCESS_COLLECTOR, Counter, Histogram

logger = logging.getLogger(__name__)

# ── 对话与缓存 ──
chat_requests_total = Counter(
    "chat_requests_total",
    "Chat 流式请求总数（含缓存命中）",
)
chat_cache_hit_total = Counter(
    "chat_cache_hit_total",
    "缓存命中次数（按缓存类型）",
    ["type"],  # exact / semantic
)
chat_stream_duration_seconds = Histogram(
    "chat_stream_duration_seconds",
    "Chat 流式请求耗时（秒）",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120),
)
chat_errors_total = Counter(
    "chat_errors_total",
    "Chat 流式请求内部错误数",
)

# ── LLM / Embedding 调用 ──
llm_calls_total = Counter(
    "llm_calls_total",
    "LLM API 调用次数（按模型，含重试尝试）",
    ["model"],
)
embedding_calls_total = Counter(
    "embedding_calls_total",
    "Embedding API 调用次数",
)

# ── 文档上传 ──
uploads_total = Counter(
    "uploads_total",
    "文档上传成功提交数",
)
uploads_failed_total = Counter(
    "uploads_failed_total",
    "文档上传失败数（超限/空文件/队列满等）",
)


def _register_process_collector() -> None:
    """注册进程级指标（内存/CPU/文件描述符）

    0.26.0 起默认不再自动注册；Linux 可直接读 /proc，
    macOS/Windows 需要 psutil，缺失时跳过避免 /metrics 采集失败。
    """
    try:
        import psutil  # noqa: F401
    except ImportError:
        try:
            with open("/proc/self/stat", "rb"):
                pass
        except FileNotFoundError:
            return
    try:
        REGISTRY.register(PROCESS_COLLECTOR)
    except ValueError:
        pass  # 已注册


_register_process_collector()
