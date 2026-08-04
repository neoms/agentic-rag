"""运行指标 - Prometheus 指标定义（prometheus-client）

指标注册到默认 REGISTRY（含进程级指标：内存/CPU/文件描述符），
由 GET /metrics 端点按 Prometheus 文本协议导出。
指标名使用稳定英文标识，HELP 说明中英双语。
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
    "缓存命中次数（按缓存类型）/ Cache hits by type",
    ["type"],  # exact / semantic
)
chat_stream_duration_seconds = Histogram(
    "chat_stream_duration_seconds",
    "Chat 流式请求耗时（秒）/ Chat stream request duration (seconds)",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120),
)
chat_ttft_seconds = Histogram(
    "chat_stream_ttft_seconds",
    "Chat 首 token 延迟（秒）/ Time to first token (seconds)",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60),
)
chat_stage_duration_seconds = Histogram(
    "chat_stage_duration_seconds",
    "各阶段耗时（秒，按阶段）/ Per-stage duration (seconds)",
    ["stage"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
chat_cache_saved_llm_calls = Counter(
    "chat_cache_saved_llm_calls",
    "缓存命中所节省的 LLM 调用数 / LLM calls saved by cache hits",
)
chat_errors_total = Counter(
    "chat_errors_total",
    "Chat 流式请求内部错误数 / Chat stream internal errors",
)

# ── LLM / Embedding 调用 ──
llm_calls_total = Counter(
    "llm_calls_total",
    "LLM API 调用次数（按模型，含重试尝试）/ LLM API calls by model",
    ["model"],
)
llm_tokens_total = Counter(
    "llm_tokens_total",
    "LLM token 用量（按模型与方向）/ LLM token usage by model and direction",
    ["model", "type"],  # type: input / output
)
chat_cost_estimate_total = Counter(
    "chat_cost_estimate_total",
    "LLM 成本估算（元，按模型）/ Estimated LLM cost (CNY) by model",
    ["model"],
)
embedding_calls_total = Counter(
    "embedding_calls_total",
    "Embedding API 调用次数 / Embedding API calls",
)

# ── 文档上传 ──
uploads_total = Counter(
    "uploads_total",
    "上传文档成功提交数 / Documents successfully submitted",
)
uploads_failed_total = Counter(
    "uploads_failed_total",
    "上传文档失败数（超限/空文件/队列满等）/ Failed uploads",
)


def record_llm_tokens(model: str, input_tokens: int, output_tokens: int) -> None:
    """记录一次 LLM 调用的 token 用量与成本估算（单价为 0 时不计成本）"""
    from src.config.settings import settings

    input_tokens = max(0, int(input_tokens or 0))
    output_tokens = max(0, int(output_tokens or 0))
    if input_tokens:
        llm_tokens_total.labels(model=model, type="input").inc(input_tokens)
    if output_tokens:
        llm_tokens_total.labels(model=model, type="output").inc(output_tokens)
    cost = (
        settings.llm_price_input_per_1m * input_tokens
        + settings.llm_price_output_per_1m * output_tokens
    ) / 1_000_000
    if cost > 0:
        chat_cost_estimate_total.labels(model=model).inc(cost)


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
