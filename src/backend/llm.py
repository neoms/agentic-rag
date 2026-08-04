"""百炼 LLM 客户端工厂 - 基于 langchain-openai 的 ChatOpenAI

所有客户端创建时自动注入 tenacity 指数退避重试与请求超时，
覆盖 invoke（同步）和 ainvoke（异步）方法。
"""

import functools
import json
import logging

import tenacity
from langchain_openai import ChatOpenAI

from src.config.settings import settings
from src.metrics import llm_calls_total

logger = logging.getLogger(__name__)

# 模块级客户端缓存：按 (model, temperature, max_tokens, streaming, api_key,
# base_url, extra_body) 组合缓存 ChatOpenAI 实例，进程内永久复用。
# ChatOpenAI 为无状态客户端，跨请求复用安全；组合数量有限，不会无界增长
_client_cache: dict[tuple[str, float, int, bool, str, str, str], ChatOpenAI] = {}


def _should_retry(exception: BaseException) -> bool:
    """判断异常是否值得重试

    仅对网络/服务端类的临时错误重试，客户端参数错误直接透传。
    """
    msg = str(exception).lower()
    # 服务端错误 / 限流 / 超时 / 连接错误
    return any(kw in msg for kw in (
        "5xx", "500", "502", "503", "504",
        "rate limit", "too many requests",
        "timeout", "timed out",
        "connection", "reset", "refused",
        "service unavailable",
        "server error",
        "internal error",
        "temporary",
    ))


def _build_retry() -> tenacity.retry:
    """构建带指数退避的 tenacity 重试装饰器"""
    return tenacity.retry(
        stop=tenacity.stop_after_attempt(settings.llm_max_retries),
        wait=tenacity.wait_exponential(
            multiplier=1,
            min=settings.llm_retry_min_wait,
            max=settings.llm_retry_max_wait,
        ),
        retry=tenacity.retry_if_exception(_should_retry),
        before_sleep=lambda retry_state: logger.warning(
            "LLM 调用失败 (尝试 %d/%d, 等待 %.1fs): %s",
            retry_state.attempt_number,
            settings.llm_max_retries,
            retry_state.next_action.sleep if retry_state.next_action else 0,
            retry_state.outcome.exception(),
        ),
        reraise=True,
    )


def _install_retry_on_llm(llm: ChatOpenAI) -> ChatOpenAI:
    """在 ChatOpenAI 实例上安装重试包装

    替换 invoke（同步）和 ainvoke（异步）方法为带重试的版本。
    stream / astream 返回生成器，不重试（错误在迭代时捕获）。
    """
    retry_decorator = _build_retry()
    model = getattr(llm, "model", "unknown")

    # 同步 invoke 重试
    original_invoke = llm.invoke

    @functools.wraps(original_invoke)
    @retry_decorator
    def retry_invoke(*args, **kwargs):
        llm_calls_total.labels(model=model).inc()
        return original_invoke(*args, **kwargs)

    # 使用 object.__setattr__ 绕过 Pydantic v2 的字段验证
    # ChatOpenAI 是 Pydantic 模型，直接属性赋值会触发 "no field" 异常
    object.__setattr__(llm, "invoke", retry_invoke)

    # 异步 ainvoke 重试
    original_ainvoke = llm.ainvoke

    @functools.wraps(original_ainvoke)
    @retry_decorator
    async def retry_ainvoke(*args, **kwargs):
        llm_calls_total.labels(model=model).inc()
        return await original_ainvoke(*args, **kwargs)

    object.__setattr__(llm, "ainvoke", retry_ainvoke)

    # stream / astream：不做重试，仅计数调用次数
    original_stream = llm.stream

    @functools.wraps(original_stream)
    def counting_stream(*args, **kwargs):
        llm_calls_total.labels(model=model).inc()
        yield from original_stream(*args, **kwargs)

    object.__setattr__(llm, "stream", counting_stream)

    original_astream = llm.astream

    @functools.wraps(original_astream)
    async def counting_astream(*args, **kwargs):
        llm_calls_total.labels(model=model).inc()
        async for chunk in original_astream(*args, **kwargs):
            yield chunk

    object.__setattr__(llm, "astream", counting_astream)

    logger.debug("LLM 重试已安装: max_retries=%d, timeout=%ds",
                 settings.llm_max_retries, settings.llm_request_timeout)
    return llm


def create_llm_client(
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
    api_key: str | None = None,
    base_url: str | None = None,
    extra_body: dict | None = None,
) -> ChatOpenAI:
    """创建百炼 ChatOpenAI 客户端（自动注入重试与超时）

    Args:
        model: 模型名，默认使用 settings.llm_model（环境变量 LLM_MODEL）
        temperature: 温度参数
        max_tokens: 最大输出 Token
        streaming: 是否启用流式输出
        api_key: 覆盖 API Key（默认 settings.dashscope_api_key，评估 judge 可独立配置）
        base_url: 覆盖 API 地址（默认 settings.llm_base_url）
        extra_body: 额外请求体参数（如 enable_thinking 等模型参数）

    Returns:
        ChatOpenAI 实例，invoke/ainvoke 已包装重试
    """
    final_model = model or settings.llm_model
    final_temp = temperature if temperature is not None else settings.llm_temperature
    final_tokens = max_tokens or settings.llm_max_tokens
    final_api_key = api_key or settings.dashscope_api_key
    final_base_url = base_url or settings.llm_base_url

    # 复用已创建的客户端，避免每个请求重复实例化并重装重试包装
    extra_key = json.dumps(extra_body or {}, sort_keys=True)
    cache_key = (final_model, final_temp, final_tokens, streaming, final_api_key, final_base_url, extra_key)
    cached = _client_cache.get(cache_key)
    if cached is not None:
        logger.debug("复用 LLM 客户端: %s (key=%s)", final_model, cache_key)
        return cached

    logger.info("创建 LLM 客户端: model=%s, temperature=%.2f, max_tokens=%d, streaming=%s, base_url=%s",
                final_model, final_temp, final_tokens, streaming, final_base_url)
    llm = ChatOpenAI(
        model=final_model,
        temperature=final_temp,
        max_tokens=final_tokens,
        streaming=streaming,
        api_key=final_api_key,
        base_url=final_base_url,
        extra_body=extra_body,
        timeout=settings.llm_request_timeout,
        max_retries=0,  # 禁用内置重试，使用 tenacity 统一管理
    )
    llm = _install_retry_on_llm(llm)
    _client_cache[cache_key] = llm
    return llm


def create_fast_llm(
    streaming: bool = False,
    extra_body: dict | None = None,
) -> ChatOpenAI:
    """获取快速 LLM 客户端（settings.llm_model_fast，环境变量 LLM_MODEL_FAST），用于评估、重排序等轻量任务

    默认关闭思考模式（复用 GENERATION_EXTRA_BODY 配置）：qwen3.x 系列默认开
    思考，轻量分类/抽取/改写任务开思考只会增加延迟，且非流式调用思考模型
    可能直接报错或长时间无输出。显式传入 extra_body 可覆盖默认值。
    """
    if extra_body is None:
        extra_body = settings.generation_extra_body_dict
    logger.debug("获取快速 LLM 客户端: model=%s", settings.llm_model_fast)
    return create_llm_client(
        model=settings.llm_model_fast,
        temperature=0.0,
        max_tokens=1024,
        streaming=streaming,
        extra_body=extra_body,
    )


def create_strong_llm(
    streaming: bool = False,
    extra_body: dict | None = None,
) -> ChatOpenAI:
    """创建强 LLM 客户端（settings.llm_model_strong，环境变量 LLM_MODEL_STRONG），用于最终答案生成"""
    logger.info("创建强 LLM: model=%s, streaming=%s", settings.llm_model_strong, streaming)
    return create_llm_client(
        model=settings.llm_model_strong,
        temperature=0.3,
        max_tokens=settings.llm_max_tokens,
        streaming=streaming,
        extra_body=extra_body,
    )
