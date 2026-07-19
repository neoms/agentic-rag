"""百炼 LLM 客户端工厂 - 基于 langchain-openai 的 ChatOpenAI"""

import logging
from langchain_openai import ChatOpenAI
from src.config.settings import settings

logger = logging.getLogger(__name__)


def create_llm_client(
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
) -> ChatOpenAI:
    """创建百炼 ChatOpenAI 客户端

    Args:
        model: 模型名，默认使用 settings.llm_model（qwen-plus）
        temperature: 温度参数
        max_tokens: 最大输出 Token
        streaming: 是否启用流式输出

    Returns:
        ChatOpenAI 实例
    """
    return ChatOpenAI(
        model=model or settings.llm_model,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        max_tokens=max_tokens or settings.llm_max_tokens,
        streaming=streaming,
        api_key=settings.dashscope_api_key,
        base_url=settings.llm_base_url,
    )


def create_fast_llm(streaming: bool = False) -> ChatOpenAI:
    """创建快速 LLM 客户端（qwen-turbo），用于评估、重排序等轻量任务"""
    return create_llm_client(
        model=settings.llm_model_fast,
        temperature=0.0,
        max_tokens=1024,
        streaming=streaming,
    )


def create_strong_llm(streaming: bool = False) -> ChatOpenAI:
    """创建强 LLM 客户端（qwen-max），用于最终答案生成"""
    return create_llm_client(
        model=settings.llm_model_strong,
        temperature=0.3,
        max_tokens=settings.llm_max_tokens,
        streaming=streaming,
    )
