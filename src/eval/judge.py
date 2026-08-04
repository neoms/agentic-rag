"""LLM-as-judge 评判模型 - 统一走独立强评判模型（EVAL_JUDGE_MODEL）

约定：
- 评判模型一律从配置读取（EVAL_JUDGE_MODEL），代码不写死任何模型名
- 未配置时回退 LLM_MODEL_STRONG，并在报告中标注"judge 与被测同源"
- RAGAS 指标统一注入该 judge 的 llm / embeddings
"""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from src.backend.embedding import get_embedding_client
from src.backend.llm import create_llm_client
from src.config.settings import settings

logger = logging.getLogger(__name__)

_judge_llm: ChatOpenAI | None = None
_judge_ragas_llm = None
_judge_ragas_embeddings = None


def judge_isolated() -> bool:
    """judge 是否与被测生成模型不同源（配置了独立评判模型）"""
    model = settings.eval_judge_model.strip()
    return bool(model and model != settings.llm_model_strong)


def judge_model_name() -> str:
    """实际使用的 judge 模型名（供报告标注）"""
    return settings.eval_judge_model.strip() or settings.llm_model_strong


def get_judge_llm() -> ChatOpenAI:
    """获取 judge 的 ChatOpenAI 客户端（懒加载 + 缓存）"""
    global _judge_llm
    if _judge_llm is None:
        model = judge_model_name()
        logger.info("初始化 judge LLM: model=%s (isolated=%s)", model, judge_isolated())
        _judge_llm = create_llm_client(
            model=model,
            temperature=0.0,
            max_tokens=2048,
            api_key=settings.eval_judge_api_key.strip() or None,
            base_url=settings.eval_judge_base_url.strip() or None,
            extra_body=settings.eval_judge_extra_body_dict,
        )
    return _judge_llm


def get_judge_ragas_llm():
    """RAGAS 使用的 judge LLM wrapper"""
    global _judge_ragas_llm
    if _judge_ragas_llm is None:
        _judge_ragas_llm = LangchainLLMWrapper(get_judge_llm())
    return _judge_ragas_llm


def get_judge_ragas_embeddings():
    """RAGAS answer_relevancy 等指标使用的 embeddings wrapper"""
    global _judge_ragas_embeddings
    if _judge_ragas_embeddings is None:
        _judge_ragas_embeddings = LangchainEmbeddingsWrapper(get_embedding_client())
    return _judge_ragas_embeddings


def reset_judge() -> None:
    """重置 judge 缓存（测试用）"""
    global _judge_llm, _judge_ragas_llm, _judge_ragas_embeddings
    _judge_llm = None
    _judge_ragas_llm = None
    _judge_ragas_embeddings = None
