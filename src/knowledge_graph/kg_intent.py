"""KGIntentAnalyzer - LLM 驱动的问题意图分析

分析用户问题是否适合使用知识图谱来回答。
"""

import logging

from src.agent.prompts import (
    KG_INTENT_ANALYZE_SYSTEM,
    KG_INTENT_ANALYZE_USER,
)
from src.backend.llm import create_fast_llm

logger = logging.getLogger(__name__)


class KGIntentAnalyzer:
    """使用快速 LLM 分析问题是否需要知识图谱

    判断标准：
        - 实体关系查询 → SHOULD_USE_KG
        - 多跳推理 → SHOULD_USE_KG
        - 比较对比 → SHOULD_USE_KG
        - 简单定义/解释 → SHOULD_NOT_USE_KG
        - 操作指引 → SHOULD_NOT_USE_KG
    """

    def analyze(self, query: str) -> bool:
        """分析问题意图，判断是否需要知识图谱

        Args:
            query: 用户问题

        Returns:
            True 表示应该使用知识图谱，False 表示降级到常规 RAG
        """
        logger.info("KG 意图分析: query='%s'", query[:100])

        try:
            llm = create_fast_llm()
            prompt = (
                f"{KG_INTENT_ANALYZE_SYSTEM}\n\n"
                f"{KG_INTENT_ANALYZE_USER.format(query=query)}"
            )
            response = llm.invoke(prompt)
            result = response.content.strip().upper()
            should_use = result == "SHOULD_USE_KG"
            logger.info("KG 意图分析结果: %s (raw='%s')",
                         "SHOULD_USE_KG" if should_use else "SHOULD_NOT_USE_KG",
                         result)
            return should_use

        except Exception as e:
            logger.warning("KG 意图分析异常，默认降级: %s", e)
            return False
