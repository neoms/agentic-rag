"""幻觉检测模块 - 验证生成答案是否忠实于参考文档

从 rag_service.py 拆分为独立模块，职责：
1. check_hallucination_async — 异步调用 LLM 检测答案忠实度
2. 返回 (passed: bool, faithfulness: float) 供调用方使用
"""

import json
import re
import logging

from langchain_core.messages import HumanMessage

from src.agent.prompts import CHECK_HALLUCINATION_SYSTEM, CHECK_HALLUCINATION_USER
from src.backend.llm import create_fast_llm

logger = logging.getLogger(__name__)


async def check_hallucination_async(
    documents: list,
    answer: str,
    max_docs: int = 8,
) -> tuple[bool, float]:
    """异步幻觉检测

    调用 LLM 检查生成的答案是否忠实于参考文档。
    返回布尔值判定 + 忠实度分数（0.0~100.0）。

    Args:
        documents: 参考文档列表
        answer: 生成的答案文本
        max_docs: 最多传入 LLM 的参考文档数

    Returns:
        (passed: bool, faithfulness: float 0.0~100.0)
    """
    passed = True
    faithfulness = 100.0

    docs_for_check = "\n---\n".join(
        f"[文档 {i+1}] {doc.page_content[:500]}"
        for i, doc in enumerate(documents[:max_docs])
    )

    try:
        check_llm = create_fast_llm()
        check_messages = [
            HumanMessage(content=CHECK_HALLUCINATION_SYSTEM),
            HumanMessage(
                content=(
                    CHECK_HALLUCINATION_USER.format(
                        documents=docs_for_check,
                        answer=answer,
                    )
                    + "\n\n输出要求：请返回一个 JSON 对象，包含两个字段："
                    '"passed" (布尔值，true 表示答案忠实于文档，false 表示存在编造)，'
                    '"faithfulness" (浮点数，0.0~100.0，精确到小数点后一位，表示答案对文档的忠实度百分比)。'
                    '只输出 JSON，不要输出其他内容。'
                    '示例：{"passed": true, "faithfulness": 92.5}'
                ),
            ),
        ]
        check_response = await check_llm.ainvoke(check_messages)
        check_raw = check_response.content.strip()

        json_match = re.search(r'\{[^{}]*\}', check_raw)
        if json_match:
            data = json.loads(json_match.group())
            faithfulness = float(data.get("faithfulness", 100))
            faithfulness = max(0.0, min(100.0, round(faithfulness, 1)))
            passed = data.get("passed", True)
        else:
            passed = "PASSED" in check_raw.upper() or "true" in check_raw.lower()

        logger.info(
            "幻觉检测: faithfulness=%.1f%%, passed=%s",
            faithfulness, str(passed),
        )
    except Exception as e:
        logger.warning("幻觉检测异常: %s", e)

    return passed, faithfulness
