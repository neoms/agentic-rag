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
    citation_metadata: dict | None = None,
) -> tuple[bool, float]:
    """异步幻觉检测

    调用 LLM 检查生成的答案是否忠实于参考文档。
    返回布尔值判定 + 忠实度分数（0.0~100.0）。

    只基于答案通过 [编号] 引用标注实际引用的文档进行判定，
    避免把无关的检索结果混入上下文导致误判（实测把全部检索结果
    喂给检查器会把"答案未被无关文档支撑"误判成"答案有编造"）。

    Args:
        documents: 参考文档列表
        answer: 生成的答案文本
        max_docs: 最多传入 LLM 的参考文档数
        citation_metadata: 生成节点的引文元数据（doc_index 映射），
            用于筛选答案实际引用的文档；缺省时回退使用全部文档

    Returns:
        (passed: bool, faithfulness: float 0.0~100.0)
    """
    passed = True
    faithfulness = 100.0

    docs_for_check_docs = _select_referenced_documents(
        documents, answer, citation_metadata, max_docs,
    )
    docs_for_check = "\n---\n".join(
        f"[文档 {i+1}] {doc.page_content[:500]}"
        for i, doc in enumerate(docs_for_check_docs)
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
            try:
                data = json.loads(json_match.group())
                passed = bool(data.get("passed"))
                faithfulness = float(data.get("faithfulness", 100))
                faithfulness = max(0.0, min(100.0, round(faithfulness, 1)))
            except (json.JSONDecodeError, ValueError, TypeError):
                # 解析失败 → fail-closed：不静默放行
                logger.warning("幻觉检测 JSON 解析失败，按未通过处理: %s", check_raw[:120])
                passed = False
                faithfulness = 0.0
        else:
            # 模型未按指令输出 JSON → fail-closed：不静默放行
            logger.warning("幻觉检测未返回 JSON，按未通过处理: %s", check_raw[:120])
            passed = False
            faithfulness = 0.0

        logger.info(
            "幻觉检测: faithfulness=%.1f%%, passed=%s",
            faithfulness, str(passed),
        )
    except Exception as e:
        # 调用异常 → fail-closed：判定不可靠时不缓存（宁可少命中，不可缓存幻觉答案）
        logger.warning("幻觉检测异常，按未通过处理: %s", e)
        passed = False
        faithfulness = 0.0

    return passed, faithfulness


def _select_referenced_documents(
    documents: list,
    answer: str,
    citation_metadata: dict | None,
    max_docs: int,
) -> list:
    """按答案的 [编号] 引用标注筛选幻觉检测实际使用的文档

    citation_metadata 由生成节点提供：{编号: {doc_index, ...}}，
    doc_index 为 1 基索引，指向生成时使用的 documents 列表。
    无引用标注、映射失败或筛选为空时回退到全部文档（原行为）。
    """
    if not citation_metadata:
        return list(documents[:max_docs])
    cited = [int(m) for m in re.findall(r"\[(\d+)\]", answer or "")]
    if not cited:
        return list(documents[:max_docs])

    selected: list = []
    seen: set[int] = set()
    for seq in cited:
        meta = citation_metadata.get(str(seq))
        if not meta:
            continue
        idx = meta.get("doc_index")
        if not isinstance(idx, int) or idx < 1 or idx > len(documents):
            continue
        pos = idx - 1
        if pos in seen:
            continue
        seen.add(pos)
        selected.append(documents[pos])
        if len(selected) >= max_docs:
            break
    return selected or list(documents[:max_docs])
