"""文本生成模块 - 带引文标注的文档格式化与 LLM 流式生成

从 rag_service.py 拆分为独立模块，职责：
1. format_documents_with_citations — 文档 → 带段落索引的文本 + 引文元数据
2. build_generate_prompt — 构建带引文标注要求的 LLM 生成 prompt
3. build_generate_node_data / build_hallucination_node_data — 前端流程图 I/O 数据
"""

import re
import logging

logger = logging.getLogger(__name__)


def format_documents_with_citations(
    documents: list,
    max_docs: int = 8,
) -> tuple[str, dict[str, dict]]:
    """构建带段落索引的文档上下文 + 对应引文元数据

    每篇文档按双换行拆分段落，为每段分配 [DocX-ParaY] 标识。
    引文元数据在前端 SourcePanel 中用于高亮引用段落。

    Args:
        documents: 文档列表
        max_docs: 最多处理的文档数

    Returns:
        (docs_text, citation_metadata) 元组
    """
    doc_parts: list[str] = []
    citation_metadata: dict[str, dict] = {}

    for doc_idx, doc in enumerate(documents[:max_docs], 1):
        src = doc.metadata.get("url") or doc.metadata.get("filename", "unknown")
        url_info = f"\n链接: {doc.metadata['url']}" if doc.metadata.get("url") else ""
        source_type = doc.metadata.get("source", "local")
        url = doc.metadata.get("url", "")

        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', doc.page_content) if p.strip()]
        if not paragraphs:
            paragraphs = [doc.page_content]

        para_lines = []
        for para_idx, para in enumerate(paragraphs, 1):
            citation_key = f"Doc{doc_idx}-Para{para_idx}"
            para_lines.append(f"  [{citation_key}] {para}")
            citation_metadata[citation_key] = {
                "filename": src,
                "source_type": source_type,
                "url": url,
                "paragraph_text": para,
                "doc_index": doc_idx,
                "para_index": para_idx,
            }
        doc_text = "\n\n".join(para_lines)
        doc_parts.append(f"来源: {src}{url_info}\n内容:\n{doc_text}")

    docs_text = "\n\n---\n\n".join(doc_parts)
    return docs_text, citation_metadata


def build_generate_prompt(
    query: str,
    docs_text: str,
    chat_history: str,
) -> str:
    """构建带引文标注要求的生成 prompt"""
    return f"""你是一个专业的知识问答助手。请基于提供的文档上下文回答用户问题。

规则：
1. 优先使用提供的文档信息回答；信息不足时明确说明
2. 回答简洁准确有条理，使用中文
3. 【重要】每句陈述性内容末尾都必须标注来源，格式为 [DocX-ParaY]
   - 单源标注: "Python是动态类型语言 [Doc1-Para2]。"
   - 多源标注: "机器学习分为三类 [Doc1-Para1, Doc2-Para3]。"
4. 每句话至少有一个引用标注（总结句可多源标注），没有来源的陈述不要写
5. 严格使用文档中提供的 [DocX-ParaY] 标识

文档上下文：
{docs_text}

对话历史：
{chat_history or '无'}

用户问题：{query}

请回答（每句话末尾都标注来源）："""


def build_generate_node_data(
    query: str,
    documents: list,
    answer: str,
    max_docs: int = 8,
) -> dict:
    """构建 generate 节点的 I/O 数据（前端流程图展示用）"""
    gen_input: list[str] = [f"问题: {query}"]
    for i, doc in enumerate(documents[:max_docs]):
        src = doc.metadata.get("url") or doc.metadata.get("filename", f"文档{i+1}")
        preview = doc.page_content[:100].replace("\n", " ")
        gen_input.append(f"参考 {src}:\n{preview}...")
    if len(documents) > max_docs:
        gen_input.append(f"... 及其他 {len(documents) - max_docs} 条")
    return {"input": gen_input, "output": answer}


def build_hallucination_node_data(
    answer: str,
    faithfulness: float,
    passed: bool,
) -> dict:
    """构建 check_hallucination 节点的 I/O 数据（前端流程图展示用）"""
    return {
        "input": [
            f"待检测答案 ({len(answer)} 字符):",
            answer[:300] + ("..." if len(answer) > 300 else ""),
        ],
        "output": [
            f"忠实度: {faithfulness}%",
            f"判定: {'PASSED ✓' if passed else 'FAILED ✗'}",
            f"结果: {'答案忠实于参考文档' if passed else '答案存在编造，需要重试'}",
        ],
    }
