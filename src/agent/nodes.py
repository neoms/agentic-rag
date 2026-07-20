"""LangGraph Agent 节点实现 - 多策略检索、自反思、幻觉检测等"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.documents import Document
from langgraph.prebuilt import ToolNode

from src.agent.state import AgentState
from src.agent.prompts import (
    GRADE_DOCUMENTS_SYSTEM,
    GRADE_DOCUMENTS_USER,
    REWRITE_QUERY_SYSTEM,
    REWRITE_QUERY_USER,
    GENERATE_ANSWER_SYSTEM,
    GENERATE_ANSWER_USER,
    CHECK_HALLUCINATION_SYSTEM,
    CHECK_HALLUCINATION_USER,
)
from src.agent.tools import ALL_TOOLS, _duckduckgo_search
from src.backend.llm import create_llm_client, create_fast_llm, create_strong_llm
from src.store.vector_store import vector_store
from src.memory.manager import memory_manager
from src.config.settings import settings

logger = logging.getLogger(__name__)

# 创建 ToolNode（LangGraph 预置，自动处理 tool_calls）
tool_node = ToolNode(ALL_TOOLS)


def retrieve(state: AgentState) -> dict[str, Any]:
    """检索节点：语义检索 + MMR 多样性检索

    使用混合策略：
    1. 先进行语义相似度检索
    2. 再用 MMR 做多样性补充
    3. 合并去重
    """
    query = state.get("rewritten_query") or state["query"]
    logger.info("检索节点: query='%s'", query)

    # 语义检索
    semantic_results = vector_store.search(query, top_k=settings.retrieval_top_k)
    semantic_docs = [doc for doc, _ in semantic_results]

    # MMR 多样性检索（从更多候选中挑选多样化的结果）
    try:
        mmr_docs = vector_store.search_mmr(query, top_k=settings.retrieval_top_k)
    except Exception:
        mmr_docs = []

    # 合并去重（按内容去重）
    seen = set()
    merged: list[Document] = []
    for doc in semantic_docs + mmr_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            merged.append(doc)

    logger.info("检索完成: 语义=%d, MMR=%d, 合并=%d", len(semantic_docs), len(mmr_docs), len(merged))

    return {
        "documents": merged,
        "agent_path": ["retrieve"],
    }


def grade_documents(state: AgentState) -> dict[str, Any]:
    """文档评估节点：判断检索结果是否与问题相关"""
    documents = state.get("documents", [])
    query = state["query"]

    if not documents:
        logger.info("评估节点: 无检索结果")
        return {
            "documents_relevant": False,
            "agent_path": ["grade_documents (no results)"],
        }

    logger.info("评估节点: 评估 %d 个文档", len(documents))

    # 用快速模型评估（节省成本）
    llm = create_fast_llm()

    # 格式化文档内容
    docs_text = "\n---\n".join(
        f"[文档 {i+1}] {doc.page_content[:500]}"
        for i, doc in enumerate(documents[:10])
    )

    messages = [
        HumanMessage(content=GRADE_DOCUMENTS_SYSTEM),
        HumanMessage(
            content=GRADE_DOCUMENTS_USER.format(query=query, documents=docs_text)
        ),
    ]

    response = llm.invoke(messages)
    result = response.content.strip().upper()
    relevant = result == "RELEVANT"

    logger.info("评估结果: %s (raw='%s')", "RELEVANT" if relevant else "IRRELEVANT", result)

    return {
        "documents_relevant": relevant,
        "agent_path": ["grade_documents"],
    }


def transform_query(state: AgentState) -> dict[str, Any]:
    """查询重写节点：优化用户查询以获得更好的检索效果"""
    query = state["query"]
    iteration_count = state.get("iteration_count", 0)

    logger.info("查询重写节点: 第 %d 次重写, 原查询='%s'", iteration_count + 1, query)

    llm = create_llm_client()
    messages = [
        HumanMessage(content=REWRITE_QUERY_SYSTEM),
        HumanMessage(content=REWRITE_QUERY_USER.format(query=query)),
    ]
    response = llm.invoke(messages)
    rewritten = response.content.strip()

    logger.info("重写结果: '%s'", rewritten)

    return {
        "rewritten_query": rewritten,
        "iteration_count": iteration_count + 1,
        "agent_path": ["transform_query"],
    }


def generate(state: AgentState) -> dict[str, Any]:
    """答案生成节点：基于检索到的文档生成回答"""
    query = state["query"]
    documents = state.get("documents", [])
    session_id = state.get("session_id", "default")

    if not documents:
        return {
            "answer": "未找到相关文档，无法生成回答。请尝试上传相关文档或重新提问。",
            "agent_path": ["generate (no context)"],
        }

    logger.info("生成节点: 基于 %d 个文档生成回答", len(documents))

    # 使用强模型生成高质量答案
    llm = create_strong_llm(streaming=state.get("stream", False))

    # 获取对话历史
    chat_history = memory_manager.get_chat_history_string(session_id)

    # 格式化文档上下文
    docs_text = "\n\n---\n\n".join(
        f"来源: {doc.metadata.get('filename', doc.metadata.get('url', 'unknown'))}\n"
        f"链接: {doc.metadata.get('url', '无')}\n"
        f"内容: {doc.page_content}"
        for doc in documents[:8]
    )

    messages = [
        HumanMessage(content=GENERATE_ANSWER_SYSTEM),
        HumanMessage(
            content=GENERATE_ANSWER_USER.format(
                query=query,
                documents=docs_text,
                chat_history=chat_history or "无历史对话",
            )
        ),
    ]

    if state.get("stream", False):
        # 流式模式：返回消息用于外部流式处理
        return {
            "messages": messages,
            "agent_path": ["generate (streaming)"],
        }

    response = llm.invoke(messages)
    answer = response.content.strip()

    logger.info("生成完成: %d 字符", len(answer))

    return {
        "answer": answer,
        "agent_path": ["generate"],
    }


def check_hallucination(state: AgentState) -> dict[str, Any]:
    """幻觉检测节点：验证生成的答案是否与文档上下文一致"""
    answer = state.get("answer", "")
    documents = state.get("documents", [])

    if not answer or not documents:
        return {
            "hallucination_detected": False,
            "agent_path": ["check_hallucination (skipped)"],
        }

    logger.info("幻觉检测节点: 验证答案")

    llm = create_fast_llm()
    docs_text = "\n---\n".join(
        f"[文档 {i+1}] {doc.page_content[:500]}"
        for i, doc in enumerate(documents[:8])
    )

    messages = [
        HumanMessage(content=CHECK_HALLUCINATION_SYSTEM),
        HumanMessage(
            content=CHECK_HALLUCINATION_USER.format(
                documents=docs_text,
                answer=answer,
            )
        ),
    ]

    response = llm.invoke(messages)
    result = response.content.strip().upper()
    has_hallucination = result == "FAILED"

    logger.info("幻觉检测: %s", "FAILED" if has_hallucination else "PASSED")

    return {
        "hallucination_detected": has_hallucination,
        "agent_path": ["check_hallucination"],
    }


def web_search_node(state: AgentState) -> dict[str, Any]:
    """联网搜索节点：向量库无相关文档时，调用 DuckDuckGo 搜索作为降级方案

    搜索结果转为 Document 列表，metadata 中包含 source='web'、url、title 等字段，
    供 generate 节点和前端 SourcePanel 使用。
    """
    query = state["query"]
    logger.info("联网搜索节点: query='%s'", query)

    results = _duckduckgo_search(query, max_results=5)
    if not results:
        logger.info("联网搜索无结果")
        return {
            "documents": [],
            "documents_relevant": False,
            "agent_path": ["web_search (no results)"],
        }

    # 构造 Document 对象，metadata 包含 URL 信息供前端展示
    documents: list[Document] = []
    for r in results:
        doc = Document(
            page_content=f"[{r['title']}] {r['snippet']}",
            metadata={
                "source": "web",
                "url": r["url"],
                "title": r["title"],
                "filename": f"网页: {r['title'][:30]}",
            },
        )
        documents.append(doc)

    logger.info("联网搜索完成: %d 条结果", len(documents))

    return {
        "documents": documents,
        "documents_relevant": True,  # 标记相关，跳过 transform_query 循环
        "agent_path": ["web_search"],
    }


def decide_retrieval_strategy(state: AgentState) -> dict[str, Any]:
    """策略决策节点：根据用户问题决定使用哪种检索策略"""
    query = state["query"]
    documents = state.get("documents", [])

    if not documents:
        return {"agent_path": ["decide_strategy (no documents)"]}

    return {"agent_path": ["decide_strategy"]}
