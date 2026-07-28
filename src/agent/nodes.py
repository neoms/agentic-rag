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
    HYDE_GENERATE_USER,
    MULTI_QUERY_GENERATE_USER,
)
from src.agent.tools import ALL_TOOLS, _duckduckgo_search
from src.backend.llm import create_llm_client, create_fast_llm, create_strong_llm
from src.backend.embedding import get_embedding_client
from src.backend.reranker import rerank_documents
from src.store.vector_store import vector_store
from src.retrieval.bm25 import bm25_retriever
from src.memory.manager import memory_manager
from src.config.settings import settings
from src.knowledge_graph import get_kg_intent_analyzer, get_graph_retriever, get_graph_store

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


def rerank_documents_node(state: AgentState) -> dict[str, Any]:
    """重排序节点：对检索结果做二次精排

    使用百炼 TextReRank 模型对文档列表按 query 相关性重新排序，
    保留 top_k 个最相关的文档，提升后续生成质量。
    当全局 rerank_enabled=False 或数据不足时直接透传。
    （开关控制已提升到 graph 层 route_after_merge 条件边，此处不再处理）
    """
    documents = state.get("documents", [])
    query = state["query"]

    if not documents:
        logger.info("重排序节点: 无文档，跳过")
        return {"agent_path": ["rerank (no docs)"]}

    if not settings.rerank_enabled:
        logger.info("重排序节点: 全局禁用，透传 %d 个文档", len(documents))
        return {"agent_path": ["rerank (disabled)"]}

    reranked = rerank_documents(query, documents, top_k=settings.rerank_top_k)

    return {
        "documents": reranked,
        "agent_path": ["rerank"],
    }


def grade_documents(state: AgentState) -> dict[str, Any]:
    """文档评估节点：判断检索结果是否与问题相关

    开关控制已提升到 graph 层 route_after_merge / route_after_rerank 条件边，
    此处只处理业务逻辑。
    """
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
    stream_mode = state.get("stream", False)

    if not answer or not documents:
        # 流式模式下答案由外部生成，此处仅为路径标记（实际检测在 rag_service 中）
        if stream_mode and documents:
            logger.info("幻觉检测节点: 流式模式，标记路径")
            return {
                "hallucination_detected": False,
                "agent_path": ["check_hallucination"],
            }
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


# ==================== 新增检索策略节点 ====================


def _merge_documents(
    *doc_lists: list[Document], key_len: int = 200,
) -> list[Document]:
    """合并多个文档列表并去重，以 page_content[:key_len] 为 key"""
    seen: set[str] = set()
    merged: list[Document] = []
    for docs in doc_lists:
        for doc in docs:
            key = doc.page_content[:key_len]
            if key not in seen:
                seen.add(key)
                merged.append(doc)
    return merged


def bm25_retrieve_node(state: AgentState) -> dict[str, Any]:
    """BM25 关键词检索节点

    通过条件边 Send 调度，只有 enable_bm25=True 时才会被调用。
    结果写入独立 State key documents_bm25 避免并行冲突。
    """
    query = state.get("rewritten_query") or state["query"]
    logger.info("BM25 检索节点: query='%s'", query)

    try:
        results = bm25_retriever.search(query, top_k=settings.retrieval_top_k)
        logger.info("BM25 检索完成: %d 个结果", len(results))
        return {
            "documents_bm25": results,
            "agent_path": ["bm25_retrieve"],
        }
    except Exception as e:
        logger.warning("BM25 检索异常: %s", e)
        return {
            "documents_bm25": [],
            "agent_path": ["bm25_retrieve (error)"],
        }


def hyde_retrieve_node(state: AgentState) -> dict[str, Any]:
    """HyDE 假设文档嵌入检索节点

    1. LLM 生成假设答案
    2. Embedding 向量化假设答案
    3. 用假设答案的向量做语义检索
    结果写入独立 State key documents_hyde。
    """
    query = state.get("rewritten_query") or state["query"]
    logger.info("HyDE 检索节点: query='%s'", query)

    try:
        # 1. 生成假设答案
        fast_llm = create_fast_llm()
        hyde_prompt = HYDE_GENERATE_USER.format(query=query)
        response = fast_llm.invoke(hyde_prompt)
        hypothetical_doc = response.content.strip()
        logger.info("HyDE: 假设答案生成完成，%d 字", len(hypothetical_doc))

        # 2. 向量化假设答案
        embedder = get_embedding_client()
        hyde_embedding = embedder.embed_query(hypothetical_doc)

        # 3. 用假设答案的向量做语义检索
        k = settings.retrieval_top_k
        results = vector_store.vector_store.similarity_search_by_vector(
            hyde_embedding, k=k,
        )

        logger.info("HyDE 检索完成: %d 个结果", len(results))
        return {
            "documents_hyde": list(results),
            "agent_path": ["hyde_retrieve"],
        }
    except Exception as e:
        logger.warning("HyDE 检索异常: %s", e)
        return {
            "documents_hyde": [],
            "agent_path": ["hyde_retrieve (error)"],
        }


def multi_query_retrieve_node(state: AgentState) -> dict[str, Any]:
    """Multi-Query 多角度查询检索节点

    1. LLM 生成 N 个查询变体
    2. 对每个变体做语义检索
    3. 全域去重合并
    结果写入独立 State key documents_multi_query。
    """
    query = state.get("rewritten_query") or state["query"]
    logger.info("Multi-Query 检索节点: query='%s'", query)

    try:
        # 1. 生成查询变体
        fast_llm = create_fast_llm()
        num_vars = settings.multi_query_num_variations
        mq_prompt = MULTI_QUERY_GENERATE_USER.format(
            query=query, num_variations=num_vars,
        )
        response = fast_llm.invoke(mq_prompt)
        raw = response.content.strip()
        # 解析 LLM 输出的多行查询变体
        variants = [line.strip() for line in raw.split("\n") if line.strip()]
        # 去掉可能的编号前缀（如 "1. ", "- "）
        variants = [
            v.split(". ", 1)[-1] if ". " in v[:5] else v
            for v in variants
        ]
        variants = [v for v in variants if v != query][:num_vars]
        logger.info("Multi-Query: 生成 %d 个变体: %s", len(variants), variants)

        # 2. 对每个变体做语义检索，合并去重
        all_results: list[Document] = []
        for v in [query] + variants:
            results = vector_store.search(v, top_k=settings.retrieval_top_k)
            all_results.extend(doc for doc, _ in results)

        # 3. 全域去重
        merged = _merge_documents(*all_results)
        logger.info("Multi-Query 检索完成: 去重后 %d 个结果", len(merged))

        return {
            "documents_multi_query": merged,
            "agent_path": ["multi_query_retrieve"],
        }
    except Exception as e:
        logger.warning("Multi-Query 检索异常: %s", e)
        return {
            "documents_multi_query": [],
            "agent_path": ["multi_query_retrieve (error)"],
        }


def merge_retrieval_node(state: AgentState) -> dict[str, Any]:
    """检索结果合并节点

    从多个来源合并文档并去重：
    1. documents（基础向量+MMR 检索）
    2. documents_bm25（BM25 检索，仅当 enable_bm25=True 时有值）
    3. documents_hyde（HyDE 检索，仅当 enable_hyde=True 时有值）
    4. documents_multi_query（Multi-Query 检索，仅当 enable_multi_query=True 时有值）
    5. kg_context（知识图谱检索结果，作为特殊 Document 附加）
    """
    base = state.get("documents", [])
    bm25 = state.get("documents_bm25", [])
    hyde = state.get("documents_hyde", [])
    multi = state.get("documents_multi_query", [])
    kg_context = state.get("kg_context", "")

    merged = _merge_documents(base, bm25, hyde, multi)

    # 将 KG 上下文作为特殊 Document 附加（标记来源）
    if kg_context:
        from langchain_core.documents import Document
        kg_doc = Document(
            page_content=kg_context,
            metadata={
                "source": "knowledge_graph",
                "filename": "知识图谱",
            },
        )
        merged.insert(0, kg_doc)  # 放最前面，让 LLM 优先参考
        logger.info("KG 上下文已附加到文档列表 (%d 字符)", len(kg_context))

    logger.info(
        "合并检索结果: base=%d, bm25=%d, hyde=%d, multi=%d, kg=%s → merged=%d",
        len(base), len(bm25), len(hyde), len(multi),
        "yes" if kg_context else "no", len(merged),
    )

    return {
        "documents": merged,
        "agent_path": ["merge_retrieval"],
    }


# ==================== 知识图谱检索节点 ====================


def analyze_kg_intent_node(state: AgentState) -> dict[str, Any]:
    """知识图谱意图分析节点

    在检索之前执行，分析用户问题是否适合用知识图谱回答。
    只有当 enable_kg=True 且图谱非空时才会被调用。

    设置 kg_intent 标志：
        - True → 后续会触发 kg_retrieve 并行检索
        - False → 降级，只走原有检索流程
    """
    enable_kg = state.get("enable_kg", False)
    query = state["query"]

    if not enable_kg:
        logger.info("KG 意图分析: 已禁用, kg_intent=False")
        return {"kg_intent": False, "agent_path": ["analyze_kg_intent (disabled)"]}

    store = get_graph_store()
    if store.is_empty():
        logger.info("KG 意图分析: 图谱为空, kg_intent=False")
        return {"kg_intent": False, "agent_path": ["analyze_kg_intent (empty kg)"]}

    try:
        analyzer = get_kg_intent_analyzer()
        should_use = analyzer.analyze(query)
        logger.info("KG 意图分析结果: %s", "SHOULD_USE_KG" if should_use else "SHOULD_NOT_USE_KG")
        return {
            "kg_intent": should_use,
            "agent_path": ["analyze_kg_intent"],
        }
    except Exception as e:
        logger.warning("KG 意图分析异常，默认降级: %s", e)
        return {"kg_intent": False, "agent_path": ["analyze_kg_intent (error)"]}


def kg_retrieve_node(state: AgentState) -> dict[str, Any]:
    """知识图谱检索节点

    通过条件边 Send 调度，只有 enable_kg=True AND kg_intent=True 时才会被调用。

    流程:
        1. 从 query 抽取实体
        2. Entity Linking 定位种子节点
        3. BFS 子图提取
        4. 多跳路径推理
        5. 生成结构化上下文文本 → 写入 kg_context

    kg_context 在 merge_retrieval 中作为特殊 Document 附加到 documents 列表。
    """
    query = state.get("rewritten_query") or state["query"]
    logger.info("KG 检索节点: query='%s'", query[:100])

    try:
        store = get_graph_store()
        retriever = get_graph_retriever()
        context, entities = retriever.search(query, store)

        if not context:
            logger.info("KG 检索无结果")
            return {
                "kg_context": "",
                "kg_intent": False,  # 避免后续循环重复 Send
                "agent_path": ["kg_retrieve (no results)"],
            }

        logger.info("KG 检索完成: %d 实体, %d 字符",
                     len(entities), len(context))

        return {
            "kg_context": context,
            "kg_intent": False,  # 已完成 KG 检索，后续循环不再重复触发
            "agent_path": ["kg_retrieve"],
        }

    except Exception as e:
        logger.warning("KG 检索异常: %s", e)
        return {
            "kg_context": "",
            "kg_intent": False,  # 避免后续循环重复 Send
            "agent_path": ["kg_retrieve (error)"],
        }
