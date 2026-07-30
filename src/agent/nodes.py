"""LangGraph Agent 节点实现 - 多策略检索节点

生成（generate）和幻觉检测（check_hallucination）已移至独立模块：
- src/services/generator.py
- src/services/hallucination_checker.py

此文件仅保留图内执行的检索、排序、评估、重写等节点。
"""

import logging
import jieba
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langgraph.prebuilt import ToolNode

from src.agent.state import AgentState
from src.agent.prompts import (
    GRADE_DOCUMENTS_SYSTEM,
    GRADE_DOCUMENTS_USER,
    REWRITE_QUERY_SYSTEM,
    REWRITE_QUERY_USER,
    MULTI_QUERY_GENERATE_USER,
    JUDGE_COMPLEXITY_SYSTEM,
    JUDGE_COMPLEXITY_USER,
)
from src.agent.tools import ALL_TOOLS, _duckduckgo_search
from src.backend.llm import create_llm_client, create_fast_llm, create_strong_llm
from src.backend.embedding import get_embedding_client
from src.backend.reranker import rerank_documents
from src.store.vector_store import vector_store
from src.retrieval.bm25 import bm25_retriever
from src.config.settings import settings
from src.knowledge_graph import get_kg_intent_analyzer, get_graph_retriever, get_graph_store
from src.services.generator import format_documents_with_citations, build_generate_prompt
from src.memory.manager import memory_manager

from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# 创建 ToolNode（LangGraph 预置，自动处理 tool_calls）
tool_node = ToolNode(ALL_TOOLS)

# 线程池单例（P2: 避免每次请求重复创建销毁）
_EXECUTOR = ThreadPoolExecutor(max_workers=4)


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

    优化策略：
    1. 关键词预过滤：提取 query 中的名词关键词，快速检测文档中是否包含
    2. LLM 精确评估：仅当关键词匹配失败时调用 qwen-turbo
    3. 精简 token：最多 5 篇文档 × 每篇 300 字符

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

    # ── 关键词预过滤（jieba 分词 + Set 交集，跳过 LLM） ──
    # 1) 停用词过滤：去除跨领域高频词，保留实质性关键词
    # 2) 自适应阈值：短 query 降低门槛，避免锁死在 LLM 路径
    query_words = {
        w.lower() for w in jieba.lcut(query)
        if len(w) >= 2 and w.lower() not in _GRADE_STOP_WORDS
    }

    if query_words:
        # 自适应阈值：query 词数越多，要求匹配越多
        #   query 词数 1-2 → 阈值 1 (有任一实质性词匹配即判定相关)
        #   query 词数 3   → 阈值 2
        #   query 词数 4+  → 阈值 3
        qw_count = len(query_words)
        threshold = 3 if qw_count >= 4 else (2 if qw_count >= 3 else 1)

        for doc in documents:
            doc_words = {
                w.lower() for w in jieba.lcut(doc.page_content)
                if len(w) >= 2 and w.lower() not in _GRADE_STOP_WORDS
            }
            overlap = query_words & doc_words
            if len(overlap) >= threshold:
                logger.info(
                    "评估节点: jieba 预过滤 → RELEVANT (overlap=%d/%d, thresh=%d, words=%s)",
                    len(overlap), qw_count, threshold, overlap,
                )
                return {
                    "documents_relevant": True,
                    "agent_path": ["grade_documents (keyword match)"],
                }

    # ── LLM 精确评估（关键词匹配不足时降级到 LLM） ──
    llm = create_fast_llm()

    # 精简文档上下文：最多 3 篇 × 200 字符（仅需判断相关性，无需全文）
    max_docs = min(len(documents), 3)
    docs_text = "\n".join(
        f"[{i+1}] {doc.page_content[:200]}"
        for i, doc in enumerate(documents[:max_docs])
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


# ── 评估节点停用词表（跨领域高频词，无语义区分度） ──
_GRADE_STOP_WORDS: frozenset[str] = frozenset({
    "影响", "使用", "情况", "方面", "进行", "通过", "可以", "需要",
    "相关", "不同", "包括", "以及", "用于", "基于", "作为", "其中",
    "应用", "提供", "实现", "产生", "具有", "主要", "比较", "一定",
    "很大", "可能", "利用", "方式", "方法", "问题", "特点", "特征",
    "优势", "不足", "发展", "研究", "内容", "信息", "数据", "系统",
})

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


def parallel_retrieve_merge_node(state: AgentState) -> dict[str, Any]:
    """并行检索合并节点（实时流推送子策略状态）

    在一个节点内完成所有策略的并行执行和合并。
    使用 get_stream_writer() 将子策略的开始/结束事件实时推送到前端流程图。

    1. 预计算 query embedding（一次 API 调用）
    2. 语义检索 + MMR（复用同一 embedding，零额外 API 开销）
    3. BM25 / Multi-Query / KG_LLM（线程池并行）
    4. KG Kuzu 查询（主线程）
    5. 合并去重（仅执行一次）
    """
    from langgraph.config import get_stream_writer
    writer = get_stream_writer()

    def _push(event: str, node: str) -> None:
        """推送自定义事件到前端（实时）"""
        writer({"event": event, "node": node})
        logger.debug("[PARALLEL_MERGE] 实时推送: %s → %s", node, event)

    query = state.get("rewritten_query") or state["query"]
    logger.info("[PARALLEL_MERGE] 入口: query='%s'", query[:80])

    import time
    t_start = time.perf_counter()
    enable_bm25 = state.get("enable_bm25", True)
    enable_mq = state.get("enable_multi_query", False)
    enable_kg = state.get("enable_kg", False) and state.get("kg_intent", False)

    # 各策略独立耗时（毫秒）
    strategy_timings_ms: dict[str, float] = {}

    # ── 1. 预计算 query embedding（仅一次 API 调用） ──
    query_embedding = get_embedding_client().embed_query(query)

    # 语义检索（复用已计算的 embedding）
    semantic_results = vector_store.search_by_embedding(
        query_embedding, top_k=settings.retrieval_top_k,
    )
    semantic_docs = [doc for doc, _ in semantic_results]

    # MMR 多样性检索（复用同一 embedding，零额外 API 开销）
    try:
        mmr_docs = vector_store.search_mmr_by_embedding(
            query_embedding, top_k=settings.retrieval_top_k,
        )
    except Exception:
        mmr_docs = []

    # 合并去重
    base = _merge_documents(semantic_docs, mmr_docs)
    elapsed_sem = round(time.perf_counter() - t_start, 3)
    strategy_timings_ms["retrieve"] = round(elapsed_sem * 1000, 1)
    _push("node_step", "retrieve")
    logger.info("[PARALLEL_MERGE] 语义+MMR: %d docs (%.3fs)", len(base), elapsed_sem)

    # ── 2. 线程池并行（BM25 / Multi-Query / KG_LLM） ──
    parallel_docs: dict[str, Any] = {}
    futures: dict[str, Any] = {}

    if enable_bm25:
        futures["bm25"] = _EXECUTOR.submit(
            bm25_retriever.search, query, settings.retrieval_top_k,
        )
    if enable_mq:
        futures["multi"] = _EXECUTOR.submit(_run_multi_query, query)

    for name, fut in futures.items():
        t_fut_start = time.perf_counter()
        try:
            parallel_docs[name] = fut.result(timeout=8)
        except TimeoutError:
            logger.warning("[PARALLEL_MERGE] %s 超时 (>8s), 跳过", name)
            parallel_docs[name] = [] if name != "kg_extract" else None
        except Exception as e:
            logger.warning("[PARALLEL_MERGE] %s 线程异常: %s", name, e)
            parallel_docs[name] = [] if name != "kg_extract" else None
        # 线程完成 → 推送 node_step + 记录该策略耗时
        step_name = {
            "bm25": "bm25_retrieve",
            "multi": "multi_query_retrieve",
        }.get(name, name)
        strategy_timings_ms[step_name] = round((time.perf_counter() - t_fut_start) * 1000, 1)
        _push("node_step", step_name)

    elapsed_parallel = round(time.perf_counter() - t_start, 3)
    logger.info(
        "[PARALLEL_MERGE] 并行策略: bm25=%d, multi=%d, kg_extract=%s (%.3fs)",
        len(parallel_docs.get("bm25", [])),
        len(parallel_docs.get("multi", [])),
        "done" if parallel_docs.get("kg_extract") else "skip",
        elapsed_parallel - elapsed_sem,
    )

    # 所有策略子任务已完成，现在才激活检索合并节点（让父节点在子任务后点亮）
    _push("node_start", "parallel_retrieve_merge")

    # ── 3. KG Kuzu 查询（主线程，非线程安全，~10ms） ──
    # 直接通过 Kuzu 模糊搜索匹配实体（替代 LLM 抽取，从 8s -> 10ms）
    t_kg = time.perf_counter()
    kg_context = ""
    if enable_kg:
        try:
            store = get_graph_store()
            retriever = get_graph_retriever()
            raw_matches = store.search_entities(query, top_k=settings.kg_max_entities)
            seed_entities = [name for name, score in raw_matches if score >= 0.6]
            if seed_entities:
                subgraph = store.get_subgraph(seed_entities, hops=settings.kg_max_hops)
                if subgraph.number_of_nodes() > 0:
                    paths_text = retriever._find_entity_paths(seed_entities, store)
                    context = retriever._subgraph_to_text(subgraph, seed_entities)
                    if paths_text:
                        context += f"\n\n关联路径:\n{paths_text}"
                    kg_context = context
                    logger.info("[PARALLEL_MERGE] KG: %d 实体, %d 字符",
                                len(seed_entities), len(kg_context))
                else:
                    logger.info("[PARALLEL_MERGE] KG: 子图为空")
            else:
                logger.info("[PARALLEL_MERGE] KG: 无匹配实体")
        except Exception as e:
            logger.warning("[PARALLEL_MERGE] KG 异常: %s", e)

    if enable_kg:
        strategy_timings_ms["kg_retrieve"] = round((time.perf_counter() - t_kg) * 1000, 1)
        _push("node_step", "kg_retrieve")

    elapsed_total = round(time.perf_counter() - t_start, 3)

    # ── 4. 合并（仅执行一次） ──
    merged = _merge_documents(
        base,
        parallel_docs.get("bm25", []),
        parallel_docs.get("multi", []),
    )

    if kg_context:
        kg_doc = Document(
            page_content=kg_context,
            metadata={"source": "knowledge_graph", "filename": "知识图谱"},
        )
        merged.insert(0, kg_doc)
        logger.info("[PARALLEL_MERGE] KG 上下文已前置插入")

    logger.info(
        "[PARALLEL_MERGE] 结果: base=%d, bm25=%d, multi=%d, "
        "kg=%s → merged=%d (%.3fs)",
        len(base),
        len(parallel_docs.get("bm25", [])),
        len(parallel_docs.get("multi", [])),
        "yes" if kg_context else "no",
        len(merged),
        elapsed_total,
    )

    return {
        "documents": merged,
        "agent_path": ["parallel_retrieve_merge"],
        "kg_context": kg_context,
        "strategy_timings_ms": strategy_timings_ms,
        # 保留各策略独立结果（供前端节点详情分别展示）
        "documents_semantic": base,
        "documents_bm25": parallel_docs.get("bm25", []),
        "documents_multi_query": parallel_docs.get("multi", []),
    }


def _run_multi_query(query: str) -> list[Document]:
    """Thread-safe Multi-Query 内部实现"""
    from src.backend.llm import create_fast_llm
    fast_llm = create_fast_llm()
    num_vars = settings.multi_query_num_variations
    response = fast_llm.invoke(
        MULTI_QUERY_GENERATE_USER.format(query=query, num_variations=num_vars),
    )
    raw = response.content.strip()
    variants = [line.strip() for line in raw.split("\n") if line.strip()]
    variants = [
        v.split(". ", 1)[-1] if ". " in v[:5] else v for v in variants
    ]
    variants = [v for v in variants if v != query][:num_vars]

    all_results: list[Document] = []
    for v in [query] + variants:
        results = vector_store.search(v, top_k=settings.retrieval_top_k)
        all_results.extend(doc for doc, _ in results)

    seen: set[str] = set()
    merged: list[Document] = []
    for doc in all_results:
        key = doc.page_content[:200]
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    return merged


# ==================== 知识图谱节点 ====================


def analyze_kg_intent_node(state: AgentState) -> dict[str, Any]:
    """知识图谱意图分析节点（双阶段分级判断）

    在检索之前执行，分析用户问题是否适合用知识图谱回答。
    只有当 enable_kg=True 且图谱非空时才会被调用。

    Stage 1: Kuzu 模糊搜索快速匹配（~10ms）
        - 如果 query 中包含图库中存在的实体名 → 直接判定需要 KG
        - 适用于明确提到实体名称的场景（~80% case）
    Stage 2: LLM 语义分析兜底（~1s）
        - 处理指代、隐式引用等边界 case
        - 复用原有的 KGIntentAnalyzer

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

    # ── Stage 1: Kuzu 快速实体匹配（10ms） ──
    try:
        results = store.search_entities(query, top_k=3)
        # 实体名在 query 中明确出现 (score >= 0.6) → 确定提到了图谱中的实体
        # 0.6 对应 name_lower in query_lower（实体名是 query 子串），
        # 0.7 对应 query_lower in name_lower，0.9+ 对应别名/精确匹配
        has_exact_match = any(score >= 0.6 for _, score in results)
        if has_exact_match:
            logger.info(
                "KG 意图分析: Stage 1 快速匹配成功 → kg_intent=True (matches=%s)",
                [(name, f"{s:.2f}") for name, s in results],
            )
            return {
                "kg_intent": True,
                "agent_path": ["analyze_kg_intent"],
            }
        logger.debug(
            "KG 意图分析: Stage 1 无精确匹配 (best=%.2f)，进入 Stage 2 LLM 分析",
            max((s for _, s in results), default=0),
        )
    except Exception as e:
        logger.warning("KG 意图分析: Stage 1 异常，降级到 Stage 2: %s", e)

    # ── Stage 2: LLM 语义分析兜底（1s） ──
    try:
        analyzer = get_kg_intent_analyzer()
        should_use = analyzer.analyze(query)
        logger.info(
            "KG 意图分析: Stage 2 LLM 分析 → %s",
            "kg_intent=True" if should_use else "kg_intent=False",
        )
        return {
            "kg_intent": should_use,
            "agent_path": ["analyze_kg_intent"],
        }
    except Exception as e:
        logger.warning("KG 意图分析: Stage 2 异常，默认降级: %s", e)
        return {"kg_intent": False, "agent_path": ["analyze_kg_intent (error)"]}


# ==================== 复杂度判定节点 ====================


def _quick_complexity(query: str) -> str | None:
    """基于规则的快速复杂度判定（0ms，无需 LLM）

    Returns:
        "SIMPLE" / "COMPLEX" / None（不确定，需 LLM 兜底）
    """
    q = query.strip()
    q_lower = q.lower()

    # ── 匹配检查（同时检测两种信号） ──
    complex_signals = [
        "为什么", "怎么", "如何", "怎样",
        "对比", "比较", "区别", "差异", "vs",
        "原因", "后果", "影响", "导致",
        "是否应该", "应不应该", "是否可以",
        "分析", "论证", "评价", "讨论",
        "关系", "联系", "关联",
        "利弊", "优劣", "优缺点",
        "if", "whether", "should", "compare", "difference",
    ]
    simple_signals = [
        "什么是", "是什么", "指的是",
        "谁", "哪一年", "什么时候",
        "在哪里", "多少",
        "定义", "概念", "解释",
        "列举", "列出", "有哪些",
        "的别名", "的简称", "的全称",
        "what is", "who is", "define", "list", "what are",
    ]

    has_complex = any(sig in q_lower for sig in complex_signals)
    has_simple = any(sig in q_lower for sig in simple_signals)

    # 同时匹配两种信号 → 复杂优先
    if has_complex and has_simple:
        return "COMPLEX"
    if has_complex:
        return "COMPLEX"
    if has_simple:
        return "SIMPLE"

    # 无信号词 → 不确定，走 LLM 兜底
    return None


def judge_complexity_node(state: AgentState) -> dict[str, Any]:
    """复杂度判定节点：先走规则快速路径，不确定时降级 qwen-turbo

    规则路径：0ms，覆盖 ~60% 的常见 query 模式
    LLM 路径：~200ms（去掉文档预览，仅 query+模型名）
    """
    from langgraph.config import get_stream_writer
    writer = get_stream_writer()
    writer({"event": "node_start", "node": "judge_complexity"})

    query = state["query"]

    # ── 规则快速路径（0ms） ──
    complexity = _quick_complexity(query)
    if complexity:
        logger.info("复杂度判定: 规则路径 → %s (query='%s')", complexity, query[:50])
        return {
            "complexity": complexity,
            "agent_path": ["judge_complexity (rule)"],
        }

    # ── LLM 兜底（去掉文档预览，复杂度只看 query 本身） ──
    llm = create_fast_llm()
    messages = [
        HumanMessage(content=JUDGE_COMPLEXITY_SYSTEM),
        HumanMessage(content=JUDGE_COMPLEXITY_USER.format(
            query=query,
            doc_count=len(state.get("documents", [])),
            doc_previews="",
        )),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip().upper()
    complexity = "COMPLEX" if "COMPLEX" in raw else "SIMPLE"

    logger.info("复杂度判定: LLM 路径 → %s (raw='%s')", complexity, raw)

    return {
        "complexity": complexity,
        "agent_path": ["judge_complexity"],
    }


# ==================== 路由函数（判定后） ====================


def route_after_judge(state: AgentState) -> Literal["generate_simple", "generate_complex"]:
    """复杂度判定后的条件路由

    - SIMPLE → generate_simple（qwen-turbo 快速生成）
    - COMPLEX → generate_complex（qwen-max 高质量生成）
    """
    complexity = state.get("complexity", "SIMPLE")
    if complexity == "COMPLEX":
        logger.info("路由: 复杂度 COMPLEX → generate_complex")
        return "generate_complex"
    logger.info("路由: 复杂度 SIMPLE → generate_simple")
    return "generate_simple"


# ==================== 生成节点（简单 / 复杂） ====================


def generate_simple_node(state: AgentState) -> dict[str, Any]:
    """简单生成节点：使用 qwen-turbo 流式生成"""
    return _generate_node(state, is_simple=True)


def generate_complex_node(state: AgentState) -> dict[str, Any]:
    """复杂生成节点：使用 qwen-max 流式生成"""
    return _generate_node(state, is_simple=False)


def _generate_node(state: AgentState, is_simple: bool) -> dict[str, Any]:
    """内部生成逻辑（简单/复杂共用）

    通过 get_stream_writer 实时推送 token 和 citation 事件，
    rag_service 在 astream custom 循环中转发为 SSE。
    """
    from langgraph.config import get_stream_writer
    writer = get_stream_writer()

    node_id = "generate_simple" if is_simple else "generate_complex"
    # 立即推送 node_start，让前端流程图实时点亮生成节点
    writer({"event": "node_start", "node": node_id})

    query = state["query"]
    documents = state.get("documents", [])
    session_id = state.get("session_id", "")

    # 1. 带引文的文档格式化
    docs_text, citation_metadata = format_documents_with_citations(documents)

    # 2. 推送引文元数据（仅需一次）
    if citation_metadata:
        writer({
            "event": "citations",
            "data": citation_metadata,
        })

    # 3. 获取对话历史
    chat_history = ""
    if session_id:
        try:
            chat_history = memory_manager.get_chat_history_string(session_id)
        except Exception:
            chat_history = ""

    # 4. 构建 prompt
    prompt = build_generate_prompt(query, docs_text, chat_history)

    # 5. 创建 LLM（qwen-turbo 或 qwen-max）
    if is_simple:
        llm = create_fast_llm(streaming=True)
    else:
        llm = create_strong_llm(streaming=True)

    # 6. 流式生成 & 实时推送 token
    full_answer = ""
    for chunk in llm.stream(prompt):
        if chunk.content:
            full_answer += chunk.content
            writer({
                "event": "token",
                "content": chunk.content,
            })

    logger.info(
        "生成完成: model=%s, answer_len=%d, citations=%d",
        "qwen-turbo" if is_simple else "qwen-max",
        len(full_answer),
        len(citation_metadata),
    )

    return {
        "answer": full_answer,
        "citation_metadata": citation_metadata,
        "agent_path": ["generate_simple" if is_simple else "generate_complex"],
    }
