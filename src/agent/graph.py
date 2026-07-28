"""LangGraph 状态图构建 - Agent 节点编排与条件路由

状态流转图：
    START → retrieve → rerank_documents → grade_documents
                                            ├── [RELEVANT] → generate → check_hallucination
                                            │                              ├── [PASSED] → END
                                            │                              └── [FAILED] → END
                                            └── [IRRELEVANT] →
                                                ├── enable_web_search → web_search → generate → ...
                                                └── !enable_web_search → transform_query → retrieve (循环)
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

from src.agent.state import AgentState
from src.agent.nodes import (
    retrieve,
    rerank_documents_node,
    grade_documents,
    transform_query,
    generate,
    check_hallucination,
    web_search_node,
    tool_node,
    bm25_retrieve_node,
    hyde_retrieve_node,
    multi_query_retrieve_node,
    merge_retrieval_node,
    analyze_kg_intent_node,
    kg_retrieve_node,
)
from src.config.settings import settings

logger = logging.getLogger(__name__)


def should_continue_after_grade(state: AgentState) -> Literal["generate", "transform_query", "web_search"]:
    """文档评估后的条件路由

    - 文档相关 → 进入答案生成
    - 文档不相关 + 开启联网搜索 → 联网搜索（降级方案）
    - 文档不相关 + 开启查询重写 + 未超过重试次数 → 查询重写
    - 其他情况 → 进入生成（降级处理）
    """
    enable_web = state.get("enable_web_search", False)
    enable_transform = state.get("enable_transform_query", True)
    logger.info("路由决策: documents_relevant=%s, web_search=%s, transform_query=%s",
                state.get("documents_relevant", False), enable_web, enable_transform)

    if state.get("documents_relevant", False):
        logger.info("路由: 文档相关 → generate")
        return "generate"

    # 向量库无相关文档且开启了联网搜索 → 走联网搜索降级
    if enable_web:
        logger.info("路由: 文档不相关 + 联网搜索已开启 → web_search")
        return "web_search"

    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)

    if enable_transform and iteration < max_iter:
        logger.info("路由: 文档不相关 → transform_query (第 %d/%d 次)", iteration + 1, max_iter)
        return "transform_query"

    logger.info("路由: transform_query 已关闭或超过重试次数 → generate (降级)")
    return "generate"


def should_continue_after_hallucination(
    state: AgentState,
) -> Literal["generate", "end"]:
    """幻觉检测后的条件路由

    - 检测到幻觉且未超过重试次数 → 重新生成
    - 通过或超过重试次数 → 结束
    """
    if state.get("hallucination_detected", False):
        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", 3)
        if iteration < max_iter:
            logger.info("路由: 幻觉检测失败 → 重新生成")
            return "generate"
    logger.info("路由: 幻觉检测通过 → END")
    return "end"


def should_check_hallucination(state: AgentState) -> Literal["check_hallucination", "end"]:
    """生成后路由：是否进行幻觉检测"""
    if state.get("enable_reflection", True):
        return "check_hallucination"
    logger.info("路由: 自反思已关闭，跳过幻觉检测 → END")
    return "end"


def route_retrieval_strategies(
    state: AgentState,
) -> list[Send]:
    """retrieve 后的条件路由：根据开关动态 fan-out 到启用的检索策略。

    只有 enable_xxx=True 的节点才会被 Send 调度执行，
    关闭的节点完全不被调用（零开销）。
    KG 检索只有在 enable_kg=True AND kg_intent=True 时才触发。
    全部关闭时直达 merge_retrieval。
    """
    sends: list[Send] = []
    if state.get("enable_bm25", False):
        sends.append(Send("bm25_retrieve", state))
    if state.get("enable_hyde", False):
        sends.append(Send("hyde_retrieve", state))
    if state.get("enable_multi_query", False):
        sends.append(Send("multi_query_retrieve", state))
    if state.get("enable_kg", False) and state.get("kg_intent", False):
        sends.append(Send("kg_retrieve", state))

    if not sends:
        sends.append(Send("merge_retrieval", state))

    logger.info(
        "检索策略路由: bm25=%s, hyde=%s, multi_query=%s, kg=%s → %d 个 Send",
        state.get("enable_bm25", False),
        state.get("enable_hyde", False),
        state.get("enable_multi_query", False),
        state.get("enable_kg", False) and state.get("kg_intent", False),
        len(sends),
    )
    return sends


def route_after_merge(
    state: AgentState,
) -> Literal["rerank_documents", "grade_documents", "generate"]:
    """merge_retrieval 后的条件路由：根据 rerank 和 grade 开关决定路径。

    - rerank ON  → rerank_documents → (后续再根据 grade 开关决定)
    - rerank OFF + grade ON → 直达 grade_documents
    - 两者均 OFF → 直达 generate
    """
    if state.get("enable_rerank", True):
        return "rerank_documents"
    if state.get("enable_grade_documents", True):
        return "grade_documents"
    logger.info("路由: rerank 和 grade 均已关闭 → generate")
    return "generate"


def route_after_rerank(
    state: AgentState,
) -> Literal["grade_documents", "generate"]:
    """rerank 后的条件路由：是否进行文档评估"""
    if state.get("enable_grade_documents", True):
        return "grade_documents"
    logger.info("路由: grade 已关闭 → generate")
    return "generate"


def build_agent_graph() -> StateGraph:
    """构建 Agent 状态图

    节点:
        retrieve        - 语义检索 + MMR 混合
        bm25_retrieve   - BM25 关键词检索（条件 Send 调度）
        hyde_retrieve   - HyDE 假设文档嵌入检索（条件 Send 调度）
        multi_query_retrieve - Multi-Query 多角度检索（条件 Send 调度）
        merge_retrieval - 合并去重所有检索结果
        rerank_documents - 重排序精排
        grade_documents  - 文档相关性评估
        web_search       - 联网搜索降级
        transform_query  - 查询重写优化
        generate         - 答案生成
        check_hallucination - 幻觉检测

    流转:
        START → retrieve → [条件 Send fan-out bm25/hyde/multi_query]
                 │                    │
                 │                    └→ merge_retrieval
                 │                         │
                 └────────(全关直达)───────┘
                                           │
                                      rerank_documents → grade_documents
                                                              ├── [RELEVANT] → generate → ...
                                                              └── [IRRELEVANT] → web_search / transform_query
    """
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("analyze_kg_intent", analyze_kg_intent_node)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("bm25_retrieve", bm25_retrieve_node)
    workflow.add_node("hyde_retrieve", hyde_retrieve_node)
    workflow.add_node("multi_query_retrieve", multi_query_retrieve_node)
    workflow.add_node("kg_retrieve", kg_retrieve_node)
    workflow.add_node("merge_retrieval", merge_retrieval_node)
    workflow.add_node("rerank_documents", rerank_documents_node)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("generate", generate)
    workflow.add_node("check_hallucination", check_hallucination)
    workflow.add_node("tools", tool_node)

    # 设置入口: KG 意图分析 → 语义检索
    workflow.set_entry_point("analyze_kg_intent")

    # analyze_kg_intent → retrieve（无论是否启用 KG 都执行语义检索）
    workflow.add_edge("analyze_kg_intent", "retrieve")

    # retrieve 后的条件边：fan-out 到启用的检索策略（或直达 merge）
    workflow.add_conditional_edges(
        "retrieve",
        route_retrieval_strategies,
        [
            "bm25_retrieve",
            "hyde_retrieve",
            "multi_query_retrieve",
            "kg_retrieve",
            "merge_retrieval",
        ],
    )

    # 各检索策略 → merge_retrieval fan-in 收敛
    workflow.add_edge("bm25_retrieve", "merge_retrieval")
    workflow.add_edge("hyde_retrieve", "merge_retrieval")
    workflow.add_edge("multi_query_retrieve", "merge_retrieval")
    workflow.add_edge("kg_retrieve", "merge_retrieval")

    # merge 后的条件边：根据开关走 rerank / grade / generate
    workflow.add_conditional_edges(
        "merge_retrieval",
        route_after_merge,
        {
            "rerank_documents": "rerank_documents",
            "grade_documents": "grade_documents",
            "generate": "generate",
        },
    )

    # rerank 后的条件边：是否进行文档评估
    workflow.add_conditional_edges(
        "rerank_documents",
        route_after_rerank,
        {
            "grade_documents": "grade_documents",
            "generate": "generate",
        },
    )

    # 条件边：grade_documents → generate / web_search / transform_query
    workflow.add_conditional_edges(
        "grade_documents",
        should_continue_after_grade,
        {
            "generate": "generate",
            "web_search": "web_search",
            "transform_query": "transform_query",
        },
    )

    # 边：web_search → generate（联网搜索后直接生成答案）
    workflow.add_edge("web_search", "generate")

    # 边：transform_query → retrieve（重新检索循环）
    workflow.add_edge("transform_query", "retrieve")

    # 条件边：generate → check_hallucination（开启自反思）或 END（关闭）
    workflow.add_conditional_edges(
        "generate",
        should_check_hallucination,
        {
            "check_hallucination": "check_hallucination",
            "end": END,
        },
    )

    # 条件边：check_hallucination → generate（重试）或 END
    workflow.add_conditional_edges(
        "check_hallucination",
        should_continue_after_hallucination,
        {
            "generate": "generate",
            "end": END,
        },
    )

    # 编译图（带内存检查点，支持状态持久化）
    checkpointer = MemorySaver()
    compiled_graph = workflow.compile(checkpointer=checkpointer)

    logger.info("Agent 状态图构建完成")
    return compiled_graph


# 全局单例
agent_graph = build_agent_graph()
