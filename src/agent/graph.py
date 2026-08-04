"""LangGraph 状态图构建 - Agent 节点编排与条件路由

状态流转图：
    START → analyze_kg_intent → parallel_retrieve_merge → rerank_documents
                                                           ↓
                                                     grade_documents
                                                     ├── [RELEVANT] → judge_complexity (LLM_MODEL_FAST)
                                                     │                  ├── SIMPLE → generate_simple (LLM_MODEL_FAST)
                                                     │                  └── COMPLEX → generate_complex (LLM_MODEL_STRONG)
                                                     │                               ↓
                                                     │                         check_hallucination (图外)
                                                     └── [IRRELEVANT] →
                                                         ├── enable_web_search → web_search → judge_complexity
                                                         └── !enable_web_search → transform_query → retrieve (循环)

parallel_retrieve_merge 内部合并了所有并行策略（BM25/Multi-Query/KG），
消除了旧架构中 LangGraph Send fan-in 导致 merge 被多次调用的状态覆盖 bug。

judge_complexity + generate_simple/complex 已迁移到图内。
check_hallucination 仍在图外由 rag_service 完成。
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import AgentState
from src.agent.nodes import (
    retrieve,
    rerank_documents_node,
    grade_documents,
    transform_query,
    web_search_node,
    analyze_kg_intent_node,
    parallel_retrieve_merge_node,
    judge_complexity_node,
    generate_simple_node,
    generate_complex_node,
    route_after_judge,
)

logger = logging.getLogger(__name__)


def should_continue_after_grade(state: AgentState) -> Literal["judge_complexity", "transform_query", "web_search"]:
    """文档评估后的条件路由

    - 文档相关 → judge_complexity（统一走图内复杂度判定）
    - 文档不相关 + 文档为空（向量库无匹配）→ 跳过查询重写，走 web_search 或降级 judge
    - 文档不相关 + 开启联网搜索 → 联网搜索（降级方案）
    - 文档不相关 + 开启查询重写 + 未超过重试次数 → 查询重写
    - 其他情况 → judge_complexity（降级处理）
    """
    enable_web = state.get("enable_web_search", False)
    enable_transform = state.get("enable_transform_query", True)
    documents = state.get("documents", [])
    logger.info("路由决策: documents_relevant=%s, documents_count=%d, web_search=%s, transform_query=%s",
                state.get("documents_relevant", False), len(documents), enable_web, enable_transform)

    if state.get("documents_relevant", False):
        logger.info("路由: 文档相关 → judge_complexity")
        return "judge_complexity"

    if not documents:
        if enable_web:
            logger.info("路由: 文档为空 + 联网搜索已开启 → web_search")
            return "web_search"
        logger.info("路由: 文档为空 → judge_complexity (降级)")
        return "judge_complexity"

    if enable_web:
        logger.info("路由: 文档不相关 + 联网搜索已开启 → web_search")
        return "web_search"

    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)

    if enable_transform and iteration < max_iter:
        # 重写循环止损：重写后检索质量（top1 重排分）没有提升则停止重写，
        # 避免空转多轮（重写查询语义等价时检索结果不变，再重写也是浪费）。
        if not state.get("rerank_improved", True):
            logger.info(
                "路由: 重写后检索无改善 (top1=%.4f) → judge_complexity (降级)",
                state.get("rerank_top_score", 0.0),
            )
            return "judge_complexity"
        logger.info("路由: 文档不相关 → transform_query (第 %d/%d 次)", iteration + 1, max_iter)
        return "transform_query"

    logger.info("路由: transform_query 已关闭或超过重试次数 → judge_complexity (降级)")
    return "judge_complexity"


def route_after_merge(
    state: AgentState,
) -> Literal["rerank_documents", "grade_documents", "judge_complexity"]:
    """并行检索合并后的条件路由

    - rerank 开启 → 正常走重排序
    - rerank 关闭 + grade 开启 → 跳过 rerank 直接评估
    - 两者都关 → 跳过所有，直接去 judge_complexity
    """
    if state.get("enable_rerank", True):
        return "rerank_documents"
    if state.get("enable_grade_documents", True):
        logger.info("路由: rerank 已关闭 → 跳过 rerank，直接 grade")
        return "grade_documents"
    logger.info("路由: rerank+grade 均关闭 → judge_complexity")
    return "judge_complexity"


def route_after_rerank(
    state: AgentState,
) -> Literal["grade_documents", "judge_complexity"]:
    """rerank 后的条件路由：是否进行文档评估"""
    if state.get("enable_grade_documents", True):
        return "grade_documents"
    logger.info("路由: grade 已关闭 → judge_complexity")
    return "judge_complexity"


def build_agent_graph() -> StateGraph:
    """构建 Agent 状态图

    节点:
        parallel_retrieve_merge - 语义+MMR + 线程池并行策略 + 合并（多合一）
        retrieve               - 语义 + MMR（仅用于查询重写循环）
        rerank_documents       - 重排序精排
        grade_documents        - 文档相关性评估
        web_search              - 联网搜索降级
        transform_query        - 查询重写优化
        judge_complexity       - 复杂度判定（LLM_MODEL_FAST）
        generate_simple        - 简单生成（LLM_MODEL_FAST 流式）
        generate_complex       - 复杂生成（LLM_MODEL_STRONG 流式）

    并行策略（BM25/Multi-Query/KG）全部在 parallel_retrieve_merge 内部
    使用 ThreadPoolExecutor 并行执行，merge 只调用一次，消除状态覆盖 bug。

    judge_complexity / generate_simple / generate_complex 为图内节点。
    check_hallucination 仍由外部 rag_service 处理。
    """
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("analyze_kg_intent", analyze_kg_intent_node)
    workflow.add_node("parallel_retrieve_merge", parallel_retrieve_merge_node)
    workflow.add_node("retrieve", retrieve)  # 仅用于 transform_query 循环
    workflow.add_node("rerank_documents", rerank_documents_node)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("judge_complexity", judge_complexity_node)
    workflow.add_node("generate_simple", generate_simple_node)
    workflow.add_node("generate_complex", generate_complex_node)

    # ── 入口 → 意图分析 → 并行检索 + 合并 ──
    workflow.set_entry_point("analyze_kg_intent")
    workflow.add_edge("analyze_kg_intent", "parallel_retrieve_merge")

    # ── 并行检索 → rerank / grade / judge（条件跳过 rerank） ──
    workflow.add_conditional_edges(
        "parallel_retrieve_merge",
        route_after_merge,
        {
            "rerank_documents": "rerank_documents",
            "grade_documents": "grade_documents",
            "judge_complexity": "judge_complexity",
        },
    )

    # ── 查询重写循环 → retrieve → rerank ──
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("retrieve", "rerank_documents")

    # ── rerank → grade / judge ──
    workflow.add_conditional_edges(
        "rerank_documents",
        route_after_rerank,
        {
            "grade_documents": "grade_documents",
            "judge_complexity": "judge_complexity",
        },
    )

    # ── grade → judge / web_search / transform_query ──
    workflow.add_conditional_edges(
        "grade_documents",
        should_continue_after_grade,
        {
            "judge_complexity": "judge_complexity",
            "web_search": "web_search",
            "transform_query": "transform_query",
        },
    )

    # ── web_search → judge_complexity ──
    workflow.add_edge("web_search", "judge_complexity")

    # ── judge_complexity → generate_simple / generate_complex（条件路由） ──
    workflow.add_conditional_edges(
        "judge_complexity",
        route_after_judge,
        {
            "generate_simple": "generate_simple",
            "generate_complex": "generate_complex",
        },
    )

    # ── 生成节点 → END ──
    workflow.add_edge("generate_simple", END)
    workflow.add_edge("generate_complex", END)

    # 编译图（带内存检查点，支持状态持久化）
    checkpointer = MemorySaver()
    compiled_graph = workflow.compile(checkpointer=checkpointer)

    logger.info("Agent 状态图构建完成（含复杂度判定+双生成节点）")
    return compiled_graph


# 全局单例
agent_graph = build_agent_graph()
