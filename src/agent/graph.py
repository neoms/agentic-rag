"""LangGraph 状态图构建 - Agent 节点编排与条件路由

状态流转图：
    START → retrieve → grade_documents
                          ├── [RELEVANT] → generate → check_hallucination
                          │                              ├── [PASSED] → END
                          │                              └── [FAILED] → END
                          └── [IRRELEVANT] → transform_query → retrieve (循环, 最多 max_iterations 次)
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import AgentState
from src.agent.nodes import (
    retrieve,
    grade_documents,
    transform_query,
    generate,
    check_hallucination,
    tool_node,
)
from src.config.settings import settings

logger = logging.getLogger(__name__)


def should_continue_after_grade(state: AgentState) -> Literal["generate", "transform_query"]:
    """文档评估后的条件路由

    - 文档相关 → 进入答案生成
    - 文档不相关且未超过重试次数 → 进入查询重写
    - 文档不相关且超过重试次数 → 也进入生成（降级处理）
    """
    if state.get("documents_relevant", False):
        logger.info("路由: 文档相关 → generate")
        return "generate"

    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)

    if iteration < max_iter:
        logger.info("路由: 文档不相关 → transform_query (第 %d/%d 次)", iteration + 1, max_iter)
        return "transform_query"
    else:
        logger.info("路由: 超过重试次数 → generate (降级)")
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


def build_agent_graph() -> StateGraph:
    """构建 Agent 状态图

    节点:
        retrieve        - 语义检索
        grade_documents  - 文档相关性评估（自反思）
        transform_query  - 查询重写优化
        generate         - 答案生成
        check_hallucination - 幻觉检测
        tools            - Tool Calling（联网搜索、计算器等）

    流转:
        START → retrieve → grade_documents
                 ↑            ├── [RELEVANT] → generate → check_hallucination → END
                 │            └── [IRRELEVANT] → transform_query ──┘
                 └────────── (循环, max_iterations 次)
    """
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("generate", generate)
    workflow.add_node("check_hallucination", check_hallucination)
    workflow.add_node("tools", tool_node)

    # 设置入口
    workflow.set_entry_point("retrieve")

    # 边：retrieve → grade_documents
    workflow.add_edge("retrieve", "grade_documents")

    # 条件边：grade_documents → generate 或 transform_query
    workflow.add_conditional_edges(
        "grade_documents",
        should_continue_after_grade,
        {
            "generate": "generate",
            "transform_query": "transform_query",
        },
    )

    # 边：transform_query → retrieve（重新检索循环）
    workflow.add_edge("transform_query", "retrieve")

    # 边：generate → check_hallucination
    workflow.add_edge("generate", "check_hallucination")

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
