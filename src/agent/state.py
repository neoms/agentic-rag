"""Agent 状态定义 - LangGraph StateGraph 的全局共享状态

generate（生成）和 check_hallucination（幻觉检测）已移至外部模块，
因此 answer、hallucination_detected、citation_metadata 字段已移除。
"""

import operator
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Agent 全局状态

    这个 TypedDict 定义了整个 LangGraph 状态图中
    所有节点共享和传递的数据结构。
    """

    # 用户原始问题
    query: str

    # 预计算的问题向量（由缓存层计算后传入，检索节点复用，避免重复调用 Embedding API）
    query_embedding: list[float] | None

    # 当前会话 ID
    session_id: str

    # 对话历史消息（使用 add_messages 自动追加）
    messages: Annotated[list[BaseMessage], add_messages]

    # 当前检索到的文档列表
    documents: list[Document]

    # 重写后的查询（用于查询重写节点）
    rewritten_query: str

    # 检索质量评估结果
    documents_relevant: bool

    # 当前迭代次数（防止无限循环）
    iteration_count: int

    # 最大重试次数
    max_iterations: int

    # Agent 执行路径记录（用于可观测性）
    # 使用 operator.add reducer 以支持并行节点（Send API）的并发写入合并
    agent_path: Annotated[list[str], operator.add]

    # 是否启用流式输出
    stream: bool

    # 工具调用记录
    tool_calls: list[dict]

    # 是否启用联网搜索
    enable_web_search: bool

    # 可控节点开关
    enable_reflection: bool
    enable_rerank: bool
    enable_grade_documents: bool
    enable_transform_query: bool
    enable_bm25: bool
    enable_multi_query: bool

    # 各检索策略独立结果（并行写避免冲突）
    documents_semantic: list[Document]
    documents_bm25: list[Document]
    documents_multi_query: list[Document]

    # 重排序降级原因（API 异常时为非 None 字符串，正常为 None）
    rerank_degraded: str | None

    # 重排序质量跟踪（供"查询重写循环止损"判定）
    rerank_top_score: float              # 本轮 top1 重排分
    best_rerank_score: float             # 跨轮最优 top1 重排分
    rerank_improved: bool                # 本轮相对上轮是否提升（首轮恒为 True）

    # 各策略耗时（毫秒），由 parallel_retrieve_merge_node 返回
    strategy_timings_ms: dict[str, float]

    # 知识图谱检索（由意图分析自动决定是否启用）
    enable_kg: bool                     # 是否启用知识图谱检索
    kg_intent: bool                    # 意图分析结果：是否需要 KG
    kg_context: str                    # KG 检索到的结构化上下文文本

    # 复杂度判定 & 生成结果（judge_complexity / generate 节点使用）
    complexity: str                    # "SIMPLE" 或 "COMPLEX"，由 judge_complexity 节点输出
    answer: str                        # 最终生成的完整回答文本
    citation_metadata: dict            # 引文标注元数据（供前端展示）
