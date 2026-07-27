"""Agent 状态定义 - LangGraph StateGraph 的全局共享状态"""

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

    # 最终生成的答案
    answer: str

    # 幻觉检测结果
    hallucination_detected: bool

    # Agent 执行路径记录（用于可观测性）
    agent_path: list[str]

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
    enable_hyde: bool
    enable_multi_query: bool

    # 各检索策略独立结果（并行写避免冲突）
    documents_bm25: list[Document]
    documents_hyde: list[Document]
    documents_multi_query: list[Document]
