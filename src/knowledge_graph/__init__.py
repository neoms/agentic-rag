"""知识图谱模块 - 实体关系抽取、图存储、图谱检索、意图分析

模块结构:
    graph_store.py    - GraphStore：NetworkX 图存储 + JSON 持久化
    graph_builder.py  - GraphBuilder：LLM 实体关系抽取 + 图构建
    graph_retriever.py - GraphRetriever：实体链接 + 子图提取 + 路径推理
    kg_intent.py      - KGIntentAnalyzer：LLM 问题意图分析
"""

from src.knowledge_graph.graph_store import GraphStore
from src.knowledge_graph.graph_builder import GraphBuilder
from src.knowledge_graph.graph_retriever import GraphRetriever
from src.knowledge_graph.kg_intent import KGIntentAnalyzer

# 全局单例（懒加载，避免在导入时就需要依赖）
_graph_store: GraphStore | None = None
_graph_builder: GraphBuilder | None = None
_graph_retriever: GraphRetriever | None = None
_kg_intent_analyzer: KGIntentAnalyzer | None = None


def get_graph_store() -> GraphStore:
    """获取全局单例 GraphStore"""
    global _graph_store
    if _graph_store is None:
        _graph_store = GraphStore()
    return _graph_store


def get_graph_builder() -> GraphBuilder:
    """获取全局单例 GraphBuilder"""
    global _graph_builder
    if _graph_builder is None:
        _graph_builder = GraphBuilder()
    return _graph_builder


def get_graph_retriever() -> GraphRetriever:
    """获取全局单例 GraphRetriever"""
    global _graph_retriever
    if _graph_retriever is None:
        _graph_retriever = GraphRetriever()
    return _graph_retriever


def get_kg_intent_analyzer() -> KGIntentAnalyzer:
    """获取全局单例 KGIntentAnalyzer"""
    global _kg_intent_analyzer
    if _kg_intent_analyzer is None:
        _kg_intent_analyzer = KGIntentAnalyzer()
    return _kg_intent_analyzer
