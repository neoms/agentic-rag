"""知识图谱模块 - 实体关系抽取、图存储、图谱检索、意图分析

模块结构:
    graph_store.py     - GraphStore：Kuzu 图数据库存储（原生持久化）
    graph_builder.py   - GraphBuilder：LLM 实体关系抽取 + 图构建
    graph_retriever.py - GraphRetriever：实体链接 + 子图提取 + 路径推理 + numpy 向量索引
    kg_intent.py       - KGIntentAnalyzer：LLM 问题意图分析
"""

import logging
from pathlib import Path

from src.knowledge_graph.graph_store import GraphStore
from src.knowledge_graph.graph_builder import GraphBuilder
from src.knowledge_graph.graph_retriever import GraphRetriever
from src.knowledge_graph.kg_intent import KGIntentAnalyzer

logger = logging.getLogger(__name__)

# 全局单例（懒加载，避免在导入时就需要依赖）
_graph_store: GraphStore | None = None
_graph_builder: GraphBuilder | None = None
_graph_retriever: GraphRetriever | None = None
_kg_intent_analyzer: KGIntentAnalyzer | None = None


def get_graph_store() -> GraphStore:
    """获取全局单例 GraphStore

    自动检测 Kuzu 数据库文件是否被外部删除/重建（如数据清理操作），
    如果是则重置单例，避免持久化连接引用已删除的文件。
    """
    global _graph_store
    if _graph_store is None:
        _graph_store = GraphStore()
        return _graph_store

    # 检测数据库文件是否被外部删除/重建
    try:
        db_path = Path(_graph_store._db_path)
        if not db_path.exists():
            logger.warning("Kuzu 数据库文件不存在（可能被清理），重置 GraphStore")
            _graph_store = GraphStore()
        elif db_path.stat().st_mtime > _graph_store._init_time:
            logger.info("Kuzu 数据库文件已被重建，重置 GraphStore")
            _graph_store = GraphStore()
            # 同时重置 GraphRetriever 单例（其 NumpyVectorIndex 也需重建）
            reset_graph_retriever()
    except Exception as e:
        logger.warning("检测 Kuzu 数据库状态时异常，重置 GraphStore: %s", e)
        _graph_store = GraphStore()

    return _graph_store


def reset_graph_retriever() -> None:
    """重置 GraphRetriever 单例（伴随 GraphStore 重建时调用）"""
    global _graph_retriever
    _graph_retriever = None


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
