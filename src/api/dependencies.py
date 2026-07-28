"""API 依赖注入 - 管理全局 Service 单例的获取"""

from src.services.document_service import document_service
from src.services.rag_service import rag_service
from src.knowledge_graph import get_graph_store
from src.knowledge_graph.graph_store import GraphStore


def get_document_service():
    return document_service


def get_rag_service():
    return rag_service


def get_kg_store() -> GraphStore:
    return get_graph_store()
