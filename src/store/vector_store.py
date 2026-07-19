"""ChromaDB 向量存储封装 - 支持文档增删查和语义检索"""

import logging
from pathlib import Path
from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.config.settings import settings
from src.backend.embedding import get_embedding_client

logger = logging.getLogger(__name__)

COLLECTION_NAME = "agentic_rag_docs"


class VectorStore:
    """ChromaDB 向量存储管理器（本地持久化模式）"""

    def __init__(self):
        self._client: PersistentClient | None = None
        self._vector_store: Chroma | None = None

    @property
    def persist_dir(self) -> Path:
        """持久化目录路径，自动创建"""
        path = settings.chroma_persist_dir_path
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def client(self) -> PersistentClient:
        """懒连接 ChromaDB 本地持久化客户端"""
        if self._client is None:
            self._client = PersistentClient(
                path=str(self.persist_dir),
            )
            logger.info("ChromaDB 本地持久化已初始化: %s", self.persist_dir)
        return self._client

    @property
    def vector_store(self) -> Chroma:
        """懒初始化 LangChain Chroma 向量存储"""
        if self._vector_store is None:
            embedding = get_embedding_client()
            self._vector_store = Chroma(
                client=self.client,
                collection_name=COLLECTION_NAME,
                embedding_function=embedding,
            )
        return self._vector_store

    def add_documents(self, documents: list[Document], batch_size: int = 100) -> int:
        """批量添加文档到向量库

        Args:
            documents: 文档列表
            batch_size: 批次大小

        Returns:
            添加的文档数量
        """
        total = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.vector_store.add_documents(batch)
            total += len(batch)
            logger.info("已入库 %d 个文档块", total)
        return total

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filter_metadata: dict | None = None,
    ) -> list[tuple[Document, float]]:
        """语义检索

        Args:
            query: 查询文本
            top_k: 返回结果数
            filter_metadata: 元数据过滤条件

        Returns:
            (Document, 相似度分数) 列表
        """
        k = top_k or settings.retrieval_top_k
        results = self.vector_store.similarity_search_with_relevance_scores(
            query, k=k, filter=filter_metadata,
        )
        # 按阈值过滤
        filtered = [
            (doc, score) for doc, score in results
            if score >= settings.retrieval_similarity_threshold
        ]
        logger.info("检索 %d 个结果，过滤后 %d 个", len(results), len(filtered))
        return filtered

    def search_mmr(
        self,
        query: str,
        top_k: int | None = None,
        fetch_k: int = 20,
        lambda_mult: float = 0.7,
    ) -> list[Document]:
        """MMR（最大边际相关性）检索 - 保证结果多样性

        Args:
            query: 查询文本
            top_k: 返回结果数
            fetch_k: 初始获取数量
            lambda_mult: 多样性权重（0=最大多样性，1=最大相似度）

        Returns:
            文档列表
        """
        k = top_k or settings.retrieval_top_k
        return self.vector_store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult,
        )

    def _get_or_create_collection(self):
        """安全获取或创建集合（集合不存在时自动创建）"""
        return self.client.get_or_create_collection(COLLECTION_NAME)

    def delete_by_doc_id(self, doc_id: str) -> int:
        """删除指定 doc_id 的所有文档块

        Args:
            doc_id: 文档 ID

        Returns:
            删除的文档块数量
        """
        collection = self._get_or_create_collection()
        results = collection.get(where={"doc_id": doc_id})
        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
        logger.info("删除文档 %s，共 %d 个块", doc_id, len(ids_to_delete))
        return len(ids_to_delete)

    def list_documents(self) -> list[dict]:
        """列出所有唯一的文档（按 doc_id 去重）"""
        collection = self._get_or_create_collection()
        results = collection.get()
        doc_map: dict[str, dict] = {}
        if results.get("metadatas"):
            for meta in results["metadatas"]:
                doc_id = meta.get("doc_id", "unknown")
                if doc_id not in doc_map:
                    doc_map[doc_id] = {
                        "doc_id": doc_id,
                        "filename": meta.get("filename", "unknown"),
                        "file_type": meta.get("file_type", "unknown"),
                        "chunk_count": 0,
                    }
                doc_map[doc_id]["chunk_count"] += 1
        return list(doc_map.values())

    def get_collection_stats(self) -> dict:
        """获取集合统计信息"""
        collection = self._get_or_create_collection()
        return {
            "name": collection.name,
            "count": collection.count(),
        }


# 全局单例
vector_store = VectorStore()
