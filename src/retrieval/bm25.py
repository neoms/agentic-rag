"""BM25 关键词检索器 — 基于 jieba 中文分词 + rank-bm25"""

import logging
import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from src.store.vector_store import vector_store
from src.config.settings import settings

logger = logging.getLogger(__name__)


class Bm25Retriever:
    """BM25 关键词检索器

    从 ChromaDB 全量加载文档并构建 BM25 索引，支持 jieba 中文分词。
    索引懒加载并缓存在内存，文档变更后通过 invalidate_cache() 标记重建。
    """

    def __init__(self):
        self._index: BM25Okapi | None = None
        self._docs: list[Document] = []
        self._dirty: bool = True

    def _build_index(self) -> None:
        """从 ChromaDB 加载全量文档并构建 BM25 索引"""
        logger.info("BM25: 开始构建索引...")
        try:
            collection = vector_store._get_or_create_collection()
            data = collection.get()
            ids = data.get("ids", [])
            documents_list = data.get("documents", [])
            metadatas = data.get("metadatas", [])

            self._docs = []
            corpus: list[list[str]] = []

            for i in range(len(ids)):
                if i < len(documents_list) and documents_list[i]:
                    doc = Document(
                        page_content=documents_list[i],
                        metadata=metadatas[i] if i < len(metadatas) else {},
                    )
                    self._docs.append(doc)
                    corpus.append(jieba.lcut(documents_list[i]))

            if corpus:
                self._index = BM25Okapi(corpus)
                logger.info("BM25: 索引构建完成，共 %d 篇文档", len(self._docs))
            else:
                self._index = None
                logger.info("BM25: 无可用文档，跳过索引构建")

            self._dirty = False
        except Exception as e:
            logger.warning("BM25: 索引构建失败: %s", e)
            self._index = None
            self._dirty = False

    def search(self, query: str, top_k: int | None = None) -> list[Document]:
        """BM25 关键词检索

        Args:
            query: 查询文本
            top_k: 返回结果数

        Returns:
            Document 列表
        """
        if self._dirty:
            self._build_index()

        if self._index is None or not self._docs:
            logger.info("BM25: 无可用索引，返回空结果")
            return []

        k = top_k or settings.retrieval_top_k
        try:
            tokens = jieba.lcut(query)
            scores = self._index.get_scores(tokens)

            # 按分数降序取 top_k
            ranked = sorted(
                enumerate(scores), key=lambda x: x[1], reverse=True
            )
            top_indices = [i for i, s in ranked[:k] if s > 0]

            results = [self._docs[i] for i in top_indices]
            logger.info("BM25: 检索完成，返回 %d 个结果", len(results))
            return results
        except Exception as e:
            logger.warning("BM25: 检索异常: %s", e)
            return []

    def invalidate_cache(self) -> None:
        """标记索引为脏，下次检索时重建"""
        self._dirty = True
        logger.info("BM25: 索引缓存已失效")


# 全局单例
bm25_retriever = Bm25Retriever()
