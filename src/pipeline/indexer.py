"""向量化与入库 - 完整的文档摄入管道"""

import uuid
import logging
from pathlib import Path
from src.pipeline.loader import load_document
from src.pipeline.chunker import chunk_texts
from src.store.vector_store import vector_store
from src.knowledge_graph import get_graph_store, get_graph_builder

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """文档索引器：加载 → 分块 → 向量化 → 入库"""

    def ingest(self, file_bytes: bytes, filename: str) -> dict:
        """处理并索引单个文档

        Args:
            file_bytes: 文件二进制内容
            filename: 文件名

        Returns:
            {"doc_id": ..., "filename": ..., "chunk_count": ...}
        """
        doc_id = str(uuid.uuid4())
        file_type = Path(filename).suffix.lower().lstrip(".")

        # Step 1: 解析文档
        logger.info("开始解析文档: %s", filename)
        texts = load_document(file_bytes, filename)

        # Step 2: 分块
        logger.info("开始分块: %s (%d 个片段)", filename, len(texts))
        documents = chunk_texts(
            texts,
            metadata={
                "doc_id": doc_id,
                "filename": filename,
                "file_type": file_type,
            },
        )

        # Step 3: 向量化入库
        logger.info("开始入库: %s (%d 个文档块)", filename, len(documents))
        count = vector_store.add_documents(documents)

        # Step 4: 构建知识图谱
        try:
            logger.info("开始构建知识图谱: %s", filename)
            store = get_graph_store()
            builder = get_graph_builder()
            builder.build_from_chunks(documents, doc_id, store)
            logger.info("知识图谱构建完成: %s", filename)
        except Exception as e:
            logger.warning("知识图谱构建失败（不影响向量检索）: %s", e)

        logger.info("文档 %s 处理完成: doc_id=%s, chunks=%d", filename, doc_id, count)
        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_count": count,
        }

    def delete_document(self, doc_id: str) -> int:
        """从向量库中删除文档"""
        return vector_store.delete_by_doc_id(doc_id)

    def list_documents(self) -> list[dict]:
        """列出所有已索引的文档"""
        return vector_store.list_documents()


# 全局单例
document_indexer = DocumentIndexer()
