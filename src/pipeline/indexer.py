"""向量化与入库 - 完整的文档摄入管道"""

import hashlib
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from src.pipeline.loader import load_document
from src.pipeline.chunker import chunk_texts
from src.store.vector_store import vector_store
from src.store.document_registry import document_registry
from src.knowledge_graph import get_graph_store, get_graph_builder, get_graph_retriever

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """文档索引器：加载 → 分块 → 向量化 → 入库"""

    def ingest(self, file_bytes: bytes, filename: str) -> dict:
        """处理并索引单个文档

        内置 SHA256 内容去重：若已有完全相同的文件入库，
        直接返回已有 doc_id 跳过处理。

        Args:
            file_bytes: 文件二进制内容
            filename: 文件名

        Returns:
            {"doc_id": ..., "filename": ..., "chunk_count": ...,
             "deduplicated": True/False}
        """
        content_hash = hashlib.sha256(file_bytes).hexdigest()

        # 内容去重：检查是否已存在相同 hash 的文档（优先注册表）
        existing_doc_id = document_registry.find_by_hash(content_hash)
        if existing_doc_id is None:
            existing_doc_id = vector_store.find_by_content_hash(content_hash)
        if existing_doc_id:
            logger.info(
                "内容重复，跳过处理: hash=%s, 已有 doc_id=%s, filename=%s",
                content_hash, existing_doc_id, filename,
            )
            return {
                "doc_id": existing_doc_id,
                "filename": filename,
                "chunk_count": 0,
                "deduplicated": True,
            }

        doc_id = str(uuid.uuid4())
        file_type = Path(filename).suffix.lower().lstrip(".")

        # Step 1: 解析文档
        logger.info("开始解析文档: %s", filename)
        texts = load_document(file_bytes, filename)

        # Step 2: 分块（按文件类型自动选择差异化分块参数）
        base_metadata = {
            "doc_id": doc_id,
            "filename": filename,
            "file_type": file_type,
            "size_bytes": len(file_bytes),
            "content_hash": content_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(
            "开始分块: %s (%d 个片段, type=%s, size=%d)",
            filename, len(texts), file_type, len(file_bytes),
        )
        documents = chunk_texts(
            texts,
            metadata=base_metadata,
            file_type="." + file_type,
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

        # Step 5: 写入独立元数据注册表
        document_registry.register(
            doc_id=doc_id,
            filename=filename,
            file_type=file_type,
            size_bytes=len(file_bytes),
            content_hash=content_hash,
            chunk_count=count,
        )

        logger.info("文档 %s 处理完成: doc_id=%s, chunks=%d", filename, doc_id, count)
        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_count": count,
            "deduplicated": False,
        }

    def delete_document(self, doc_id: str) -> int:
        """级联删除文档：注册表 → 向量库 → 知识图谱

        三处独立存储均按 doc_id 精准清理，共享实体/关系仅移除引用。
        """
        # 1. 注册表（JSON 元数据）
        document_registry.remove(doc_id)
        # 2. 向量库（ChromaDB 向量块）
        chunk_count = vector_store.delete_by_doc_id(doc_id)
        # 3. 知识图谱（实体 + 关系，引用计数式级联删除）
        try:
            kg_entities, kg_relations = get_graph_store().remove_doc_id(doc_id)
            if kg_entities or kg_relations:
                logger.debug(
                    "KG 级联清理: 移除 %d 实体, %d 关系, doc_id=%s",
                    kg_entities, kg_relations, doc_id,
                )
                # 实体发生变更，清除 FAISS/SQLite 缓存，下次搜索时自动重建
                get_graph_retriever().clear_entity_cache()
        except Exception as e:
            logger.warning("KG 清理失败（不影响主流程）: %s", e)
        return chunk_count

    def list_documents(self) -> list[dict]:
        """列出所有已索引的文档（优先使用注册表，失败回退到 ChromaDB）"""
        try:
            docs = document_registry.list_all()
            if docs:
                return docs
        except Exception as e:
            logger.warning("注册表查询失败，回退到 ChromaDB: %s", e)
        return vector_store.list_documents()


# 全局单例
document_indexer = DocumentIndexer()
