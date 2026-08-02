"""向量化与入库 - 完整的文档摄入管道"""

import hashlib
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from src.pipeline.loader import load_document
from src.pipeline.chunker import chunk_texts
from src.store.vector_store import vector_store
from src.cache import get_cache_service
from src.knowledge_graph import get_graph_store, get_graph_builder, get_graph_retriever

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """文档索引器：加载 → 分块 → 向量化 → 入库"""

    def ingest(self, file_bytes: bytes, filename: str) -> dict:
        """处理并索引单个文档

        内置 SHA256 内容去重（通过 ChromaDB metadata 查询）。
        文档元数据（doc_id, filename, content_hash 等）随每文档块的 metadata 一同写入 ChromaDB，
        无需独立注册表。

        Args:
            file_bytes: 文件二进制内容
            filename: 文件名

        Returns:
            {"doc_id": ..., "filename": ..., "chunk_count": ...,
             "deduplicated": True/False}
        """
        content_hash = hashlib.sha256(file_bytes).hexdigest()

        # 内容去重：通过 ChromaDB metadata 查询
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

        # 同名文档内容更新：精确失效引用旧版本的缓存（不删除旧文档本身）
        try:
            stale_doc_ids = [
                old_id for old_id in vector_store.find_by_filename(filename)
                if old_id != doc_id
            ]
            if stale_doc_ids:
                invalidated = get_cache_service().invalidate_documents(stale_doc_ids)
                if invalidated:
                    logger.info(
                        "文档更新，已失效引用旧版本的缓存 %d 条 (doc_ids=%s)",
                        invalidated, stale_doc_ids,
                    )
        except Exception as e:
            logger.warning("同名文档缓存失效异常（不影响索引）: %s", e)

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

        # Step 3: 向量化入库（doc_id/filename/content_hash 等元数据自动随块写入 ChromaDB）
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
            "deduplicated": False,
        }

    def delete_document(self, doc_id: str) -> int:
        """级联删除文档：向量库 → 知识图谱

        文档元数据（doc_id/filename/content_hash）随 ChromaDB 文档块一并删除。
        """
        # 1. 向量库（ChromaDB 向量块，元数据一并清除）
        chunk_count = vector_store.delete_by_doc_id(doc_id)
        # 2. 知识图谱（实体 + 关系，引用计数式级联删除）
        try:
            kg_entities, kg_relations = get_graph_store().remove_doc_id(doc_id)
            if kg_entities or kg_relations:
                logger.debug(
                    "KG 级联清理: 移除 %d 实体, %d 关系, doc_id=%s",
                    kg_entities, kg_relations, doc_id,
                )
                # 实体发生变更，标记向量索引为脏，下次检索时自动重建
                get_graph_retriever().mark_dirty()
        except Exception as e:
            logger.warning("KG 清理失败（不影响主流程）: %s", e)
        return chunk_count

    def list_documents(self) -> list[dict]:
        """列出所有已索引的文档（通过 ChromaDB metadata 去重）"""
        return vector_store.list_documents()


# 全局单例
document_indexer = DocumentIndexer()
