"""文档服务层 - 文档上传、查询、删除的业务逻辑"""

import logging
from src.pipeline.indexer import document_indexer
from src.models.document import DocumentInfo, DocumentUploadResponse

logger = logging.getLogger(__name__)


class DocumentService:
    """文档管理服务"""

    def upload_document(self, file_bytes: bytes, filename: str) -> DocumentUploadResponse:
        """上传并索引一个文档

        Args:
            file_bytes: 文件二进制数据
            filename: 原始文件名

        Returns:
            DocumentUploadResponse
        """
        result = document_indexer.ingest(file_bytes, filename)
        return DocumentUploadResponse(
            doc_id=result["doc_id"],
            filename=result["filename"],
            chunk_count=result["chunk_count"],
            message=f"文档 {filename} 上传成功，已分割为 {result['chunk_count']} 个块",
        )

    def list_documents(self) -> list[DocumentInfo]:
        """列出所有已索引的文档"""
        docs = document_indexer.list_documents()
        return [
            DocumentInfo(
                doc_id=d["doc_id"],
                filename=d["filename"],
                file_type=d.get("file_type", "unknown"),
                chunk_count=d.get("chunk_count", 0),
            )
            for d in docs
        ]

    def delete_document(self, doc_id: str) -> int:
        """删除指定文档及其所有向量块"""
        count = document_indexer.delete_document(doc_id)
        logger.info("文档 %s 已删除，移除 %d 个向量块", doc_id, count)
        return count


# 全局单例
document_service = DocumentService()
