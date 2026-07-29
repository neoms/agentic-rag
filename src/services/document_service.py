"""文档服务层 - 文档上传、查询、删除的业务逻辑"""

import logging
import threading
import uuid
from datetime import datetime, timezone
from src.pipeline.indexer import document_indexer
from src.models.document import (
    DocumentInfo,
    DocumentUploadResponse,
    TaskStatus,
    TaskInfo,
    TaskSubmitResponse,
)

logger = logging.getLogger(__name__)


class DocumentService:
    """文档管理服务"""

    def __init__(self):
        self._tasks: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _init_task(self, filename: str) -> tuple[str, str]:
        """创建任务记录并返回 (task_id, doc_id)"""
        doc_id = str(uuid.uuid4())
        task_id = doc_id  # 复用 doc_id 作为 task_id
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._tasks[task_id] = {
                "task_id": task_id,
                "doc_id": doc_id,
                "filename": filename,
                "status": TaskStatus.PENDING,
                "message": "任务已提交，等待处理",
                "created_at": now,
                "completed_at": None,
                "chunk_count": 0,
            }
        return task_id, doc_id

    def _background_process(self, file_bytes: bytes, filename: str, doc_id: str, task_id: str):
        """后台线程：执行文档索引"""
        logger.info("[background] 开始处理文档: %s, task_id=%s", filename, task_id)
        with self._lock:
            self._tasks[task_id]["status"] = TaskStatus.PROCESSING
            self._tasks[task_id]["message"] = "正在解析和索引文档..."

        try:
            result = document_indexer.ingest(file_bytes, filename)
            with self._lock:
                self._tasks[task_id]["status"] = TaskStatus.COMPLETED
                self._tasks[task_id]["message"] = (
                    f"文档 {filename} 处理完成，已分割为 {result['chunk_count']} 个块"
                )
                self._tasks[task_id]["chunk_count"] = result["chunk_count"]
                self._tasks[task_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
            logger.info("[background] 文档处理完成: %s, task_id=%s, chunks=%d",
                        filename, task_id, result["chunk_count"])
        except Exception as e:
            with self._lock:
                self._tasks[task_id]["status"] = TaskStatus.FAILED
                self._tasks[task_id]["message"] = f"处理失败: {str(e)}"
                self._tasks[task_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
            logger.exception("[background] 文档处理失败: %s, task_id=%s", filename, task_id)

    def submit_upload_task(self, file_bytes: bytes, filename: str) -> TaskSubmitResponse:
        """提交文档上传任务（异步后台处理）

        立即返回任务信息，实际处理在后台线程进行。
        """
        logger.info("[document_service] 提交后台任务: %s, size=%d bytes", filename, len(file_bytes))
        task_id, doc_id = self._init_task(filename)

        thread = threading.Thread(
            target=self._background_process,
            args=(file_bytes, filename, doc_id, task_id),
            daemon=True,
        )
        thread.start()

        return TaskSubmitResponse(
            task_id=task_id,
            doc_id=doc_id,
            filename=filename,
            status=TaskStatus.PENDING,
            message=f"文档 {filename} 已提交后台处理",
        )

    def upload_document(self, file_bytes: bytes, filename: str) -> DocumentUploadResponse:
        """上传并索引一个文档（同步方式）"""
        logger.info("[document_service] 同步上传文档: %s, size=%d bytes", filename, len(file_bytes))
        result = document_indexer.ingest(file_bytes, filename)
        logger.info("[document_service] 文档处理完成: doc_id=%s, chunks=%d",
                     result["doc_id"], result["chunk_count"])
        return DocumentUploadResponse(
            doc_id=result["doc_id"],
            filename=result["filename"],
            chunk_count=result["chunk_count"],
            message=f"文档 {filename} 上传成功，已分割为 {result['chunk_count']} 个块",
        )

    def get_task(self, task_id: str) -> TaskInfo | None:
        """查询后台任务状态"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            return TaskInfo(**task)

    def list_tasks(self) -> list[TaskInfo]:
        """列出所有后台任务"""
        with self._lock:
            return [TaskInfo(**t) for t in self._tasks.values()]

    def list_documents(self) -> list[DocumentInfo]:
        """列出所有已索引的文档"""
        docs = document_indexer.list_documents()
        logger.info("[document_service] 列出文档: %d 个", len(docs))
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
        logger.info("[document_service] 删除文档: %s", doc_id)
        count = document_indexer.delete_document(doc_id)
        logger.info("文档 %s 已删除，移除 %d 个向量块", doc_id, count)
        return count


# 全局单例
document_service = DocumentService()
