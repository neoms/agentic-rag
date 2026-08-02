"""文档服务层 - 文档上传、查询、删除的业务逻辑

支持大文件流式处理：超过 large_file_threshold_mb 的文件，
先保存到临时目录再后台处理，避免大量内存占用。
"""

import logging
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from src.config.settings import settings
from src.cache import get_cache_service
from src.store.state_store import get_runtime_state_store
from src.pipeline.indexer import document_indexer
from src.models.document import (
    DocumentInfo,
    DocumentUploadResponse,
    TaskStatus,
    TaskInfo,
    TaskSubmitResponse,
)

logger = logging.getLogger(__name__)

TEMP_UPLOAD_DIR = "data/temp_uploads"


class QueueFullError(Exception):
    """索引任务队列已满，拒绝接受新的上传"""


class DocumentService:
    """文档管理服务"""

    def __init__(self):
        self._tasks: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._store = get_runtime_state_store()
        self._restore_tasks()
        # 有界 worker 池：并发受 index_workers 限制，排队受 index_queue_max 限制
        self._executor = ThreadPoolExecutor(
            max_workers=settings.index_workers,
            thread_name_prefix="doc-index",
        )
        self._queue_sem = threading.BoundedSemaphore(
            settings.index_workers + settings.index_queue_max
        )
        logger.info(
            "DocumentService 索引队列: workers=%d, queue_max=%d",
            settings.index_workers,
            settings.index_queue_max,
        )

    def _restore_tasks(self):
        """启动时恢复任务表：先清理历史任务，未完成任务标记中断，其余载入内存"""
        try:
            self._store.prune_tasks(
                keep=settings.task_history_keep,
                ttl_days=settings.task_history_ttl_days,
            )
            interrupted = self._store.mark_interrupted_tasks()
            for task in self._store.list_tasks():
                self._tasks[task["task_id"]] = task
            if interrupted:
                logger.info("已将 %d 个中断任务标记为 failed", interrupted)
            logger.info("从持久化恢复 %d 个上传任务", len(self._tasks))
        except Exception as e:
            logger.warning("上传任务持久化恢复失败（不影响使用）: %s", e)

    def _prune_tasks(self):
        """惰性清理历史任务并同步内存（失败仅告警）"""
        try:
            self._store.prune_tasks(
                keep=settings.task_history_keep,
                ttl_days=settings.task_history_ttl_days,
            )
            with self._lock:
                self._tasks = {t["task_id"]: t for t in self._store.list_tasks()}
        except Exception as e:
            logger.warning("任务清理失败（不影响使用）: %s", e)

    def _persist_task(self, task_id: str):
        """将当前任务状态快照写入持久化存储（失败仅告警）"""
        with self._lock:
            task = dict(self._tasks[task_id])
        try:
            self._store.upsert_task(task)
        except Exception as e:
            logger.warning("任务状态持久化失败 task_id=%s: %s", task_id, e)

    def _worker_wrapper(
        self,
        file_source: bytes | Path,
        filename: str,
        doc_id: str,
        task_id: str,
    ):
        """worker 池执行入口：任务结束后释放队列信号量"""
        try:
            self._background_process(file_source, filename, doc_id, task_id)
        finally:
            self._queue_sem.release()

    @property
    def _temp_dir(self) -> Path:
        """临时文件目录"""
        path = settings.project_root / TEMP_UPLOAD_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _is_large_file(self, size_bytes: int) -> bool:
        """判断文件是否超过大文件阈值"""
        return size_bytes > settings.large_file_threshold_mb * 1024 * 1024

    def _save_to_temp(self, file_bytes: bytes, filename: str) -> Path:
        """将文件保存到临时目录

        Returns:
            临时文件路径
        """
        suffix = Path(filename).suffix or ".tmp"
        tmp_name = f"{uuid.uuid4().hex}{suffix}"
        tmp_path = self._temp_dir / tmp_name
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)
        logger.debug("大文件已写入临时路径: %s (%d bytes)", tmp_path, len(file_bytes))
        return tmp_path

    def _cleanup_temp(self, tmp_path: Path):
        """清理临时文件"""
        try:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
                logger.debug("临时文件已清理: %s", tmp_path)
        except OSError as e:
            logger.warning("临时文件清理失败 %s: %s", tmp_path, e)

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
        self._persist_task(task_id)
        return task_id, doc_id

    def _background_process(
        self,
        file_source: bytes | Path,
        filename: str,
        doc_id: str,
        task_id: str,
    ):
        """后台线程：执行文档索引

        Args:
            file_source: 文件内容（bytes）或临时文件路径（Path）
        """
        logger.info("[background] 开始处理文档: %s, task_id=%s", filename, task_id)
        with self._lock:
            self._tasks[task_id]["status"] = TaskStatus.PROCESSING
            self._tasks[task_id]["message"] = "正在解析和索引文档..."
        self._persist_task(task_id)

        tmp_path: Path | None = file_source if isinstance(file_source, Path) else None
        try:
            # 从临时文件读取（若 file_source 是 Path）
            if isinstance(file_source, Path):
                with open(file_source, "rb") as f:
                    file_bytes = f.read()
            else:
                file_bytes = file_source

            result = document_indexer.ingest(file_bytes, filename)

            with self._lock:
                self._tasks[task_id]["status"] = TaskStatus.COMPLETED
                self._tasks[task_id]["message"] = (
                    f"文档 {filename} 处理完成，已分割为 {result['chunk_count']} 个块"
                )
                self._tasks[task_id]["chunk_count"] = result["chunk_count"]
                self._tasks[task_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
            self._persist_task(task_id)
            logger.info("[background] 文档处理完成: %s, task_id=%s, chunks=%d",
                        filename, task_id, result["chunk_count"])
        except Exception as e:
            with self._lock:
                self._tasks[task_id]["status"] = TaskStatus.FAILED
                self._tasks[task_id]["message"] = f"处理失败: {str(e)}"
                self._tasks[task_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
            self._persist_task(task_id)
            logger.exception("[background] 文档处理失败: %s, task_id=%s", filename, task_id)
        finally:
            # 清理临时文件
            if tmp_path:
                self._cleanup_temp(tmp_path)

    def submit_upload_task(
        self, file_input: bytes | Path, filename: str
    ) -> TaskSubmitResponse:
        """提交文档上传任务（有界队列异步处理）

        小文件传 bytes（服务层按需落盘）；大文件可由路由层直接传临时文件 Path，
        避免二次内存拷贝与重复落盘。

        立即返回任务信息，实际处理由 worker 池执行；队列已满时抛出
        QueueFullError（路由层转为 HTTP 429）。
        """
        # 队列有界：获取信号量失败直接拒绝（不创建任务记录）
        if not self._queue_sem.acquire(blocking=False):
            raise QueueFullError(
                f"索引队列已满（最多排队 {settings.index_queue_max} 个任务），请稍后重试"
            )

        try:
            task_id, doc_id = self._init_task(filename)

            if isinstance(file_input, Path):
                size = file_input.stat().st_size
                file_source: bytes | Path = file_input
                logger.info(
                    "[document_service] 提交后台任务(Path): %s, size=%d bytes",
                    filename, size,
                )
            else:
                size = len(file_input)
                logger.info(
                    "[document_service] 提交后台任务: %s, size=%d bytes",
                    filename, size,
                )
                if self._is_large_file(size):
                    # 大文件：保存到临时文件，传递路径
                    tmp_path = self._save_to_temp(file_input, filename)
                    file_source = tmp_path
                    logger.info(
                        "大文件已暂存到临时文件: %s (%.1f MB)",
                        tmp_path, size / 1024 / 1024,
                    )
                else:
                    file_source = file_input

            # 惰性清理历史任务（新任务已落库，重建内存任务表）
            self._prune_tasks()

            self._executor.submit(
                self._worker_wrapper, file_source, filename, doc_id, task_id,
            )
        except BaseException:
            self._queue_sem.release()
            raise

        return TaskSubmitResponse(
            task_id=task_id,
            doc_id=doc_id,
            filename=filename,
            status=TaskStatus.PENDING,
            message=f"文档 {filename} 已提交后台处理",
        )

    def shutdown(self, wait: bool = True) -> None:
        """优雅关闭：等待在途索引任务完成，取消排队任务"""
        logger.info("DocumentService 关闭: 等待后台索引任务完成...")
        self._executor.shutdown(wait=wait, cancel_futures=True)

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
        """删除指定文档及其所有向量块，并精确失效引用该文档的缓存"""
        logger.info("[document_service] 删除文档: %s", doc_id)
        count = document_indexer.delete_document(doc_id)
        try:
            invalidated = get_cache_service().invalidate_documents([doc_id])
            if invalidated:
                logger.info(
                    "[document_service] 已失效引用该文档的缓存 %d 条", invalidated,
                )
        except Exception as e:
            logger.warning(
                "[document_service] 缓存失效异常（不影响文档删除）: %s", e,
            )
        logger.info("文档 %s 已删除，移除 %d 个向量块", doc_id, count)
        return count


# 全局单例
document_service = DocumentService()
