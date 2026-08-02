"""文档管理 API 路由"""

import logging
import shutil
import uuid
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi import Request
from pathlib import Path
from src.api.dependencies import get_document_service
from src.services.document_service import DocumentService, QueueFullError
from src.models.document import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentDeleteResponse,
    TaskSubmitResponse,
    TaskInfo,
)
from src.models.common import SuccessResponse
from src.config.settings import settings
from src.metrics import uploads_total, uploads_failed_total

logger = logging.getLogger(__name__)

# 上传分块大小与 multipart 头部开销预留（Content-Length 预检用）
UPLOAD_CHUNK_SIZE = 256 * 1024
_MULTIPART_OVERHEAD = 1024 * 1024

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=TaskSubmitResponse,
    summary="上传文档（异步后台索引）",
    description="支持 PDF、Markdown、TXT 格式。文件提交后立即返回任务 ID，索引在后台进行。",
    status_code=202,
)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
):
    # 校验文件扩展名
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    logger.info("API 请求: POST /documents/upload (async), filename=%s", file.filename)

    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in [f".{e}" for e in settings.allowed_extensions_list]:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}，支持的格式: {settings.allowed_extensions_list}",
        )

    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    # Content-Length 预检：multipart 开销放宽 1MB，超大请求在读取前直接拒绝
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > max_bytes + _MULTIPART_OVERHEAD:
                uploads_failed_total.inc()
                raise HTTPException(
                    status_code=413,
                    detail=f"文件大小超过限制 ({settings.max_upload_size_mb}MB)",
                )
        except ValueError:
            pass

    # 分块流式写入临时文件，边写边计数，超限立即中止（超大文件不会全量进内存）
    temp_uploads = settings.project_root / "data" / "temp_uploads"
    temp_uploads.mkdir(parents=True, exist_ok=True)
    spool_path = temp_uploads / f"{uuid.uuid4().hex}.upload"
    handoff: Path | None = None
    total = 0
    try:
        with open(spool_path, "wb") as out:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    uploads_failed_total.inc()
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件大小超过限制 ({settings.max_upload_size_mb}MB)",
                    )
                out.write(chunk)
        if total == 0:
            uploads_failed_total.inc()
            raise HTTPException(status_code=400, detail="文件内容为空")

        # 小文件读入内存走原路径；大文件把临时文件移交给后台（路径传递，避免二次内存拷贝）
        large_threshold = settings.large_file_threshold_mb * 1024 * 1024
        if total <= large_threshold:
            with open(spool_path, "rb") as f:
                file_source: bytes | Path = f.read()
        else:
            suffix = Path(file.filename).suffix or ".tmp"
            handoff = temp_uploads / f"{uuid.uuid4().hex}{suffix}"
            shutil.move(str(spool_path), str(handoff))
            file_source = handoff
            logger.info("大文件已移交后台: %s (%.1f MB)", handoff, total / 1024 / 1024)

        logger.info("文件已接收: %s, size=%d bytes", file.filename, total)

        # 提交后台任务（内容校验已在 loader.validate_content 中处理）
        try:
            result = service.submit_upload_task(file_source, file.filename)
        except QueueFullError as e:
            # 队列已满：清理已移交的临时文件后返回 429
            uploads_failed_total.inc()
            if handoff is not None and handoff.exists():
                handoff.unlink(missing_ok=True)
                logger.info("队列已满，已清理临时文件: %s", handoff)
            logger.warning("上传被拒绝（索引队列已满）: filename=%s", file.filename)
            raise HTTPException(status_code=429, detail=str(e)) from e
        except Exception:
            # 其他异常同样清理已移交的临时文件
            if handoff is not None and handoff.exists():
                handoff.unlink(missing_ok=True)
            raise
    finally:
        # 小文件路径：spool 已删除；大文件路径：已移交（成功交给后台清理，失败已在上方清理）
        if spool_path.exists():
            spool_path.unlink(missing_ok=True)

    logger.info("API 响应: POST /documents/upload → task_id=%s, status=%s",
                result.task_id, result.status)
    uploads_total.inc()
    return result


@router.get(
    "/tasks/{task_id}",
    response_model=TaskInfo,
    summary="查询后台任务状态",
    description="通过任务 ID 查询文档索引后台处理的实时状态。",
)
async def get_task_status(
    task_id: str,
    service: DocumentService = Depends(get_document_service),
):
    logger.info("API 请求: GET /documents/tasks/%s", task_id)
    task = service.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    logger.info("API 响应: GET /documents/tasks/%s → status=%s", task_id, task.status)
    return task


@router.get(
    "/tasks",
    response_model=list[TaskInfo],
    summary="获取所有后台任务状态",
)
async def list_tasks(
    service: DocumentService = Depends(get_document_service),
):
    logger.info("API 请求: GET /documents/tasks")
    tasks = service.list_tasks()
    logger.info("API 响应: GET /documents/tasks → count=%d", len(tasks))
    return tasks


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="获取已索引文档列表",
)
async def list_documents(
    service: DocumentService = Depends(get_document_service),
):
    logger.info("API 请求: GET /documents")
    docs = service.list_documents()
    logger.info("API 响应: GET /documents → total=%d", len(docs))
    return DocumentListResponse(documents=docs, total=len(docs))


@router.delete(
    "/{doc_id}",
    response_model=DocumentDeleteResponse,
    summary="删除指定文档",
    description="删除文档及其所有向量块",
)
async def delete_document(
    doc_id: str,
    service: DocumentService = Depends(get_document_service),
):
    logger.info("API 请求: DELETE /documents/%s", doc_id)
    count = service.delete_document(doc_id)
    if count == 0:
        raise HTTPException(status_code=404, detail=f"文档 {doc_id} 不存在或无向量块")
    logger.info("API 响应: DELETE /documents/%s → 删除了 %d 个向量块", doc_id, count)
    return DocumentDeleteResponse(
        doc_id=doc_id,
        message=f"文档删除成功，已移除 {count} 个向量块",
    )
