"""文档管理 API 路由"""

import logging
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from src.api.dependencies import get_document_service
from src.services.document_service import DocumentService
from src.models.document import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentDeleteResponse,
    TaskSubmitResponse,
    TaskInfo,
)
from src.models.common import SuccessResponse
from src.config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=TaskSubmitResponse,
    summary="上传文档（异步后台索引）",
    description="支持 PDF、Markdown、TXT 格式。文件提交后立即返回任务 ID，索引在后台进行。",
    status_code=202,
)
async def upload_document(
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

    # 读取文件内容
    content = await file.read()
    if len(content) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"文件大小超过限制 ({settings.max_upload_size_mb}MB)",
        )
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="文件内容为空")

    logger.info("文件已读取: %s, size=%d bytes", file.filename, len(content))

    # 提交后台任务（内容校验已在 loader.validate_content 中处理）
    result = service.submit_upload_task(content, file.filename)
    logger.info("API 响应: POST /documents/upload → task_id=%s, status=%s",
                result.task_id, result.status)
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
