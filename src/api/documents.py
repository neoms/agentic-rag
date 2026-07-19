"""文档管理 API 路由"""

import logging
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from src.api.dependencies import get_document_service
from src.services.document_service import DocumentService
from src.models.document import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentDeleteResponse,
)
from src.models.common import SuccessResponse
from src.config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    summary="上传文档并索引",
    description="支持 PDF、Markdown、TXT 格式。文件会自动分块、向量化并存入 ChromaDB。",
)
async def upload_document(
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
):
    # 校验文件扩展名
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

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

    try:
        result = service.upload_document(content, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("文档上传失败")
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")

    return result


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="获取已索引文档列表",
)
async def list_documents(
    service: DocumentService = Depends(get_document_service),
):
    docs = service.list_documents()
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
    count = service.delete_document(doc_id)
    if count == 0:
        raise HTTPException(status_code=404, detail=f"文档 {doc_id} 不存在或无向量块")
    return DocumentDeleteResponse(
        doc_id=doc_id,
        message=f"文档删除成功，已移除 {count} 个向量块",
    )
