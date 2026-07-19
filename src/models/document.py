"""文档相关 Pydantic 模型"""

from datetime import datetime
from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    """文档基本信息"""
    doc_id: str
    filename: str
    file_type: str
    chunk_count: int = 0
    size_bytes: int = 0
    created_at: datetime = Field(default_factory=datetime.now)


class DocumentUploadResponse(BaseModel):
    """文档上传响应"""
    success: bool = True
    doc_id: str
    filename: str
    chunk_count: int
    message: str


class DocumentListResponse(BaseModel):
    """文档列表响应"""
    documents: list[DocumentInfo]
    total: int


class DocumentDeleteResponse(BaseModel):
    """文档删除响应"""
    success: bool = True
    doc_id: str
    message: str
