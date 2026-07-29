"""文档相关 Pydantic 模型"""

import enum
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


class TaskStatus(str, enum.Enum):
    """后台任务状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskInfo(BaseModel):
    """后台任务信息"""
    task_id: str
    doc_id: str
    filename: str
    status: TaskStatus
    message: str
    created_at: str = ""
    completed_at: str | None = None
    chunk_count: int = 0


class TaskSubmitResponse(BaseModel):
    """任务提交响应"""
    success: bool = True
    task_id: str
    doc_id: str
    filename: str
    status: TaskStatus
    message: str
