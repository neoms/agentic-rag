"""通用响应模型"""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    checks: dict = Field(default_factory=dict)  # 逐组件健康明细


class ErrorResponse(BaseModel):
    detail: str
    error_code: str | None = None


class SuccessResponse(BaseModel):
    success: bool = True
    message: str = ""
