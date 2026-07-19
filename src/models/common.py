"""通用响应模型"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


class ErrorResponse(BaseModel):
    detail: str
    error_code: str | None = None


class SuccessResponse(BaseModel):
    success: bool = True
    message: str = ""
