"""对话相关 Pydantic 模型"""

from pydantic import BaseModel, Field


class AgenticChatRequest(BaseModel):
    """流式 Agent 对话请求"""
    query: str = Field(..., min_length=1, description="用户问题")
    session_id: str = Field(default="default", description="会话 ID")
    enable_web_search: bool = False
    enable_reflection: bool = True
    enable_rerank: bool = True
    enable_grade_documents: bool = True
    enable_transform_query: bool = True
    enable_bm25: bool = True
    enable_multi_query: bool = False
    enable_kg: bool = True


class SourceDocument(BaseModel):
    """来源文档片段"""
    content: str
    metadata: dict = Field(default_factory=dict)
    score: float | None = None


class ChatHistoryMessage(BaseModel):
    """对话历史中单条消息"""
    role: str  # user / assistant
    content: str


class ChatHistoryResponse(BaseModel):
    """对话历史响应"""
    session_id: str
    messages: list[ChatHistoryMessage]
    total: int


class StreamEvent(BaseModel):
    """SSE 流式事件"""
    event: str  # token / source / done / error
    data: str
