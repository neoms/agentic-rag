"""对话相关 Pydantic 模型"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """基础对话请求"""
    query: str = Field(..., min_length=1, description="用户问题")
    session_id: str = Field(default="default", description="会话 ID")
    top_k: int | None = Field(default=None, ge=1, le=20, description="检索数量")


class AgenticChatRequest(ChatRequest):
    """Agent 模式对话请求（含自反思、工具调用）"""
    enable_web_search: bool = False
    enable_reflection: bool = True
    stream: bool = False


class SourceDocument(BaseModel):
    """来源文档片段"""
    content: str
    metadata: dict = Field(default_factory=dict)
    score: float | None = None


class ChatResponse(BaseModel):
    """基础对话响应"""
    answer: str
    session_id: str
    sources: list[SourceDocument] = Field(default_factory=list)
    reflection_count: int = 0  # 自反思重试次数


class AgenticChatResponse(ChatResponse):
    """Agent 模式对话响应"""
    tool_calls: list[dict] = Field(default_factory=list)
    agent_path: list[str] = Field(default_factory=list)  # Agent 节点流转路径


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
