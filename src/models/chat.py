"""对话相关 Pydantic 模型"""

from pydantic import BaseModel, Field


class AgenticChatRequest(BaseModel):
    """流式 Agent 对话请求"""
    query: str = Field(..., min_length=1, description="用户问题")
    session_id: str = Field(
        default="default",
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="会话 ID（1-64 位字母/数字/下划线/连字符）",
    )
    use_cache: bool = True  # 是否启用多级缓存（评估/调试时可绕过）
    enable_web_search: bool = False
    enable_reflection: bool = True
    enable_rerank: bool = True
    enable_grade_documents: bool = True
    # 查询重写默认关闭（在查询策略中手动打开才启用，且最多重写 1 次）
    enable_transform_query: bool = False
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
    hallucination: dict | None = None  # 幻觉检测结果 {passed, faithfulness}，assistant 消息可带


class ChatHistoryResponse(BaseModel):
    """对话历史响应"""
    session_id: str
    messages: list[ChatHistoryMessage]
    total: int


class ChatSessionSummary(BaseModel):
    """会话摘要（侧边栏列表用）"""
    session_id: str
    preview: str = ""
    message_count: int = 0
    updated_at: float = 0.0  # epoch 秒（最近一条消息时间）


class ChatSessionsResponse(BaseModel):
    """会话列表响应"""
    sessions: list[ChatSessionSummary]
    total: int


class StreamEvent(BaseModel):
    """SSE 流式事件"""
    event: str  # token / source / done / error
    data: str


class FeedbackRequest(BaseModel):
    """用户对话反馈（写回 Langfuse trace）"""
    trace_id: str = Field(..., min_length=1, max_length=64, description="Langfuse trace id")
    rating: int = Field(..., ge=1, le=5, description="评分 1-5（👎=1，👍=5）")
    comment: str | None = Field(default=None, max_length=500, description="可选意见")
