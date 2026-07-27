"""对话 API 路由 - Agent 流式 RAG 对话"""

import logging
import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_rag_service
from src.services.rag_service import RAGService
from src.models.chat import (
    AgenticChatRequest,
    ChatHistoryResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "/stream",
    summary="Agent 模式流式对话",
    description="SSE 流式输出，事件类型：token（文本令牌）、source（来源文档）、path（Agent 路径）、done（完成）",
)
async def stream_chat(
    request: AgenticChatRequest,
    service: RAGService = Depends(get_rag_service),
):
    logger.info("API 请求: POST /chat/stream, session=%s, query='%s', web_search=%s",
                request.session_id, request.query[:80], request.enable_web_search)

    async def event_generator():
        try:
            async for event in service.agentic_rag_stream(request):
                # 手动构造 SSE 格式确保即时发送
                line = f"event: {event.event}\ndata: {event.data}\n\n"
                yield line.encode("utf-8")
        except Exception as e:
            logger.exception("流式对话失败")
            err_line = f"event: error\ndata: {json.dumps({'detail': str(e)}, ensure_ascii=False)}\n\n"
            yield err_line.encode("utf-8")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        },
    )


@router.get(
    "/history/{session_id}",
    response_model=ChatHistoryResponse,
    summary="获取会话历史",
)
async def get_history(
    session_id: str,
    service: RAGService = Depends(get_rag_service),
):
    logger.info("API 请求: GET /chat/history/%s", session_id)
    return service.get_history(session_id)
