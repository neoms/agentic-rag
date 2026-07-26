"""对话 API 路由 - Agent 流式 RAG 对话"""

import logging
import json
from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

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
    responses={
        200: {
            "description": "SSE 流式事件",
            "content": {"text/event-stream": {}},
        }
    },
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
                yield {
                    "event": event.event,
                    "data": event.data,
                }
        except Exception as e:
            logger.exception("流式对话失败")
            yield {
                "event": "error",
                "data": json.dumps({"detail": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


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
