"""对话 API 路由 - 支持基础 RAG、Agent 模式和 SSE 流式输出"""

import logging
import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from src.api.dependencies import get_rag_service
from src.services.rag_service import RAGService
from src.models.chat import (
    ChatRequest,
    AgenticChatRequest,
    ChatResponse,
    AgenticChatResponse,
    ChatHistoryResponse,
    StreamEvent,
)
from src.models.common import ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "/simple",
    response_model=ChatResponse,
    summary="基础 RAG 对话",
    description="检索 + 生成，不涉及 Agent 自反思等高级能力，适合简单问答场景",
)
async def simple_chat(
    request: ChatRequest,
    service: RAGService = Depends(get_rag_service),
):
    logger.info("API 请求: POST /chat/simple, session=%s, query='%s'", request.session_id, request.query[:80])
    try:
        response = service.simple_rag(request)
        logger.info("API 响应: POST /chat/simple → answer_len=%d, sources=%d", len(response.answer), len(response.sources))
        return response
    except Exception as e:
        logger.exception("简单 RAG 对话失败")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/agentic",
    response_model=AgenticChatResponse,
    summary="Agent 模式对话",
    description="完整的 Agent 流程：检索 → 相关性评估 → 查询重写 → 生成 → 幻觉检测，支持工具调用",
)
async def agentic_chat(
    request: AgenticChatRequest,
    service: RAGService = Depends(get_rag_service),
):
    logger.info("API 请求: POST /chat/agentic, session=%s, query='%s', web_search=%s",
                request.session_id, request.query[:80], request.enable_web_search)
    try:
        response = service.agentic_rag(request)
        logger.info("API 响应: POST /chat/agentic → answer_len=%d, path=%s, iterations=%d",
                     len(response.answer), response.agent_path, response.reflection_count)
        return response
    except Exception as e:
        logger.exception("Agent 对话失败")
        raise HTTPException(status_code=500, detail=str(e))


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
