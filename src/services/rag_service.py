"""Agent/RAG 服务层 - 对话交互、Agent 编排的业务逻辑"""

import time
import logging
import json
from typing import AsyncIterator

from src.agent.graph import agent_graph
from src.agent.state import AgentState
from src.models.chat import (
    ChatRequest,
    AgenticChatRequest,
    ChatResponse,
    AgenticChatResponse,
    SourceDocument,
    ChatHistoryResponse,
    ChatHistoryMessage,
    StreamEvent,
)
from src.store.vector_store import vector_store
from src.memory.manager import memory_manager
from src.backend.llm import create_strong_llm

logger = logging.getLogger(__name__)


class RAGService:
    """RAG 服务"""

    def simple_rag(self, request: ChatRequest) -> ChatResponse:
        """简单 RAG：检索 + 生成，不走 Agent 图"""
        t0 = time.time()
        query = request.query
        session_id = request.session_id
        top_k = request.top_k
        logger.info("[simple_rag] 请求: session=%s, query='%s', top_k=%s", session_id, query[:80], top_k)

        # 检索
        results = vector_store.search(query, top_k=top_k)
        documents = [doc for doc, _ in results]
        sources = [
            SourceDocument(
                content=doc.page_content,
                metadata=doc.metadata,
                score=score,
            )
            for doc, score in results
        ]
        logger.info("[simple_rag] 检索完成: %d 个结果", len(sources))

        # 生成答案
        if not documents:
            answer = "未找到相关文档，请尝试上传相关文档或换个问题。"
            logger.info("[simple_rag] 无文档, 返回兜底回答")
        else:
            docs_text = "\n\n---\n\n".join(
                f"来源: {doc.metadata.get('filename', 'unknown')}\n内容: {doc.page_content}"
                for doc in documents[:5]
            )
            chat_history = memory_manager.get_chat_history_string(session_id)

            prompt = f"""你是一个专业的知识问答助手。请基于提供的文档上下文回答用户问题。

规则：
1. 优先使用提供的文档信息回答
2. 如果文档信息不足以回答问题，请明确说明
3. 回答要简洁、准确、有条理
4. 使用中文回答

文档上下文：
{docs_text}

对话历史：
{chat_history or '无'}

用户问题：{query}

请回答："""

            llm = create_strong_llm()
            result = llm.invoke(prompt)
            answer = result.content.strip()

        # 记录对话
        memory_manager.add_interaction(session_id, query, answer)

        elapsed = time.time() - t0
        logger.info("[simple_rag] 完成: answer_len=%d, sources=%d, elapsed=%.2fs", len(answer), len(sources), elapsed)

        return ChatResponse(
            answer=answer,
            session_id=session_id,
            sources=sources,
        )

    def agentic_rag(self, request: AgenticChatRequest) -> AgenticChatResponse:
        """Agent 模式 RAG：走完整的 LangGraph 状态图（含自反思、查询重写、幻觉检测等）"""
        t0 = time.time()
        logger.info("[agentic_rag] 请求: session=%s, query='%s', web_search=%s, reflection=%s",
                     request.session_id, request.query[:80], request.enable_web_search, request.enable_reflection)

        initial_state: AgentState = {
            "query": request.query,
            "session_id": request.session_id,
            "messages": [],
            "documents": [],
            "rewritten_query": "",
            "documents_relevant": False,
            "iteration_count": 0,
            "max_iterations": 3,
            "answer": "",
            "hallucination_detected": False,
            "agent_path": [],
            "stream": request.stream,
            "tool_calls": [],
            "enable_web_search": request.enable_web_search,
        }

        # 运行 Agent 图
        config = {"configurable": {"thread_id": request.session_id}}
        result = agent_graph.invoke(initial_state, config)

        # 提取结果
        answer = result.get("answer", "Agent 处理完成，但未生成回答。")
        agent_path = result.get("agent_path", [])
        tool_calls_raw = result.get("tool_calls", [])
        iteration_count = result.get("iteration_count", 0)
        documents = result.get("documents", [])

        # 构建来源信息
        sources = [
            SourceDocument(
                content=doc.page_content,
                metadata=doc.metadata,
            )
            for doc in documents[:5]
        ]

        # 记录对话
        memory_manager.add_interaction(request.session_id, request.query, answer)

        elapsed = time.time() - t0
        logger.info("[agentic_rag] 完成: answer_len=%d, path=%s, iterations=%d, docs=%d, elapsed=%.2fs",
                     len(answer), agent_path, iteration_count, len(documents), elapsed)

        return AgenticChatResponse(
            answer=answer,
            session_id=request.session_id,
            sources=sources,
            reflection_count=iteration_count,
            tool_calls=tool_calls_raw,
            agent_path=agent_path,
        )

    async def agentic_rag_stream(
        self, request: AgenticChatRequest
    ) -> AsyncIterator[StreamEvent]:
        """Agent 模式流式 RAG（SSE）"""
        t0 = time.time()
        logger.info("[stream_rag] 请求: session=%s, query='%s', web_search=%s",
                     request.session_id, request.query[:80], request.enable_web_search)

        initial_state: AgentState = {
            "query": request.query,
            "session_id": request.session_id,
            "messages": [],
            "documents": [],
            "rewritten_query": "",
            "documents_relevant": False,
            "iteration_count": 0,
            "max_iterations": 3,
            "answer": "",
            "hallucination_detected": False,
            "agent_path": [],
            "stream": True,
            "tool_calls": [],
            "enable_web_search": request.enable_web_search,
        }

        # 先跑状态图获取检索结果和 Agent 路径
        config = {"configurable": {"thread_id": request.session_id}}
        logger.info("[stream_rag] 开始执行 Agent 状态图...")
        result = agent_graph.invoke(initial_state, config)

        agent_path = result.get("agent_path", [])
        documents = result.get("documents", [])
        logger.info("[stream_rag] 状态图完成: path=%s, docs=%d, graph_elapsed=%.2fs",
                     agent_path, len(documents), time.time() - t0)

        # 发送检索结果来源
        sources = [
            SourceDocument(
                content=doc.page_content[:300],
                metadata=doc.metadata,
            )
            for doc in documents[:3]
        ]
        yield StreamEvent(
            event="source",
            data=json.dumps([s.model_dump() for s in sources], ensure_ascii=False),
        )

        # 发送 Agent 路径
        yield StreamEvent(
            event="path",
            data=json.dumps(agent_path, ensure_ascii=False),
        )

        # 流式生成答案
        if documents:
            doc_parts: list[str] = []
            for doc in documents[:5]:
                src = doc.metadata.get("url") or doc.metadata.get("filename", "unknown")
                url_info = f"\n链接: {doc.metadata['url']}" if doc.metadata.get("url") else ""
                doc_parts.append(f"来源: {src}{url_info}\n内容: {doc.page_content}")
            docs_text = "\n\n---\n\n".join(doc_parts)
            chat_history = memory_manager.get_chat_history_string(request.session_id)

            prompt = f"""你是一个专业的知识问答助手。请基于提供的文档上下文回答用户问题。

规则：
1. 优先使用提供的文档信息回答
2. 如果文档信息不足以回答问题，请明确说明
3. 回答要简洁、准确、有条理
4. 使用中文回答

文档上下文：
{docs_text}

对话历史：
{chat_history or '无'}

用户问题：{request.query}

请回答："""

            llm = create_strong_llm(streaming=True)
            full_answer = ""
            for chunk in llm.stream(prompt):
                if chunk.content:
                    full_answer += chunk.content
                    yield StreamEvent(event="token", data=chunk.content)

            # 记录对话
            memory_manager.add_interaction(request.session_id, request.query, full_answer)
            logger.info("[stream_rag] 流式生成完成: answer_len=%d", len(full_answer))
        else:
            msg = "未找到相关文档。"
            yield StreamEvent(event="token", data=msg)
            memory_manager.add_interaction(request.session_id, request.query, msg)
            logger.info("[stream_rag] 无文档，返回兜底回答")

        # 完成
        yield StreamEvent(event="done", data="")
        logger.info("[stream_rag] 流式对话全部完成: elapsed=%.2fs", time.time() - t0)

    def get_history(self, session_id: str) -> ChatHistoryResponse:
        """获取会话历史"""
        logger.info("[get_history] session=%s", session_id)
        messages = memory_manager.get_history(session_id)
        logger.info("[get_history] 返回 %d 条消息", len(messages))
        return ChatHistoryResponse(
            session_id=session_id,
            messages=[
                ChatHistoryMessage(role=m["role"], content=m["content"])
                for m in messages
            ],
            total=len(messages),
        )


# 全局单例
rag_service = RAGService()
