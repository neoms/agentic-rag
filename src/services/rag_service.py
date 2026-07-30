"""Agent/RAG 服务层 - 流式 Agent 编排与对话交互

职责：
- 组装 AgentState → 驱动 LangGraph 执行检索 → 外部流式生成 + 幻觉检测
- 生成（generate）和幻觉检测（check_hallucination）委托给独立模块
"""

import time
import asyncio
import logging
import json
from typing import Any, AsyncIterator

from langchain_core.messages import HumanMessage

from src.agent.graph import agent_graph
from src.agent.state import AgentState
from src.models.chat import (
    AgenticChatRequest,
    SourceDocument,
    ChatHistoryResponse,
    ChatHistoryMessage,
    StreamEvent,
)
from src.memory.manager import memory_manager
from src.backend.llm import create_strong_llm
from src.services.generator import (
    format_documents_with_citations,
    build_generate_prompt,
    build_generate_node_data,
    build_hallucination_node_data,
)
from src.services.hallucination_checker import check_hallucination_async

logger = logging.getLogger(__name__)


class RAGService:
    """RAG 服务（仅流式模式）"""

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
            "agent_path": [],
            "stream": True,
            "tool_calls": [],
            "enable_web_search": request.enable_web_search,
            "enable_reflection": request.enable_reflection,
            "enable_rerank": request.enable_rerank,
            "enable_grade_documents": request.enable_grade_documents,
            "enable_transform_query": request.enable_transform_query,
            "enable_bm25": request.enable_bm25,
            "enable_multi_query": request.enable_multi_query,
            "documents_semantic": [],
            "documents_bm25": [],
            "documents_multi_query": [],
            "enable_kg": True,  # 由意图分析自动决定是否实际启用
            "kg_intent": False,
            "kg_context": "",
        }

        # recursion_limit=50：KG 开启时 Send 分支 + 查询重写循环可能超过默认 25
        config = {"configurable": {"thread_id": request.session_id}, "recursion_limit": 50}
        GRAPH_NODES = {
            "retrieve", "rerank_documents", "grade_documents",
            "web_search", "transform_query", "tools",
            "analyze_kg_intent", "parallel_retrieve_merge",
        }
        # 线性节点后继（用于预测 node_start）
        LINEAR_NEXT: dict[str, str] = {
            "analyze_kg_intent": "parallel_retrieve_merge",
            "parallel_retrieve_merge": "rerank_documents",
        }

        # 每节点执行耗时追踪（毫秒）
        node_start_ts: dict[str, float] = {}
        node_timings: dict[str, float] = {}

        # 图入口第一个节点一开始就算活跃
        node_start_ts["analyze_kg_intent"] = time.perf_counter() * 1000
        yield StreamEvent(event="node_start", data="analyze_kg_intent")

        logger.info("[stream_rag] 开始执行 Agent 状态图（stream_mode=updates+custom）...")
        async for mode, chunk in agent_graph.astream(
            initial_state, config, stream_mode=["updates", "custom"],
        ):
            # ── 实时自定义事件（节点内 get_stream_writer() 推送） ──
            if mode == "custom":
                if isinstance(chunk, dict):
                    custom_event = chunk.get("event", "")
                    node_name = chunk.get("node", "")
                    if custom_event and node_name:
                        logger.info(
                            "[stream_rag] 自定义事件: %s → %s", node_name, custom_event,
                        )
                        # 记录策略子节点的耗时
                        if custom_event == "node_start":
                            node_start_ts[node_name] = time.perf_counter() * 1000
                        elif custom_event == "node_step" and node_name in node_start_ts:
                            node_timings[node_name] = round(
                                time.perf_counter() * 1000 - node_start_ts.pop(node_name), 1
                            )
                        yield StreamEvent(event=custom_event, data=node_name)
                continue

            # ── LangGraph 节点完成（updates 模式） ──
            if mode == "updates":
                for node_name in chunk:
                    if node_name not in GRAPH_NODES:
                        continue
                    logger.info("[stream_rag] 节点完成: %s", node_name)
                    # 记录图节点的耗时
                    if node_name in node_start_ts:
                        node_timings[node_name] = round(
                            time.perf_counter() * 1000 - node_start_ts.pop(node_name), 1
                        )
                    yield StreamEvent(event="node_step", data=node_name)

                    # 预测下一个线性节点（分支节点由后续 updates 自动发现）
                    next_node = LINEAR_NEXT.get(node_name)
                    if next_node:
                        node_start_ts[next_node] = time.perf_counter() * 1000
                        yield StreamEvent(event="node_start", data=next_node)
                    elif node_name == "rerank_documents" and request.enable_grade_documents:
                        node_start_ts["grade_documents"] = time.perf_counter() * 1000
                        yield StreamEvent(event="node_start", data="grade_documents")

        # 获取最终状态
        final_state = await asyncio.to_thread(agent_graph.get_state, config)
        result = final_state.values if final_state else {}
        logger.info("[stream_rag] 状态图完成, graph_elapsed=%.2fs", time.time() - t0)

        # 构建图内节点的 I/O 数据 + 处理 agent_path
        node_data: dict = self._build_node_data(result, request)

        # 将流中追踪的节点耗时合并到 node_data
        for node_id, duration_ms in node_timings.items():
            if node_id in node_data:
                node_data[node_id]["durationMs"] = duration_ms

        agent_path = result.get("agent_path", [])
        agent_path = [p for p in agent_path if "skipped" not in p]
        # 补充外部生成路径标记（generate/check_hallucination 已从图中移除）
        agent_path.append("generate")
        if request.enable_reflection:
            agent_path.append("check_hallucination")

        documents = result.get("documents", [])
        logger.info("[stream_rag] 状态图完成: path=%s, docs=%d, graph_elapsed=%.2fs",
                     agent_path, len(documents), time.time() - t0)

        # 发送检索结果来源
        sources = [
            SourceDocument(
                content=doc.page_content[:300],
                metadata=doc.metadata,
            )
            for doc in documents
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

        # 图执行完成，开始外部流式生成
        yield StreamEvent(event="node_start", data="generate")

        if documents:
            # 1. 构建带引文的文档上下文
            docs_text, citation_metadata = format_documents_with_citations(documents)

            # 2. 发送引文元数据
            if citation_metadata:
                yield StreamEvent(
                    event="citations",
                    data=json.dumps(citation_metadata, ensure_ascii=False),
                )

            # 3. 构建 prompt 并流式生成
            chat_history = memory_manager.get_chat_history_string(request.session_id)
            prompt = build_generate_prompt(request.query, docs_text, chat_history)
            llm = create_strong_llm(streaming=True)

            full_answer = ""
            async for chunk in llm.astream(prompt):
                if chunk.content:
                    full_answer += chunk.content
                    yield StreamEvent(event="token", data=chunk.content)

            # 标记 generate 已完成（让流程图在幻觉检测前就显示生成完成）
            yield StreamEvent(event="node_step", data="generate")

            # 4. 填充 generate 节点 I/O 数据
            node_data["generate"] = build_generate_node_data(
                request.query, documents, full_answer,
            )

            # 5. 幻觉检测（自反思开启时）
            hallucination_passed = True
            hallucination_faithfulness = 100.0
            if request.enable_reflection:
                yield StreamEvent(event="node_start", data="check_hallucination")
                hallucination_passed, hallucination_faithfulness = (
                    await check_hallucination_async(documents, full_answer)
                )
                yield StreamEvent(event="node_step", data="check_hallucination")
                yield StreamEvent(
                    event="hallucination",
                    data=json.dumps({
                        "passed": hallucination_passed,
                        "result": "PASSED" if hallucination_passed else "FAILED",
                        "faithfulness": hallucination_faithfulness,
                    }, ensure_ascii=False),
                )
                node_data["check_hallucination"] = build_hallucination_node_data(
                    full_answer, hallucination_faithfulness, hallucination_passed,
                )

            # 6. 记录对话
            memory_manager.add_interaction(request.session_id, request.query, full_answer)
            logger.info("[stream_rag] 流式生成完成: answer_len=%d", len(full_answer))
        else:
            msg = "未找到相关文档。"
            yield StreamEvent(event="token", data=msg)
            memory_manager.add_interaction(request.session_id, request.query, msg)
            logger.info("[stream_rag] 无文档，返回兜底回答")
            node_data["generate"] = build_generate_node_data(request.query, [], msg)

        # 发送节点 I/O 数据（前端流程图点击展示）
        yield StreamEvent(
            event="node_data",
            data=json.dumps(node_data, ensure_ascii=False),
        )
        yield StreamEvent(event="done", data="")
        logger.info("[stream_rag] 流式对话全部完成: elapsed=%.2fs", time.time() - t0)

    @staticmethod
    def _doc_detail(doc: Any) -> dict:
        """将单个 Document 提取为结构化详情（含完整 content 和 metadata）"""
        meta = doc.metadata
        return {
            "source": meta.get("url") or meta.get("filename", "未知来源"),
            "content_length": len(doc.page_content),
            "content": doc.page_content,
            "score": getattr(doc, "score", None),
            "metadata": {k: str(v) for k, v in meta.items()},
        }

    @staticmethod
    def _build_node_data(
        result: dict, request: AgenticChatRequest, documents: list | None = None
    ) -> dict:
        """从 final_state 提取各节点完整的输入/输出数据

        返回结构支持：
          - input / output: string | list | dict（前端根据类型自适应渲染）
          - durationMs: float（由调用方在流中追踪后注入）
        """
        docs = result.get("documents", []) if documents is None else documents
        query = request.query

        data: dict = {}

        # ── 意图分析 ──
        kg_intent = result.get("kg_intent", False)
        data["analyze_kg_intent"] = {
            "input": {
                "query": query,
            },
            "output": {
                "kg_intent": kg_intent,
                "explanation": (
                    "问题涉及实体关系查询，需要启用图谱检索"
                    if kg_intent
                    else "问题不涉及实体关系，走标准 RAG 流程"
                ),
            },
        }

        # ── 子策略 I/O（各策略使用独立结果） ──
        kg_context = result.get("kg_context", "")
        semantic_docs = result.get("documents_semantic", docs)
        bm25_docs = result.get("documents_bm25", [])
        mq_docs = result.get("documents_multi_query", [])

        data["retrieve"] = {
            "input": {
                "query": query,
                "method": "语义检索 + MMR（最大边际相关性）",
                "result_count": len(semantic_docs),
            },
            "output": {
                "total_documents": len(semantic_docs),
                "documents": [RAGService._doc_detail(d) for d in semantic_docs],
            },
        }
        if request.enable_bm25:
            data["bm25_retrieve"] = {
                "input": {
                    "query": query,
                    "method": "BM25 关键词检索",
                    "result_count": len(bm25_docs),
                },
                "output": {
                    "total_documents": len(bm25_docs),
                    "note": "BM25 独立检索结果（后续与语义检索合并去重）",
                    "documents": [RAGService._doc_detail(d) for d in bm25_docs],
                },
            }
        if request.enable_multi_query:
            data["multi_query_retrieve"] = {
                "input": {
                    "query": query,
                    "method": "多角度查询分解 + 语义检索",
                    "result_count": len(mq_docs),
                },
                "output": {
                    "total_documents": len(mq_docs),
                    "documents": [RAGService._doc_detail(d) for d in mq_docs],
                },
            }
        if kg_intent:
            data["kg_retrieve"] = {
                "input": {
                    "query": query,
                    "kg_intent": True,
                    "retrieval_method": "Kuzu 图 + LLM 实体抽取",
                },
                "output": {
                    "context_length": len(kg_context),
                    "has_result": bool(kg_context),
                    "context": kg_context if kg_context else "图谱检索无结果",
                },
            }

        # ── 并行检索合并 ──
        strategies = ["语义+MMR"]
        if request.enable_bm25: strategies.append("BM25")
        if request.enable_multi_query: strategies.append("多角度查询")
        if kg_intent: strategies.append("知识图谱")
        data["parallel_retrieve_merge"] = {
            "input": {
                "strategies_count": len(strategies),
                "strategies": strategies,
            },
            "output": {
                "total_documents": len(docs),
                "documents": [RAGService._doc_detail(d) for d in docs],
            },
        }

        # ── 重排序 ──
        if request.enable_rerank:
            data["rerank_documents"] = {
                "input": {
                    "method": "Cohere Rerank 重排序",
                    "documents_count": len(docs),
                    "documents": [RAGService._doc_detail(d) for d in docs],
                },
                "output": {
                    "reranked_count": len(docs),
                    "result": f"已按相关性对 {len(docs)} 份文档重新排序",
                    "documents": [RAGService._doc_detail(d) for d in docs],
                },
            }

        # ── 文档评估 ──
        if request.enable_grade_documents:
            relevant = result.get("documents_relevant", False)
            data["grade_documents"] = {
                "input": {
                    "documents_count": len(docs),
                    "documents": [RAGService._doc_detail(d) for d in docs],
                },
                "output": {
                    "all_relevant": relevant,
                    "verdict": "全部通过" if relevant else "部分/全部未通过",
                    "action": (
                        "文档均合格，进入生成阶段"
                        if relevant
                        else "不合格，触发查询重写或联网搜索降级"
                    ),
                },
            }

        # ── 查询重写 ──
        if request.enable_transform_query:
            rewritten_q = result.get("rewritten_query", query)
            iteration = result.get("iteration_count", 0)
            data["transform_query"] = {
                "input": {
                    "original_query": query,
                    "iteration": iteration,
                    "max_iterations": result.get("max_iterations", 3),
                },
                "output": {
                    "rewritten_query": rewritten_q,
                    "changed": rewritten_q != query,
                    "note": (
                        "查询已改写以优化检索效果"
                        if rewritten_q != query
                        else "查询未发生变化"
                    ),
                },
            }

        # ── 联网搜索 ──
        if request.enable_web_search:
            web_results = (
                [d for d in docs if d.metadata.get("source") == "web"]
                if docs
                else []
            )
            data["web_search"] = {
                "input": {
                    "query": query,
                    "total_web_results": len(web_results),
                },
                "output": {
                    "total_results": len(web_results),
                    "results": (
                        [RAGService._doc_detail(d) for d in web_results]
                        if web_results
                        else [{"note": "联网搜索结果已合并到主文档集"}]
                    ),
                },
            }

        return data

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
