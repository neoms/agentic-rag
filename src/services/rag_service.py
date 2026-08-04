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
from src.config.settings import settings
from src.models.chat import (
    AgenticChatRequest,
    SourceDocument,
    ChatHistoryResponse,
    ChatHistoryMessage,
    ChatSessionSummary,
    ChatSessionsResponse,
    StreamEvent,
)
from src.memory.manager import memory_manager
from src.cache import get_cache_service
from src.cache.service import build_config_signature
from src.metrics import (
    chat_requests_total,
    chat_cache_hit_total,
    chat_stream_duration_seconds,
    chat_ttft_seconds,
    chat_stage_duration_seconds,
    chat_cache_saved_llm_calls,
)
from src.eval.langfuse import (
    attach_request_span,
    build_graph_callback,
    create_trace_id,
)
from src.services.generator import (
    build_generate_node_data,
    build_hallucination_node_data,
)
from src.services.hallucination_checker import check_hallucination_async

logger = logging.getLogger(__name__)

FALLBACK_NO_DOCS = "未找到相关文档。"
FALLBACK_GENERATION_FAILED = "生成回答失败，请稍后重试。"


class RAGService:
    """RAG 服务（仅流式模式）"""

    async def agentic_rag_stream(
        self, request: AgenticChatRequest
    ) -> AsyncIterator[StreamEvent]:
        """Agent 模式流式 RAG（SSE）"""
        # 统一使用单调时钟（time.perf_counter）：与 first_token_at / 节点计时
        # 一致，避免 time.time()（墙上时钟）混用导致 TTFT 计算出巨大负值
        t0 = time.perf_counter()
        logger.info("[stream_rag] 请求: session=%s, query='%s', web_search=%s",
                     request.session_id, request.query[:80], request.enable_web_search)
        chat_requests_total.inc()

        # ── Langfuse 追踪（未配置时全链路优雅降级） ──
        trace_id = create_trace_id()
        langfuse_handler = build_graph_callback(trace_id)
        cache_type: str = "none"
        first_token_at: float | None = None  # TTFT 观测用（秒）

        def _observe_ttft() -> None:
            nonlocal first_token_at
            if first_token_at is not None:
                chat_ttft_seconds.observe(first_token_at - t0)

        # ── 多级缓存（精准 + 语义）虚拟节点：命中回放，未命中复用问题向量 ──
        # cache_lookup / cache_replay / cache_store 均为服务层虚拟节点
        # （沿用 check_hallucination 模式：本服务发 node_start/node_step/node_data 事件）
        query_embedding: list[float] | None = None
        cache_service = get_cache_service()
        cache_lookup_data: dict | None = None
        node_start_ts: dict[str, float] = {}
        node_timings: dict[str, float] = {}

        if request.use_cache:
            signature = build_config_signature(request)
            # ── cache_lookup 虚拟节点：计时 → 查询 → 完成 ──
            node_start_ts["cache_lookup"] = time.perf_counter() * 1000
            yield StreamEvent(event="node_start", data="cache_lookup")
            cache_entry, query_embedding, cache_info = cache_service.lookup(
                request.query, signature,
            )
            node_timings["cache_lookup"] = round(
                time.perf_counter() * 1000 - node_start_ts.pop("cache_lookup"), 1
            )
            cache_lookup_data = {
                "input": {
                    "query": request.query,
                    "normalized_query": cache_info.get("query_norm", ""),
                    "config_signature": signature,
                },
                "output": {
                    "hit": cache_entry is not None,
                    "cache_type": cache_info.get("cache_type", "none"),
                    "similarity": cache_info.get("similarity"),
                    "hit_count": cache_entry["hit_count"] if cache_entry else None,
                    # 前端拆分为「精准缓存」「语义缓存」两个节点所需的分层状态
                    "exact_checked": cache_info.get("exact_checked", False),
                    "exact_hit": cache_info.get("exact_hit", False),
                    "exact_ms": cache_info.get("exact_ms"),
                    "semantic_checked": cache_info.get("semantic_checked", False),
                    "semantic_hit": cache_info.get("semantic_hit", False),
                    "semantic_ms": cache_info.get("semantic_ms"),
                },
                "durationMs": node_timings["cache_lookup"],
            }
            yield StreamEvent(event="node_step", data="cache_lookup")

            if cache_entry is not None:
                # ── 命中：输出回放（cache_replay 虚拟节点） ──
                logger.info("[stream_rag] 缓存命中: session=%s, query='%s', type=%s",
                            request.session_id, request.query[:80],
                            cache_info.get("cache_type", "exact"))
                chat_cache_hit_total.labels(
                    cache_info.get("cache_type", "exact")
                ).inc()
                cache_type = cache_info.get("cache_type", "exact")
                chat_cache_saved_llm_calls.inc()
                cache_path = ["cache_lookup", "cache_replay"]
                node_start_ts["cache_replay"] = time.perf_counter() * 1000
                yield StreamEvent(event="node_start", data="cache_replay")
                for ev in cache_service.replay(cache_entry, cache_path):
                    if ev.event == "token" and first_token_at is None:
                        first_token_at = time.perf_counter()
                    yield ev
                node_timings["cache_replay"] = round(
                    time.perf_counter() * 1000 - node_start_ts.pop("cache_replay"), 1
                )
                # 语义命中后把当前问法写回精准缓存：同文本下次提问可直接精准命中。
                # 仅当问法与缓存问法高度一致时才写回，防止"焦点不同"的问法
                # 把错误答案固化并扩散到精准缓存。
                if (
                    cache_info.get("cache_type") == "semantic"
                    and query_embedding
                    and cache_service.should_promote_to_exact(
                        request.query, cache_entry["query"]
                    )
                ):
                    try:
                        cache_service.store(
                            query=request.query,
                            signature=signature,
                            vector=query_embedding,
                            answer=cache_entry["answer"],
                            sources=cache_entry["sources"],
                            agent_path=cache_entry["agent_path"],
                            citations=cache_entry["citations"],
                            hallucination=cache_entry["hallucination"],
                        )
                    except Exception as e:
                        logger.warning("[stream_rag] 语义命中写回失败: %s", e)
                elif cache_info.get("cache_type") == "semantic":
                    logger.info(
                        "[stream_rag] 语义命中但问法差异较大，跳过精准写回: query='%s'",
                        request.query[:60],
                    )
                node_data = {
                    "cache_lookup": cache_lookup_data,
                    "cache_replay": {
                        "input": {
                            "query": request.query,
                        },
                        "output": {
                            "answer_length": len(cache_entry["answer"]),
                            "hit_count": cache_entry["hit_count"],
                            "original_path": cache_entry["agent_path"],
                        },
                        "durationMs": node_timings["cache_replay"],
                    },
                }
                yield StreamEvent(
                    event="node_data",
                    data=json.dumps(node_data, ensure_ascii=False),
                )
                yield StreamEvent(event="node_step", data="cache_replay")
                _observe_ttft()
                for node_id, duration_ms in node_timings.items():
                    chat_stage_duration_seconds.labels(stage=node_id).observe(
                        duration_ms / 1000.0
                    )
                attach_request_span(
                    trace_id,
                    input_data={"query": request.query},
                    output_data={
                        "answer": cache_entry["answer"],
                        "sources": cache_entry.get("sources", []),
                        "latency_seconds": round(time.perf_counter() - t0, 3),
                        "cache_type": cache_type,
                    },
                    metadata={"use_cache": request.use_cache},
                )
                yield StreamEvent(
                    event="done",
                    data=json.dumps({"trace_id": trace_id or ""}, ensure_ascii=False),
                )
                memory_manager.add_interaction(
                    request.session_id, request.query, cache_entry["answer"],
                )
                # 缓存条目中存有幻觉结果时一并恢复
                cached_h = cache_entry.get("hallucination")
                if isinstance(cached_h, dict):
                    memory_manager.add_hallucination_result(
                        request.session_id,
                        cached_h.get("passed", True),
                        cached_h.get("faithfulness", 100.0),
                    )
                logger.info("[stream_rag] 缓存回放完成: answer_len=%d, elapsed=%.2fs",
                            len(cache_entry["answer"]), time.perf_counter() - t0)
                chat_stream_duration_seconds.observe(time.perf_counter() - t0)
                return

        initial_state: AgentState = {
            "query": request.query,
            "query_embedding": query_embedding,
            "session_id": request.session_id,
            "messages": [],
            "documents": [],
            "rewritten_query": "",
            "documents_relevant": False,
            "iteration_count": 0,
            "max_iterations": 3,
            "rerank_top_score": 0.0,
            "best_rerank_score": 0.0,
            "rerank_improved": True,
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
            "strategy_timings_ms": {},
            "enable_kg": request.enable_kg,  # 由意图分析自动决定是否实际启用
            "kg_intent": False,
            "kg_context": "",
            "complexity": "SIMPLE",
            "answer": "",
            "citation_metadata": {},
        }

        # recursion_limit=50：KG 开启时 Send 分支 + 查询重写循环可能超过默认 25
        config: dict = {
            "configurable": {"thread_id": request.session_id},
            "recursion_limit": 50,
        }
        if langfuse_handler is not None and trace_id:
            config["callbacks"] = [langfuse_handler]
            config["metadata"] = {
                "langfuse_trace_name": "agentic-rag-chat",
                "langfuse_session_id": request.session_id,
                "langfuse_tags": ["chat", "agentic-rag"],
                "query": request.query[:2000],
                "cache_type": cache_type,
                "use_cache": request.use_cache,
            }
        GRAPH_NODES = {
            "retrieve", "rerank_documents", "grade_documents",
            "web_search", "transform_query", "tools",
            "analyze_kg_intent", "parallel_retrieve_merge",
            "judge_complexity", "generate_simple", "generate_complex",
        }
        # 线性节点后继（用于预测 node_start）
        # 绝大多数节点已通过条件边或节点内部自定义事件（node_start）自动激活，
        # 此处仅保留合并节点的默认后继
        LINEAR_NEXT: dict[str, str] = {
            "parallel_retrieve_merge": "rerank_documents",
        }

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

                    # 流式 token（来自 generate_simple/complex 节点）
                    if custom_event == "token":
                        content = chunk.get("content", "")
                        if content:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            yield StreamEvent(event="token", data=content)
                        continue

                    # 引文元数据（来自 generate_simple/complex 节点）
                    if custom_event == "citations":
                        data = chunk.get("data", {})
                        if data:
                            yield StreamEvent(
                                event="citations",
                                data=json.dumps(data, ensure_ascii=False),
                            )
                        continue

                    # 节点生命周期事件（node_start / node_step）
                    node_name = chunk.get("node", "")
                    if custom_event and node_name:
                        logger.info(
                            "[stream_rag] 自定义事件: %s → %s", node_name, custom_event,
                        )
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
        logger.info("[stream_rag] 状态图完成, graph_elapsed=%.2fs", time.perf_counter() - t0)

        # 构建图内节点的 I/O 数据 + 处理 agent_path
        node_data: dict = self._build_node_data(result, request)

        # 将流中追踪的节点耗时合并到 node_data
        for node_id, duration_ms in node_timings.items():
            if node_id in node_data:
                node_data[node_id]["durationMs"] = duration_ms

        # 将策略级独立耗时注入 node_data（从 parallel_retrieve_merge_node 返回的 state）
        strategy_timings = result.get("strategy_timings_ms", {})
        for node_id, duration_ms in strategy_timings.items():
            if node_id in node_data:
                node_data[node_id]["durationMs"] = duration_ms

        agent_path = result.get("agent_path", [])
        agent_path = [p for p in agent_path if "skipped" not in p]
        # check_hallucination 仍在图外，由 rag_service 补充
        if request.enable_reflection:
            agent_path.append("check_hallucination")

        documents = result.get("documents", [])
        logger.info("[stream_rag] 状态图完成: path=%s, docs=%d, graph_elapsed=%.2fs",
                     agent_path, len(documents), time.perf_counter() - t0)

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

        # 从 state 读取图内生成结果（generate_simple/complex 已产出 answer）
        answer = result.get("answer", "")
        has_docs = bool(documents)
        hallucination_passed = True
        hallucination_faithfulness = 100.0

        # 确定实际执行的生成节点 ID
        generate_node_id = "generate_simple"
        for p in agent_path:
            if p.startswith("generate_"):
                generate_node_id = p
                break

        if has_docs:
            # 检索到文档但生成结果为空 → 明确失败提示（不允许"有文档却空答"）
            answer_is_fallback = False
            if not answer:
                answer_is_fallback = True
                logger.error(
                    "[stream_rag] 已检索到 %d 篇文档但生成结果为空，返回失败提示",
                    len(documents),
                )
                answer = FALLBACK_GENERATION_FAILED
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                yield StreamEvent(event="token", data=answer)
            # 填充图内生成节点的 I/O 数据
            node_data[generate_node_id] = build_generate_node_data(
                request.query, documents, answer,
            )
            # 注入生成节点耗时（node_timings 在 astream 循环中已记录，但
            # generate node_data 此时才创建，错过第 171 行的 merge 循环）
            if generate_node_id in node_timings:
                node_data[generate_node_id]["durationMs"] = node_timings[generate_node_id]

            # 幻觉检测（自反思开启且非兜底/失败回答时）
            if request.enable_reflection and not answer_is_fallback:
                node_start_ts["check_hallucination"] = time.perf_counter() * 1000
                yield StreamEvent(event="node_start", data="check_hallucination")
                hallucination_passed, hallucination_faithfulness = (
                    await check_hallucination_async(
                        documents,
                        answer,
                        citation_metadata=result.get("citation_metadata", {}),
                    )
                )
                if "check_hallucination" in node_start_ts:
                    node_timings["check_hallucination"] = round(
                        time.perf_counter() * 1000 - node_start_ts.pop("check_hallucination"), 1
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
                    answer, hallucination_faithfulness, hallucination_passed,
                )
                if "check_hallucination" in node_timings:
                    node_data["check_hallucination"]["durationMs"] = node_timings["check_hallucination"]

            # 记录对话
            memory_manager.add_interaction(request.session_id, request.query, answer)
            if request.enable_reflection:
                memory_manager.add_hallucination_result(
                    request.session_id, hallucination_passed, hallucination_faithfulness,
                )
            logger.info("[stream_rag] 流式生成完成: answer_len=%d, node=%s",
                         len(answer), generate_node_id)
        else:
            # 无检索文档 → 降级兜底
            if not answer:
                answer = FALLBACK_NO_DOCS
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                yield StreamEvent(event="token", data=answer)
            node_data[generate_node_id] = build_generate_node_data(request.query, [], answer)
            if generate_node_id in node_timings:
                node_data[generate_node_id]["durationMs"] = node_timings[generate_node_id]
            memory_manager.add_interaction(request.session_id, request.query, answer)
            logger.info("[stream_rag] 无文档，返回兜底回答")

        # ── cache_store 虚拟节点：写回多级缓存 ──
        # 反射开启时仅幻觉检测通过才写入；答案为空/兜底/缓存关闭时不写
        if request.use_cache:
            signature = build_config_signature(request)
            node_start_ts["cache_store"] = time.perf_counter() * 1000
            yield StreamEvent(event="node_start", data="cache_store")
            store_result: dict = {"written": False, "reason": ""}
            if not settings.cache_enabled:
                store_result["reason"] = "缓存功能已关闭"
            elif not answer or answer in (FALLBACK_NO_DOCS, FALLBACK_GENERATION_FAILED):
                store_result["reason"] = "答案为空或为兜底/失败回答，不缓存"
            elif request.enable_reflection and not hallucination_passed:
                store_result["reason"] = "幻觉检测未通过，不缓存"
            else:
                try:
                    get_cache_service().store(
                        query=request.query,
                        signature=signature,
                        vector=query_embedding,
                        answer=answer,
                        sources=[s.model_dump() for s in sources],
                        agent_path=agent_path,
                        citations=result.get("citation_metadata", {}),
                        hallucination=(
                            {
                                "passed": hallucination_passed,
                                "result": "PASSED" if hallucination_passed else "FAILED",
                                "faithfulness": hallucination_faithfulness,
                            }
                            if request.enable_reflection
                            else None
                        ),
                    )
                    store_result["written"] = True
                    store_result["reason"] = "已写入缓存"
                except Exception as e:
                    logger.warning("[stream_rag] 缓存写回失败: %s", e)
                    store_result["reason"] = f"写回异常: {e}"
            node_timings["cache_store"] = round(
                time.perf_counter() * 1000 - node_start_ts.pop("cache_store"), 1
            )
            node_data["cache_store"] = {
                "input": {
                    "query": request.query,
                    "config_signature": signature,
                },
                "output": store_result,
                "durationMs": node_timings["cache_store"],
            }
            yield StreamEvent(event="node_step", data="cache_store")
            agent_path.append("cache_store")

        # 合并 cache_lookup 虚拟节点数据
        if cache_lookup_data:
            node_data["cache_lookup"] = cache_lookup_data

        # 发送 Agent 路径（含缓存节点；在 cache_store 完成后发出，保证路径一致）
        yield StreamEvent(
            event="path",
            data=json.dumps(agent_path, ensure_ascii=False),
        )

        # 发送节点 I/O 数据（前端流程图点击展示）
        yield StreamEvent(
            event="node_data",
            data=json.dumps(node_data, ensure_ascii=False),
        )
        _observe_ttft()
        for node_id, duration_ms in node_timings.items():
            chat_stage_duration_seconds.labels(stage=node_id).observe(
                duration_ms / 1000.0
            )
        attach_request_span(
            trace_id,
            input_data={"query": request.query},
            output_data={
                "answer": answer,
                "sources": [s.model_dump() for s in sources],
                "latency_seconds": round(time.perf_counter() - t0, 3),
                "cache_type": cache_type,
            },
            metadata={"use_cache": request.use_cache},
        )
        yield StreamEvent(
            event="done",
            data=json.dumps({"trace_id": trace_id or ""}, ensure_ascii=False),
        )
        logger.info("[stream_rag] 流式对话全部完成: elapsed=%.2fs", time.perf_counter() - t0)
        chat_stream_duration_seconds.observe(time.perf_counter() - t0)

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
            rerank_output: dict = {
                "reranked_count": len(docs),
                "result": f"已按相关性对 {len(docs)} 份文档重新排序",
                "documents": [RAGService._doc_detail(d) for d in docs],
            }
            degraded = result.get("rerank_degraded")
            if degraded:
                rerank_output["degraded"] = True
                rerank_output["degraded_reason"] = degraded
                rerank_output["result"] = (
                    f"重排序接口异常，已降级为原始排序（{degraded}）"
                )
            data["rerank_documents"] = {
                "input": {
                    "method": "百炼 TextReRank 重排序",
                    "documents_count": len(docs),
                    "documents": [RAGService._doc_detail(d) for d in docs],
                },
                "output": rerank_output,
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

        # ── 复杂度判定 ──
        complexity = result.get("complexity", "SIMPLE")
        data["judge_complexity"] = {
            "input": {
                "query": query,
                "documents_count": len(docs),
                "doc_previews": [
                    d.page_content[:200] for d in docs[:3]
                ],
            },
            "output": {
                "complexity": complexity,
                "verdict": "SIMPLE" if complexity == "SIMPLE" else "COMPLEX",
                "action": (
                    f"问题较简单，使用快速模型 {settings.llm_model_fast} 生成"
                    if complexity == "SIMPLE"
                    else f"问题需多步推理，使用强模型 {settings.llm_model_strong} 生成"
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
                ChatHistoryMessage(
                    role=m["role"],
                    content=m["content"],
                    hallucination=m.get("hallucination"),
                )
                for m in messages
            ],
            total=len(messages),
        )

    def list_sessions(self) -> ChatSessionsResponse:
        """获取全部会话摘要（按最近活跃时间倒序）"""
        sessions = memory_manager.list_sessions()
        return ChatSessionsResponse(
            sessions=[ChatSessionSummary(**s) for s in sessions],
            total=len(sessions),
        )

    def clear_history(self, session_id: str) -> None:
        """删除会话历史（内存 + 持久化存储）"""
        logger.info("[clear_history] session=%s", session_id)
        memory_manager.clear(session_id)


# 全局单例
rag_service = RAGService()
