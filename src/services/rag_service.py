"""Agent/RAG 服务层 - 流式 Agent 编排与对话交互

职责：
- 组装 AgentState → 驱动 LangGraph 执行检索 → 外部流式生成 + 幻觉检测
- 生成（generate）和幻觉检测（check_hallucination）委托给独立模块
"""

import time
import asyncio
import logging
import json
from typing import AsyncIterator

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
            "enable_hyde": request.enable_hyde,
            "enable_multi_query": request.enable_multi_query,
            "documents_bm25": [],
            "documents_hyde": [],
            "documents_multi_query": [],
            "enable_kg": request.enable_kg,
            "kg_intent": False,
            "kg_context": "",
        }

        # recursion_limit=50：KG 开启时 Send 分支 + 查询重写循环可能超过默认 25
        config = {"configurable": {"thread_id": request.session_id}, "recursion_limit": 50}
        GRAPH_NODES = {
            "retrieve", "rerank_documents", "grade_documents",
            "web_search", "transform_query", "tools",
            "bm25_retrieve", "hyde_retrieve", "multi_query_retrieve", "merge_retrieval",
            "analyze_kg_intent", "kg_retrieve",
        }
        logger.info("[stream_rag] 开始执行 Agent 状态图（逐步模式）...")
        async for evt in agent_graph.astream_events(initial_state, config, version="v2"):
            name = evt.get("name", "")
            kind = evt.get("event", "")
            if name not in GRAPH_NODES:
                continue
            if kind == "on_chain_start":
                logger.info("[stream_rag] 节点开始: %s", name)
                yield StreamEvent(event="node_start", data=name)
            elif kind == "on_chain_end":
                logger.info("[stream_rag] 节点完成: %s", name)
                yield StreamEvent(event="node_step", data=name)

        # 获取最终状态
        final_state = await asyncio.to_thread(agent_graph.get_state, config)
        result = final_state.values if final_state else {}
        logger.info("[stream_rag] 状态图完成, graph_elapsed=%.2fs", time.time() - t0)

        # 构建图内节点的 I/O 数据 + 处理 agent_path
        node_data: dict[str, dict[str, str | list[str]]] = self._build_node_data(result, request)

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
    def _doc_items(doc_list: list, max_docs: int = 8) -> list[str]:
        """生成文档列表项（每项含来源和内容预览）"""
        items = []
        for i, doc in enumerate(doc_list[:max_docs]):
            src = doc.metadata.get("url") or doc.metadata.get("filename", f"文档{i+1}")
            preview = doc.page_content[:100].replace("\n", " ")
            items.append(f"来源: {src}\n{preview}...")
        if len(doc_list) > max_docs:
            items.append(f"... 及其他 {len(doc_list) - max_docs} 条")
        return items

    @staticmethod
    def _build_node_data(
        result: dict, request: AgenticChatRequest, documents: list | None = None
    ) -> dict[str, dict[str, str | list[str]]]:
        """从 final_state 提取各图内节点的输入/输出数据（含具体内容列表）"""
        docs = result.get("documents", []) if documents is None else documents
        query = request.query
        kg_context = result.get("kg_context", "")

        data: dict[str, dict[str, str | list[str]]] = {}

        # ── 意图分析 ──
        # kg_intent 在最终状态中恒为 False（kg_retrieve_node 会重置它用于防循环）。
        # 因此通过 agent_path 判断 KG 是否实际执行过。
        agent_path = result.get("agent_path", [])
        kg_actually_ran = any(
            p == "kg_retrieve" for p in agent_path
        )
        kg_intent = result.get('kg_intent', False) or kg_actually_ran
        data["analyze_kg_intent"] = {
            "input": f"用户问题: {query}",
            "output": [
                f"KG意图判定: {'是' if kg_intent else '否'}",
                f"说明: {'问题涉及实体关系查询，需要启用图谱检索' if kg_intent else '问题不涉及实体关系，走标准RAG流程'}",
            ],
        }

        # ── 检索节点 ──
        rewritten = result.get("rewritten_query", "") or query
        data["retrieve"] = {
            "input": f"查询语句: {rewritten}",
            "output": RAGService._doc_items(docs),
        }

        if request.enable_bm25:
            bm25_docs = result.get("documents_bm25", [])
            data["bm25_retrieve"] = {
                "input": f"BM25查询: {query}",
                "output": RAGService._doc_items(bm25_docs),
            }

        if request.enable_hyde:
            hyde_docs = result.get("documents_hyde", [])
            data["hyde_retrieve"] = {
                "input": f"HyDE查询: {query}",
                "output": RAGService._doc_items(hyde_docs),
            }

        if request.enable_multi_query:
            mq_docs = result.get("documents_multi_query", [])
            data["multi_query_retrieve"] = {
                "input": f"多角度查询: {query}",
                "output": RAGService._doc_items(mq_docs),
            }

        # ── 图谱检索 ──
        data["kg_retrieve"] = {
            "input": [
                f"用户问题: {query}",
                f"KG意图: {'是' if kg_intent else '否'}",
            ],
            "output": [
                f"图谱上下文 ({len(kg_context)} 字符):",
                (kg_context[:300] + "...") if kg_context else "无图谱结果",
            ],
        }

        # ── 合并检索 ──
        strategies = ["语义检索"]
        if request.enable_bm25: strategies.append("BM25")
        if request.enable_hyde: strategies.append("HyDE")
        if request.enable_multi_query: strategies.append("多角度查询")
        data["merge_retrieval"] = {
            "input": [f"已汇聚 {len(strategies)} 路检索策略:"] + strategies,
            "output": RAGService._doc_items(docs),
        }

        # ── 重排序 ──
        if request.enable_rerank:
            data["rerank_documents"] = {
                "input": [f"待重排序文档 ({len(docs)} 份):"] + (RAGService._doc_items(docs) if docs else []),
                "output": [f"重排序结果 ({len(docs)} 份):"] + (RAGService._doc_items(docs) if docs else []),
            }

        # ── 文档评估 ──
        if request.enable_grade_documents:
            relevant = result.get('documents_relevant', False)
            grade_verdict = "全部通过" if relevant else "部分/全部未通过"
            data["grade_documents"] = {
                "input": [f"待评估文档 ({len(docs)} 份):"] + (RAGService._doc_items(docs) if docs else []),
                "output": [
                    f"评估结论: {grade_verdict}",
                    f"相关性: {'文档均合格，进入生成' if relevant else '不合格，触发重写/联网搜索'}",
                ],
            }

        # ── 查询重写 ──
        if request.enable_transform_query:
            rewritten_q = result.get('rewritten_query', query)
            data["transform_query"] = {
                "input": f"原始问题: {query}",
                "output": f"改写后查询: {rewritten_q}",
            }

        # ── 联网搜索 ──
        if request.enable_web_search:
            web_results = [d for d in docs if d.metadata.get("source") == "web"] if docs else []
            if web_results:
                web_items = []
                for d in web_results[:5]:
                    url = d.metadata.get("url", "未知来源")
                    preview = d.page_content[:80].replace("\n", " ")
                    web_items.append(f"{url}\n{preview}...")
                if len(web_results) > 5:
                    web_items.append(f"... 及其他 {len(web_results) - 5} 条")
                data["web_search"] = {
                    "input": f"联网搜索词: {query}",
                    "output": web_items,
                }
            else:
                data["web_search"] = {
                    "input": f"联网搜索词: {query}",
                    "output": ["搜索结果已合并到文档集，可在对应文档节点查看详情"],
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
