"""Agent/RAG 服务层 - 流式 Agent 编排与对话交互"""

import time
import asyncio
import logging
import json
import re
from typing import AsyncIterator

from langchain_core.messages import HumanMessage

from src.agent.graph import agent_graph
from src.agent.state import AgentState
from src.agent.prompts import CHECK_HALLUCINATION_SYSTEM, CHECK_HALLUCINATION_USER
from src.models.chat import (
    AgenticChatRequest,
    SourceDocument,
    ChatHistoryResponse,
    ChatHistoryMessage,
    StreamEvent,
)
from src.memory.manager import memory_manager
from src.backend.llm import create_strong_llm, create_fast_llm

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
            "answer": "",
            "hallucination_detected": False,
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
            "citation_metadata": {},
        }

        # 使用 astream_events 获取节点的开始/完成事件（on_chain_start/on_chain_end）
        # generate/check_hallucination 不在此处发送，由外部流式生成和幻觉检测阶段手动控制
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
        # 图执行完成，开始外部流式生成（generate 节点从此刻起持续活跃）
        yield StreamEvent(event="node_start", data="generate")

        # 获取最终状态（含检索到的文档和引文元数据）— 放到线程池避免阻塞事件循环
        final_state = await asyncio.to_thread(agent_graph.get_state, config)
        result = final_state.values if final_state else {}
        logger.info("[stream_rag] 状态图完成, graph_elapsed=%.2fs", time.time() - t0)

        # 从 final_state 构建各图内节点的 I/O 数据
        node_data: dict[str, dict[str, str | list[str]]] = self._build_node_data(result, request)

        agent_path = result.get("agent_path", [])
        # 过滤掉 graph 内部标记为 skipped 的节点（非流式模式或空状态下的安全兜底）
        agent_path = [p for p in agent_path if "skipped" not in p]
        # 统一 generate / check_hallucination 标签（流式模式下保留运行过的节点）
        agent_path = [p.replace(" (streaming)", "") for p in agent_path]
        documents = result.get("documents", [])
        logger.info("[stream_rag] 状态图完成: path=%s, docs=%d, graph_elapsed=%.2fs",
                     agent_path, len(documents), time.time() - t0)

        # 发送检索结果来源（全部传给 LLM 的参考源）
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

        # 流式生成答案（带引文标注）
        if documents:
            # ===== 1. 构建带段落索引的文档上下文 + 同步构建 citation_metadata =====
            doc_parts: list[str] = []
            citation_metadata: dict[str, dict] = {}
            for doc_idx, doc in enumerate(documents, 1):
                src = doc.metadata.get("url") or doc.metadata.get("filename", "unknown")
                url_info = f"\n链接: {doc.metadata['url']}" if doc.metadata.get("url") else ""
                source_type = doc.metadata.get("source", "local")
                url = doc.metadata.get("url", "")
                # 按段落拆分（双换行分隔），为每段分配 [DocX-ParaY] 标识
                paragraphs = [p.strip() for p in re.split(r'\n\s*\n', doc.page_content) if p.strip()]
                if not paragraphs:
                    paragraphs = [doc.page_content]
                para_lines = []
                for para_idx, para in enumerate(paragraphs, 1):
                    citation_key = f"Doc{doc_idx}-Para{para_idx}"
                    para_lines.append(f"  [{citation_key}] {para}")
                    # 同步构建 citation_metadata，确保索引与 LLM 看到的完全一致
                    citation_metadata[citation_key] = {
                        "filename": src,
                        "source_type": source_type,
                        "url": url,
                        "paragraph_text": para,
                        "doc_index": doc_idx,
                        "para_index": para_idx,
                    }
                doc_text = "\n\n".join(para_lines)
                doc_parts.append(f"来源: {src}{url_info}\n内容:\n{doc_text}")
            docs_text = "\n\n---\n\n".join(doc_parts)

            # 发送引文标注元数据（用刚刚构建的，确保与 prompt 内索引一致）
            if citation_metadata:
                yield StreamEvent(
                    event="citations",
                    data=json.dumps(citation_metadata, ensure_ascii=False),
                )

            chat_history = memory_manager.get_chat_history_string(request.session_id)

            prompt = f"""你是一个专业的知识问答助手。请基于提供的文档上下文回答用户问题。

规则：
1. 优先使用提供的文档信息回答；信息不足时明确说明
2. 回答简洁准确有条理，使用中文
3. 【重要】每句陈述性内容末尾都必须标注来源，格式为 [DocX-ParaY]
   - 单源标注: "Python是动态类型语言 [Doc1-Para2]。"
   - 多源标注: "机器学习分为三类 [Doc1-Para1, Doc2-Para3]。"
4. 每句话至少有一个引用标注（总结句可多源标注），没有来源的陈述不要写
5. 严格使用文档中提供的 [DocX-ParaY] 标识

文档上下文：
{docs_text}

对话历史：
{chat_history or '无'}

用户问题：{request.query}

请回答（每句话末尾都标注来源）："""

            llm = create_strong_llm(streaming=True)
            full_answer = ""
            async for chunk in llm.astream(prompt):
                if chunk.content:
                    full_answer += chunk.content
                    yield StreamEvent(event="token", data=chunk.content)

            # 填充 generate 节点的 I/O 数据（含具体文档列表）
            gen_input: list[str] = [f"问题: {request.query}"]
            for i, doc in enumerate(documents[:8]):
                src = doc.metadata.get("url") or doc.metadata.get("filename", f"文档{i+1}")
                preview = doc.page_content[:100].replace("\n", " ")
                gen_input.append(f"参考 {src}:\n{preview}...")
            if len(documents) > 8:
                gen_input.append(f"... 及其他 {len(documents) - 8} 条")
            node_data["generate"] = {
                "input": gen_input,
                "output": full_answer,
            }

            # 流式生成完成后，进行幻觉检测（仅在开启自反思时）
            hallucination_passed = True
            hallucination_faithfulness = 100.0
            if request.enable_reflection:
                yield StreamEvent(event="node_start", data="check_hallucination")
                try:
                    docs_for_check = "\n---\n".join(
                        f"[文档 {i+1}] {doc.page_content[:500]}"
                        for i, doc in enumerate(documents[:8])
                    )
                    check_llm = create_fast_llm()
                    check_messages = [
                        HumanMessage(content=CHECK_HALLUCINATION_SYSTEM),
                        HumanMessage(
                            content=(
                                CHECK_HALLUCINATION_USER.format(
                                    documents=docs_for_check,
                                    answer=full_answer,
                                )
                                + "\n\n输出要求：请返回一个 JSON 对象，包含两个字段："
                                '"passed" (布尔值，true 表示答案忠实于文档，false 表示存在编造)，'
                                '"faithfulness" (浮点数，0.0~100.0，精确到小数点后一位，表示答案对文档的忠实度百分比)。'
                                '只输出 JSON，不要输出其他内容。'
                                '示例：{"passed": true, "faithfulness": 92.5}'
                            ),
                        ),
                    ]
                    check_response = await check_llm.ainvoke(check_messages)
                    check_raw = check_response.content.strip()
                    # 从 LLM 返回中提取 JSON
                    import re as _re
                    json_match = _re.search(r'\{[^{}]*\}', check_raw)
                    if json_match:
                        check_data = json.loads(json_match.group())
                        hallucination_faithfulness = float(check_data.get("faithfulness", 100))
                        hallucination_faithfulness = max(0.0, min(100.0, round(hallucination_faithfulness, 1)))
                        hallucination_passed = check_data.get("passed", True)
                    else:
                        hallucination_passed = "PASSED" in check_raw.upper() or "true" in check_raw.lower()
                    logger.info(
                        "[stream_rag] 幻觉检测: faithfulness=%.1f%%, passed=%s",
                        hallucination_faithfulness,
                        str(hallucination_passed),
                    )
                except Exception as e:
                    logger.warning("[stream_rag] 幻觉检测异常: %s", e)

                yield StreamEvent(event="node_step", data="check_hallucination")
                yield StreamEvent(
                    event="hallucination",
                    data=json.dumps({
                        "passed": hallucination_passed,
                        "result": "PASSED" if hallucination_passed else "FAILED",
                        "faithfulness": hallucination_faithfulness,
                    }, ensure_ascii=False),
                )
                # 填充 check_hallucination 节点的 I/O 数据
                node_data["check_hallucination"] = {
                    "input": [
                        f"待检测答案 ({len(full_answer)} 字符):",
                        full_answer[:300] + ("..." if len(full_answer) > 300 else ""),
                    ],
                    "output": [
                        f"忠实度: {hallucination_faithfulness}%",
                        f"判定: {'PASSED ✓' if hallucination_passed else 'FAILED ✗'}",
                        f"结果: {'答案忠实于参考文档' if hallucination_passed else '答案存在编造，需要重试'}",
                    ],
                }

            # 记录对话
            memory_manager.add_interaction(request.session_id, request.query, full_answer)
            logger.info("[stream_rag] 流式生成完成: answer_len=%d", len(full_answer))
        else:
            msg = "未找到相关文档。"
            yield StreamEvent(event="token", data=msg)
            memory_manager.add_interaction(request.session_id, request.query, msg)
            logger.info("[stream_rag] 无文档，返回兜底回答")
            node_data["generate"] = {
                "input": [f"用户问题: {request.query}", "参考文档: 无（未检索到相关文档）"],
                "output": "兜底回答: 未找到相关文档",
            }

        # 发送节点 I/O 数据（用于前端流程图点击展示）
        yield StreamEvent(
            event="node_data",
            data=json.dumps(node_data, ensure_ascii=False),
        )
        # 完成
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
        kg_intent = result.get('kg_intent', False)
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

        # ── 重排序（输出与输入文档内容一致，仅排序变化） ──
        if request.enable_rerank:
            data["rerank_documents"] = {
                "input": [f"待重排序文档 ({len(docs)} 份):"] + (RAGService._doc_items(docs) if docs else []),
                "output": [f"重排序结果 ({len(docs)} 份):"] + (RAGService._doc_items(docs) if docs else []),
            }

        # ── 文档评估 ──
        if request.enable_grade_documents:
            relevant = result.get('documents_relevant', False)
            # 从 agent_path 推断哪些文档被标记为相关
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
            # 从最终文档中找到来源为 web 的结果
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
