"""GraphRetriever - 知识图谱检索与推理

从用户问题中提取实体 → 实体链接 → 子图提取 → 多跳路径推理 → 生成结构化上下文
"""

import json
import logging
from typing import Any

from src.agent.prompts import KG_RETRIEVE_ENTITY_EXTRACT_USER
from src.backend.llm import create_fast_llm
from src.backend.embedding import get_embedding_client
from src.knowledge_graph.graph_store import GraphStore
from src.config.settings import settings

logger = logging.getLogger(__name__)


class GraphRetriever:
    """图谱检索推理器

    流程:
        1. Entity Extraction: LLM 从 query 抽取关键实体
        2. Entity Linking: 精确匹配 + 别名匹配 + 语义相似度 → 定位种子节点
        3. Subgraph Extraction: BFS N-hop 遍历
        4. Path Reasoning: 种子实体间的关联路径
        5. Context Generation: 子图转自然语言文本
    """

    def __init__(self):
        # 缓存：全量实体名称及其 embedding，避免重复调用 API
        self._entity_embeddings_cache: list[tuple[str, list[float]]] | None = None
        self._entity_embeddings_node_count: int = 0

    def search(
        self,
        query: str,
        store: GraphStore,
        max_hops: int | None = None,
        max_entities: int | None = None,
    ) -> tuple[str, list[str]]:
        """执行图谱检索

        Args:
            query: 用户问题
            store: 图存储实例
            max_hops: 子图提取最大跳数（默认使用 settings.kg_max_hops）
            max_entities: 最多提取实体数（默认使用 settings.kg_max_entities）

        Returns:
            (上下文文本, 实体列表) 元组
        """
        if store.is_empty():
            logger.info("图谱为空，跳过检索")
            return "", []

        hops = max_hops if max_hops is not None else settings.kg_max_hops
        max_ent = max_entities if max_entities is not None else settings.kg_max_entities

        # Step 1: LLM 实体抽取
        extracted_entities = self._extract_entities_from_query(query, max_ent)
        if not extracted_entities:
            logger.info("未从问题中抽取到实体")
            return "", []

        logger.info("LLM 抽取实体: %s", extracted_entities)

        # Step 2: Entity Linking — 在图谱中定位实体
        seed_entities = self._link_entities(extracted_entities, store)
        if not seed_entities:
            logger.info("未在图谱中找到匹配实体")
            return "", []

        logger.info("实体链接成功: %s", seed_entities)

        # Step 3: 子图提取
        subgraph = store.get_subgraph(seed_entities, hops=hops)
        if subgraph.number_of_nodes() == 0:
            logger.info("子图为空")
            return "", seed_entities

        logger.info("子图: %d nodes, %d edges",
                     subgraph.number_of_nodes(), subgraph.number_of_edges())

        # Step 4: 路径推理（多实体之间）
        paths_text = self._find_entity_paths(seed_entities, store)

        # Step 5: 生成上下文文本
        context = self._subgraph_to_text(subgraph, seed_entities)
        if paths_text:
            context += f"\n\n关联路径:\n{paths_text}"

        return context, seed_entities

    # ── 私有方法 ──────────────────────────────────────────

    def _extract_entities_from_query(
        self, query: str, max_entities: int
    ) -> list[str]:
        """Step 1: LLM 从问题中抽取关键实体"""
        try:
            llm = create_fast_llm()
            prompt = KG_RETRIEVE_ENTITY_EXTRACT_USER.format(
                query=query, max_entities=max_entities
            )
            response = llm.invoke(prompt)
            raw = response.content.strip()
            logger.debug("LLM 实体抽取原始返回: %s", raw[:200])

            entities: list[str] = []
            for line in raw.split("\n"):
                line = line.strip()
                # 去除编号前缀
                if line and len(line) > 1:
                    # 去掉可能的编号
                    if line[0].isdigit() and (". " in line[:4] or "、" in line[:4]):
                        line = line.split(". ", 1)[-1] if ". " in line[:4] else line.split("、", 1)[-1]
                    line = line.strip()
                    if line:
                        entities.append(line)

            if not entities and raw:
                # fallback: 整行作为单个实体
                entities = [raw[:50]]

            return entities[:max_entities]

        except Exception as e:
            logger.warning("实体抽取失败: %s", e)
            return []

    def _link_entities(
        self, extracted: list[str], store: GraphStore
    ) -> list[str]:
        """Step 2: Entity Linking — 精确匹配 + 别名匹配 + 语义搜索"""
        if store.is_empty():
            return []

        linked: set[str] = set()
        need_semantic: list[str] = []  # 需要语义兜底的实体

        for entity_name in extracted:
            matched = False
            # 1) 精确匹配 + 别名匹配
            results = store.search_entities(entity_name, top_k=5)
            for name, score in results:
                if score >= 0.7:  # 精确匹配或别名匹配
                    linked.add(name)
                    matched = True
                    break
                elif score >= 0.5:  # 子串部分匹配
                    linked.add(name)
                    matched = True
                    break

            # 2) 未匹配到的实体收集起来，统一做语义搜索
            if not matched:
                need_semantic.append(entity_name)

        # 3) 批量语义兜底（一次 embedding 批量调用来处理所有未匹配实体）
        if need_semantic:
            logger.info("精确匹配未命中的实体: %s，尝试语义匹配", need_semantic)
            # 限制语义搜候选数量，避免全部 552 个逐个调用
            sem_results = self._semantic_entity_search_batch(
                need_semantic, store, top_candidates=50
            )
            for name in sem_results:
                linked.add(name)

        return list(linked)[:10]

    def _semantic_entity_search_batch(
        self,
        queries: list[str],
        store: GraphStore,
        top_candidates: int = 100,
    ) -> list[str]:
        """批量语义实体匹配

        一次性获取所有 entity 名称和 embedding，只做本地余弦相似度计算，
        避免对 552 个实体逐个调用 embedding API。

        Args:
            queries: 待匹配的实体名称列表
            store: 图存储实例
            top_candidates: 候选实体数量上限（按名称长度预筛选）

        Returns:
            匹配到的实体名称列表
        """
        all_entities = store.get_all_entities()
        if not all_entities:
            return []

        try:
            embedder = get_embedding_client()

            # Step A: 批量获取 query 的 embedding（一次 API 调用）
            query_embs = embedder.embed_documents(queries)

            # Step B: 检查/更新实体 embedding 缓存
            entity_names = [e["name"] for e in all_entities]
            current_count = store.node_count
            if (self._entity_embeddings_cache is None or
                    current_count != self._entity_embeddings_node_count):
                # 缓存过期或不存在：批量获取所有实体的 embedding
                logger.info("构建实体 embedding 缓存 (%d 个实体)", len(entity_names))
                # 如果实体太多，只取 top_candidates 个（按名称长度预估相关性）
                candidate_names = entity_names[:top_candidates]
                candidate_embs = embedder.embed_documents(candidate_names)
                self._entity_embeddings_cache = list(zip(candidate_names, candidate_embs))
                self._entity_embeddings_node_count = current_count

            # Step C: 本地计算余弦相似度（无 API 调用，极快）
            results: set[str] = set()
            cached_names = [name for name, _ in self._entity_embeddings_cache]
            cached_embs = [emb for _, emb in self._entity_embeddings_cache]

            # 预计算 entity embedding 的范数
            norms = [
                sum(a * a for a in emb) ** 0.5 if emb else 0.0
                for emb in cached_embs
            ]

            for i, (query, query_emb) in enumerate(zip(queries, query_embs)):
                q_norm = sum(a * a for a in query_emb) ** 0.5
                if q_norm == 0:
                    continue

                best_score = -1.0
                best_name: str | None = None
                for j, (ent_name, ent_emb) in enumerate(self._entity_embeddings_cache):
                    if norms[j] == 0:
                        continue
                    dot = sum(a * b for a, b in zip(query_emb, ent_emb))
                    score = dot / (q_norm * norms[j])
                    if score > best_score:
                        best_score = score
                        best_name = ent_name

                if best_name and best_score >= 0.7:
                    logger.debug("语义匹配: '%s' → '%s' (score=%.2f)",
                                  query, best_name, best_score)
                    results.add(best_name)

            return list(results)

        except Exception as e:
            logger.warning("批量语义实体搜索异常: %s", e)
            return []

    def _find_entity_paths(
        self, entities: list[str], store: GraphStore
    ) -> str:
        """Step 4: 查找实体间关联路径"""
        if len(entities) < 2:
            return ""

        paths_lines: list[str] = []
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                paths = store.find_paths(entities[i], entities[j], max_len=4)
                if paths:
                    for path in paths[:3]:  # 每个实体对最多 3 条路径
                        # 格式化路径：实体 → 关系 → 实体
                        formatted: list[str] = []
                        for k in range(len(path) - 1):
                            edge_data = store.graph.get_edge_data(path[k], path[k+1])
                            if edge_data:
                                rel = edge_data.get("relation", "→")
                                formatted.append(f"{path[k]} --[{rel}]--> {path[k+1]}")
                            else:
                                formatted.append(f"{path[k]} → {path[k+1]}")
                        paths_lines.append(" → ".join(formatted))

                if len(paths_lines) >= 10:
                    break
            if len(paths_lines) >= 10:
                break

        return "\n".join(paths_lines)

    @staticmethod
    def _subgraph_to_text(subgraph, seed_entities: list[str]) -> str:
        """Step 5: 将子图转为自然语言文本"""
        lines: list[str] = []
        lines.append(f"知识图谱中与问题相关的实体和关系（共 {subgraph.number_of_nodes()} 个实体，{subgraph.number_of_edges()} 条关系）：\n")

        # 描述实体
        lines.append("【实体列表】")
        for node, attrs in subgraph.nodes(data=True):
            aliases = attrs.get("aliases", [])
            alias_str = f"（别名：{', '.join(aliases)}）" if aliases else ""
            is_seed = "★" if node in seed_entities else "  "
            lines.append(f"{is_seed} {node} [{attrs.get('type', 'unknown')}] {alias_str}")

        # 描述关系
        lines.append("\n【关系列表】")
        for src, dst, attrs in subgraph.edges(data=True):
            lines.append(f"  {src} --[{attrs.get('relation', 'related_to')}]--> {dst}")

        return "\n".join(lines)
