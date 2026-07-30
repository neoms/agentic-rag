"""GraphRetriever - 知识图谱检索与推理

从用户问题中提取实体 → 实体链接 → 子图提取 → 多跳路径推理 → 生成结构化上下文
实体向量索引使用 numpy .npz 二进制持久化（替代原 FAISS + SQLite 方案）。
"""

import json
import logging
from pathlib import Path

import numpy as np

from src.agent.prompts import KG_RETRIEVE_ENTITY_EXTRACT_USER
from src.backend.llm import create_fast_llm
from src.backend.embedding import get_embedding_client
from src.knowledge_graph.graph_store import GraphStore
from src.config.settings import settings

logger = logging.getLogger(__name__)


_NPZ_FILENAME = "entity_vectors.npz"


class NumpyVectorIndex:
    """基于 numpy .npz 的轻量实体向量索引

    存储结构（单个 .npz 文件）：
        names:      实体名称数组 (np.ndarray[str])
        embeddings: 实体嵌入矩阵 (np.ndarray[float32], shape [n, dim])
                    存储时已做 L2 归一化，搜索时 dot product = cosine similarity

    相比原 FAISS + SQLite 方案的优势：
        - 无外部依赖（仅 numpy）
        - 无 OpenMP 冲突（macOS 兼容）
        - 单文件读写，无状态同步问题
        - 对于 < 10000 实体的规模，numpy 矩阵乘法的性能（~ms 级）远优于 FAISS 近似搜索的开销
    """

    def __init__(self, data_dir: str | Path) -> None:
        self._path = Path(data_dir) / _NPZ_FILENAME
        self._names: np.ndarray = np.array([], dtype=str)
        self._embeddings: np.ndarray = np.array([], dtype=np.float32).reshape(0, 0)
        self._load()

    # ── 属性 ──────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._names)

    # ── 核心操作 ──────────────────────────────────────────────

    def rebuild(self, names: list[str], embeddings: list[list[float]]) -> None:
        """全量重建索引（清空旧数据，写入新的 .npz 文件）

        Args:
            names: 实体名称列表
            embeddings: 对应的 embedding 向量列表
        """
        if not names or not embeddings:
            logger.warning("NumpyVectorIndex.rebuild 跳过：空数据")
            return

        embs = np.array(embeddings, dtype=np.float32)
        # L2 归一化，使得 dot product = cosine similarity
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 防止除零
        embs = embs / norms

        self._names = np.array(names, dtype=str)
        self._embeddings = embs
        self._save()

        logger.info("实体向量索引重建完成: %d 个实体, dim=%d", len(names), embs.shape[1])

    def search(
        self, query_emb: list[float], top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """搜索最相似的实体

        Args:
            query_emb: 查询向量
            top_k: 返回最相似的 k 个结果

        Returns:
            [(entity_name, cosine_similarity), ...]，按相似度降序排列
        """
        n = self.size
        if n == 0:
            return []

        # 归一化查询向量
        q = np.array(query_emb, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        # 矩阵乘法 → cosine 相似度（embeddings 已归一化）
        scores = np.dot(self._embeddings, q)  # shape (n,)

        top_k_actual = min(top_k, n)
        # 高效选择 top-k
        if top_k_actual == n:
            sorted_indices = np.argsort(-scores)
        else:
            sorted_indices = np.argpartition(-scores, top_k_actual)[:top_k_actual]
            sorted_indices = sorted_indices[np.argsort(-scores[sorted_indices])]

        results: list[tuple[str, float]] = []
        for idx in sorted_indices:
            sim = max(0.0, min(1.0, float(scores[idx])))
            if sim >= 0.7:
                results.append((str(self._names[idx]), round(sim, 4)))

        return results

    # ── 持久化 ──────────────────────────────────────────────

    def _save(self) -> None:
        """持久化到 .npz 文件"""
        np.savez_compressed(self._path, names=self._names, embeddings=self._embeddings)

    def _load(self) -> None:
        """从 .npz 文件加载"""
        if self._path.exists():
            try:
                data = np.load(self._path)
                self._names = data["names"]
                self._embeddings = data["embeddings"]
                logger.debug("实体向量索引已加载: %d 个实体", len(self._names))
            except Exception as e:
                logger.warning("加载向量索引失败: %s，使用空索引", e)
                self._names = np.array([], dtype=str)
                self._embeddings = np.array([], dtype=np.float32).reshape(0, 0)
        else:
            logger.debug("向量索引文件不存在，使用空索引")

    def clear(self) -> None:
        """清空磁盘和内存中的向量索引"""
        if self._path.exists():
            self._path.unlink()
        self._names = np.array([], dtype=str)
        self._embeddings = np.array([], dtype=np.float32).reshape(0, 0)
        logger.info("实体向量索引已清空")


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
        self._kg_dir = settings.project_root / settings.kg_data_dir
        self._vector_index = NumpyVectorIndex(self._kg_dir)
        # 标记是否需要重建向量索引（文档删除后置为 True）
        self._dirty = True

    def mark_dirty(self) -> None:
        """标记向量索引为脏状态，下次检索时自动重建

        在文档删除导致图谱实体变更后调用。
        """
        self._dirty = True
        logger.info("实体向量索引标记为脏，下次检索时重建")

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
            logger.info("实体链接失败: 未在 %d 个实体中找到匹配", store.node_count)
            return "", []

        logger.info("实体链接成功: %s", seed_entities)

        # Step 3: 子图提取 (BFS N-hop)
        logger.info("Step 3 — 子图提取: 种子=%s, hops=%d", seed_entities, hops)
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
        self, query: str, max_entities: int,
    ) -> list[str]:
        """Step 1: LLM 从问题中抽取关键实体"""
        try:
            logger.info("Step 1 — LLM 实体抽取 (query='%s')", query[:80])
            llm = create_fast_llm()
            prompt = KG_RETRIEVE_ENTITY_EXTRACT_USER.format(
                query=query, max_entities=max_entities,
            )
            response = llm.invoke(prompt)
            raw = response.content.strip()
            logger.debug("LLM 实体抽取原始返回: %s", raw[:200])

            entities: list[str] = []
            for line in raw.split("\n"):
                line = line.strip()
                if line and len(line) > 1:
                    if line[0].isdigit() and (". " in line[:4] or "、" in line[:4]):
                        line = line.split(". ", 1)[-1] if ". " in line[:4] else line.split("、", 1)[-1]
                    line = line.strip()
                    if line:
                        entities.append(line)

            if not entities and raw:
                entities = [raw[:50]]

            return entities[:max_entities]

        except Exception as e:
            logger.warning("实体抽取失败: %s", e)
            return []

    def _link_entities(
        self, extracted: list[str], store: GraphStore,
    ) -> list[str]:
        """Step 2: Entity Linking — 精确匹配 + 别名匹配 + 语义搜索"""
        if store.is_empty():
            return []

        linked: set[str] = set()
        need_semantic: list[str] = []

        for entity_name in extracted:
            matched = False
            # 1) 精确匹配 + 别名匹配
            results = store.search_entities(entity_name, top_k=5)
            for name, score in results:
                if score >= 0.7:
                    linked.add(name)
                    matched = True
                    break
                elif score >= 0.5:
                    linked.add(name)
                    matched = True
                    break

            # 2) 未匹配到的收集起来统一做语义搜索
            if not matched:
                need_semantic.append(entity_name)

        # 3) 批量语义兜底
        if need_semantic:
            logger.info("精确匹配未命中的实体: %s，尝试语义匹配", need_semantic)
            sem_results = self._semantic_entity_search_batch(
                need_semantic, store, top_candidates=50,
            )
            for name in sem_results:
                linked.add(name)

        return list(linked)[:10]

    # ── 语义搜索（numpy .npz）───────────────────────────

    def _sync_vector_index(self, store: GraphStore, embedder) -> None:
        """确保 numpy 向量索引与图谱中的实体保持同步

        当图谱节点数量发生变化或索引被标记为脏时，全量重建向量索引。
        """
        all_entities = store.get_all_entities()
        if not all_entities:
            return

        entity_names = [e["name"] for e in all_entities]
        current_count = store.node_count

        # 检查是否需要重建
        if not self._dirty and self._vector_index.size == current_count:
            return  # 已同步

        logger.info(
            "重建实体向量索引: %d 个实体 (索引状态: size=%d, dirty=%s)",
            len(entity_names), self._vector_index.size, self._dirty,
        )

        # 分批获取 embedding，避免单次 API 调用数据量过大导致超时
        BATCH_SIZE = 32
        all_embeddings: list[list[float]] = []
        for i in range(0, len(entity_names), BATCH_SIZE):
            batch = entity_names[i : i + BATCH_SIZE]
            batch_embs = embedder.embed_documents(batch)
            all_embeddings.extend(batch_embs)
            progress = min(i + BATCH_SIZE, len(entity_names))
            logger.info("实体 embedding 进度: %d/%d", progress, len(entity_names))

        # 全量重建 numpy 索引
        self._vector_index.rebuild(entity_names, all_embeddings)
        self._dirty = False

    def _semantic_entity_search_batch(
        self,
        queries: list[str],
        store: GraphStore,
        top_candidates: int = 100,
    ) -> list[str]:
        """批量语义实体匹配

        使用 numpy 矩阵乘法进行精确余弦相似度搜索，
        替代原 FAISS HNSW 近似搜索（小规模数据集上更简单可靠）。

        Args:
            queries: 待匹配的实体名称列表
            store: 图存储实例
            top_candidates: （保留参数，兼容旧接口）

        Returns:
            匹配到的实体名称列表（cosine >= 0.7）
        """
        all_entities = store.get_all_entities()
        if not all_entities:
            return []

        try:
            embedder = get_embedding_client()

            # Step A: 批量获取 query 的 embedding
            logger.info("Semantic Step A — 获取 query embedding: %d 个查询", len(queries))
            query_embs = embedder.embed_documents(queries)

            # Step B: 确保 numpy 向量索引已同步
            self._sync_vector_index(store, embedder)

            # Step C: numpy 矩阵乘法搜索
            logger.info("Step C — numpy 向量搜索 (%d 个查询向量)", len(query_embs))
            results: set[str] = set()
            for i, query_emb in enumerate(query_embs):
                matches = self._vector_index.search(query_emb, top_k=5)
                logger.info("向量搜索第 %d 个查询返回 %d 个结果", i, len(matches))
                for name, score in matches:
                    if score >= 0.7:
                        logger.debug("语义匹配: score=%.2f → '%s'", score, name)
                        results.add(name)

            logger.info("Step C 完成，语义匹配结果: %s", list(results))
            return list(results)

        except Exception as e:
            logger.warning("批量语义实体搜索异常: %s", e)
            return []

    def _find_entity_paths(
        self, entities: list[str], store: GraphStore,
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
                        formatted: list[str] = []
                        for k in range(len(path) - 1):
                            edge_data = store.get_edge_data(path[k], path[k + 1])
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
