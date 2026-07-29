"""GraphRetriever - 知识图谱检索与推理

从用户问题中提取实体 → 实体链接 → 子图提取 → 多跳路径推理 → 生成结构化上下文
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any

import faiss
import numpy as np

# macOS 上 numpy 和 faiss 各自携带的 OpenMP 运行时可能冲突
# （OMP: Error #15），限制 FAISS 为单线程避免死锁。
faiss.omp_set_num_threads(1)

from src.agent.prompts import KG_RETRIEVE_ENTITY_EXTRACT_USER
from src.backend.llm import create_fast_llm
from src.backend.embedding import get_embedding_client
from src.knowledge_graph.graph_store import GraphStore
from src.config.settings import settings

logger = logging.getLogger(__name__)

# FAISS 索引与 SQLite 数据库文件名
_FAISS_INDEX_FILENAME = "entity_embeddings.index"
_SQLITE_DB_FILENAME = "entity_embeddings.db"
# 索引格式版本：用于检测磁盘上的旧格式文件是否需要重建
_INDEX_VERSION = "flatip_v1"


class EntityEmbeddingStore:
    """基于 FAISS + SQLite 的实体向量持久化存储

    FAISS Index（HNSW）：存储向量并支持高效的近似最近邻搜索（O(log n)）
    SQLite：维护 faiss_id ↔ entity_name 的映射关系（持久化到磁盘）

    用法：
        store = EntityEmbeddingStore("data/kg")
        store.rebuild(names, embeddings)   # 全量重建
        results = store.search(query_emb)  # 语义搜索
        store.save()                       # 持久化
    """

    def __init__(self, data_dir: str | Path):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._index_path = self._data_dir / _FAISS_INDEX_FILENAME
        self._db_path = self._data_dir / _SQLITE_DB_FILENAME

        # 初始化 SQLite
        self._conn = sqlite3.connect(str(self._db_path))
        self._init_db()

        # 初始化或加载 FAISS 索引
        self._index: faiss.Index = self._load_or_create_index()

    # ── 初始化 ────────────────────────────────────────────────

    def _init_db(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                faiss_id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)"
        )
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def _get_metadata(self, key: str) -> str | None:
        """读取 SQLite metadata"""
        cursor = self._conn.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def _set_metadata(self, key: str, value: str) -> None:
        """写入 SQLite metadata"""
        self._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def get_synced_count(self) -> int:
        """读取上次重建时记录的图谱节点数（持久化，跨进程/重启有效）"""
        val = self._get_metadata("synced_node_count")
        return int(val) if val else 0

    def set_synced_count(self, count: int) -> None:
        """持久化图谱节点数"""
        self._set_metadata("synced_node_count", str(count))

    def _load_or_create_index(self) -> faiss.Index:
        """从磁盘加载 FAISS 索引；若版本不匹配或不存在则返回空索引（仅占位）。

        版本检测确保索引格式变更（如 HNSW → FlatIP）时自动清理重建。
        """
        index_version = self._get_metadata("index_version")
        if self._index_path.exists() and index_version == _INDEX_VERSION:
            logger.info("从磁盘加载 FAISS 索引: %s", self._index_path)
            return faiss.read_index(str(self._index_path))

        # 版本不匹配或文件不存在 → 删除旧文件，重建
        if self._index_path.exists():
            logger.info("FAISS 索引格式变更 (%s → %s)，清除重建",
                         index_version or "none", _INDEX_VERSION)
            self._index_path.unlink()
        else:
            logger.info("FAISS 索引不存在，等待首次重建: %s", self._index_path)

        # 返回空占位索引，rebuild() 时替换为正确格式
        dummy = faiss.IndexFlat(1, 1)
        return faiss.IndexIDMap(dummy)

    # ── 属性 ──────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return self._index.ntotal

    # ── 核心操作 ──────────────────────────────────────────────

    def rebuild(self, names: list[str], embeddings: list[list[float]]) -> None:
        """全量重建索引（清空旧数据，重新构建）

        Args:
            names: 实体名称列表
            embeddings: 对应的 embedding 向量列表
        """
        if not names or not embeddings:
            logger.warning("rebuild 跳过：空数据")
            return

        dim = len(embeddings[0])
        logger.info("全量重建实体向量索引: %d 个实体, dim=%d", len(names), dim)

        # 使用 FlatIP（暴力内积搜索），向量归一化后内积 = cosine。
        # 79 个实体用 HNSW 近似搜索属于过度设计，暴力搜索更简单可靠。
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

        # 归一化 → 内积等价于 cosine 相似度
        embs = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embs)
        ids = np.arange(len(names), dtype=np.int64)
        self._index.add_with_ids(embs, ids)

        # 重建 SQLite 映射
        self._conn.execute("DELETE FROM entities")
        rows = [(int(ids[i]), names[i]) for i in range(len(names))]
        self._conn.executemany(
            "INSERT INTO entities (faiss_id, name) VALUES (?, ?)", rows
        )
        self.set_synced_count(len(names))
        self._set_metadata("index_version", _INDEX_VERSION)

        logger.info("实体向量索引重建完成，共 %d 个实体", len(names))

    def search(
        self, query_emb: list[float], top_k: int = 10
    ) -> list[tuple[str, float]]:
        """搜索最相似的实体

        不使用 FAISS 内置 search()（macOS 上 OpenMP 冲突导致挂死），
        改用 numpy 手动计算余弦相似度 + argsort 取 top-k。
        对于 < 10000 个实体的规模，手动搜索足够快。

        Args:
            query_emb: 查询向量
            top_k: 返回最相似的 k 个结果

        Returns:
            [(entity_name, cosine_similarity), ...]，按相似度降序排列
        """
        n = self._index.ntotal
        if n == 0:
            return []

        q = np.array(query_emb, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm  # 归一化查询向量

        # 从 FAISS 读取所有已归一化的实体向量
        # IndexIDMap 不支持 reconstruct，直接访问内层 IndexFlatIP
        inner_index = self._index.index  # IndexFlatIP
        dim = inner_index.d
        vectors = np.zeros((n, dim), dtype=np.float32)
        for i in range(n):
            vectors[i] = inner_index.reconstruct(i)

        # 矩阵乘法 → cosine 相似度
        scores = np.dot(vectors, q)  # shape (n,)

        # numpy argsort 取 top-k
        top_k_actual = min(top_k, n)
        if top_k_actual == n:
            sorted_indices = np.argsort(-scores)
        else:
            sorted_indices = np.argpartition(-scores, top_k_actual)[:top_k_actual]
            sorted_indices = sorted_indices[np.argsort(-scores[sorted_indices])]

        results: list[tuple[str, float]] = []
        for idx in sorted_indices:
            sim = float(scores[idx])
            sim = max(0.0, min(1.0, sim))
            cursor = self._conn.execute(
                "SELECT name FROM entities WHERE faiss_id = ?", (int(idx),)
            )
            row = cursor.fetchone()
            if row:
                results.append((row[0], round(sim, 4)))

        return results

        return results

    def save(self) -> None:
        """持久化 FAISS 索引到磁盘"""
        if self._index.ntotal == 0:
            return
        logger.info("持久化 FAISS 索引 (%d 个向量)", self._index.ntotal)
        faiss.write_index(self._index, str(self._index_path))
        self._conn.commit()

    def clear(self) -> None:
        """清空全部数据：删除磁盘上的 index 和 db 文件"""
        self._conn.close()
        for path in [self._index_path, self._db_path]:
            if path.exists():
                path.unlink()
        logger.info("实体向量缓存已清空 (%s)", self._data_dir)

    def close(self) -> None:
        """关闭数据库连接"""
        self._conn.close()


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
        self._entity_store = EntityEmbeddingStore(self._kg_dir)

    def clear_entity_cache(self) -> None:
        """清除实体向量缓存，删除磁盘文件，下次搜索时自动重建

        在文档删除导致图谱实体变更后调用，确保 FAISS 索引与图谱一致。
        """
        logger.info("清除实体向量缓存")
        self._entity_store.clear()
        self._entity_store = EntityEmbeddingStore(self._kg_dir)

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
        self, query: str, max_entities: int
    ) -> list[str]:
        """Step 1: LLM 从问题中抽取关键实体"""
        try:
            logger.info("Step 1 — LLM 实体抽取 (query='%s')", query[:80])
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
            sem_results = self._semantic_entity_search_batch(
                need_semantic, store, top_candidates=50
            )
            for name in sem_results:
                linked.add(name)

        return list(linked)[:10]

    # ── 语义搜索（FAISS + SQLite）─────────────────────────

    def _sync_entity_index(self, store: GraphStore, embedder: Any) -> None:
        """确保 FAISS 索引与图谱中的实体保持同步

        当图谱节点数量发生变化时，全量重建 FAISS 索引。
        同步状态持久化在 SQLite 中，跨进程/重启有效。
        分批调用 embedding API，避免一次性传输过多数据导致超时。
        """
        all_entities = store.get_all_entities()
        if not all_entities:
            return

        entity_names = [e["name"] for e in all_entities]
        current_count = store.node_count
        stored_count = self._entity_store.get_synced_count()

        if self._entity_store.size > 0 and current_count == stored_count:
            return  # 已同步

        logger.info(
            "图谱节点数变更 (磁盘:%d → 当前:%d)，重建实体向量索引 (%d 个实体)",
            stored_count, current_count, len(entity_names),
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

        # 全量重建 FAISS + SQLite
        self._entity_store.rebuild(entity_names, all_embeddings)
        self._entity_store.save()

    def _semantic_entity_search_batch(
        self,
        queries: list[str],
        store: GraphStore,
        top_candidates: int = 100,
    ) -> list[str]:
        """批量语义实体匹配

        使用 FAISS HNSW 索引进行高效的近似最近邻搜索，
        替代原 O(n) 的全量余弦相似度遍历。

        Args:
            queries: 待匹配的实体名称列表
            store: 图存储实例
            top_candidates: （保留参数，FAISS 使用精确的 top_k 而非预筛选）

        Returns:
            匹配到的实体名称列表（cosine >= 0.7）
        """
        all_entities = store.get_all_entities()
        if not all_entities:
            return []

        try:
            embedder = get_embedding_client()

            # Step A: 批量获取 query 的 embedding（注意：dashscope SDK 超时固定 300s）
            logger.info("Semantic Step A — 获取 query embedding: %d 个查询", len(queries))
            query_embs = embedder.embed_documents(queries)

            # Step B: 确保 FAISS 索引已同步
            self._sync_entity_index(store, embedder)

            # Step C: FAISS HNSW 近似搜索
            logger.info("Step C — FAISS 搜索 (%d 个查询向量)", len(query_embs))
            results: set[str] = set()
            for i, query_emb in enumerate(query_embs):
                matches = self._entity_store.search(query_emb, top_k=5)
                logger.info("FAISS 搜索第 %d 个查询返回 %d 个结果", i, len(matches))
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
