"""GraphRetriever - 知识图谱检索与推理

从用户问题中提取实体 → 实体链接 → 子图提取 → 多跳路径推理 → 生成结构化上下文
实体向量索引使用 numpy .npz 二进制持久化（替代原 FAISS + SQLite 方案）。
"""

import logging
from pathlib import Path

import numpy as np

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

    核心能力：
      - 实体间路径推理
      - 子图转上下文文本
      - 向量索引同步标记

    注意：实体检索不再走此模块，改用 GraphStore.search_entities() 直接模糊匹配。
    """

    def __init__(self):
        self._kg_dir = settings.project_root / settings.kg_data_dir
        self._dirty = False

    def mark_dirty(self) -> None:
        """标记向量索引为脏状态（保留兼容接口）"""
        self._dirty = True
        logger.info("实体索引标记为脏")

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
