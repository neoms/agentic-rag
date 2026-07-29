"""GraphStore - 基于 NetworkX 的内存知识图谱存储 + JSON 持久化

节点：实体（Entity），边：关系（Relation）
"""

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

from src.config.settings import settings

logger = logging.getLogger(__name__)


class GraphStore:
    """NetworkX 有向图存储，JSON 文件持久化

    数据目录: kg_data/
    持久化文件: kg_data/graph.json
    """

    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self.data_dir = settings.project_root / settings.kg_data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._file_path = self.data_dir / "graph.json"
        self._load()

    # ── 属性 ──────────────────────────────────────────────

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def is_empty(self) -> bool:
        return self._graph.number_of_nodes() == 0

    # ── 实体操作 ──────────────────────────────────────────

    def add_entity(
        self,
        name: str,
        entity_type: str = "unknown",
        doc_id: str = "",
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """添加或更新实体节点（公开方法，委托给 _add_entity_impl）

        Args:
            name: 实体名称（唯一标识）
            entity_type: 实体类型（person, org, location, concept 等）
            doc_id: 来源文档 ID
            aliases: 别名列表
            metadata: 额外元数据
        """
        self._add_entity_impl(name, entity_type, doc_id, aliases, metadata)

    def _add_entity_impl(
        self,
        name: str,
        entity_type: str = "unknown",
        doc_id: str = "",
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """add_entity 的无锁实现，供 bulk_import 内部调用"""
        if name not in self._graph:
            self._graph.add_node(
                name,
                type=entity_type,
                doc_ids=[doc_id] if doc_id else [],
                aliases=aliases or [],
                metadata=metadata or {},
            )
        else:
            node = self._graph.nodes[name]
            if entity_type != "unknown" and node.get("type") == "unknown":
                node["type"] = entity_type
            if doc_id and doc_id not in node["doc_ids"]:
                node["doc_ids"].append(doc_id)
            if aliases:
                existing = set(node.get("aliases", []))
                for a in aliases:
                    if a not in existing:
                        node["aliases"].append(a)
            if metadata:
                node["metadata"].update(metadata)

    # ── 关系操作 ──────────────────────────────────────────

    def add_relation(
        self,
        source: str,
        target: str,
        relation: str,
        weight: float = 1.0,
        doc_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """添加或更新关系边（公开方法，委托给 _add_relation_impl）

        Args:
            source: 源实体名称
            target: 目标实体名称
            relation: 关系类型（is_a, part_of, located_in, works_for 等）
            weight: 关系权重
            doc_id: 来源文档 ID
            metadata: 额外元数据
        """
        self._add_relation_impl(source, target, relation, weight, doc_id, metadata)

    def _add_relation_impl(
        self,
        source: str,
        target: str,
        relation: str,
        weight: float = 1.0,
        doc_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """add_relation 的无锁实现 + 跨文档 doc_ids 合并

        当边已存在时合并 doc_ids 而非覆盖，确保跨文档来源信息不丢失。
        """
        new_doc_ids = [doc_id] if doc_id else []

        if self._graph.has_edge(source, target):
            existing = self._graph.edges[source, target]
            existing_doc_ids = existing.get("doc_ids", [])
            for d in new_doc_ids:
                if d not in existing_doc_ids:
                    existing_doc_ids.append(d)
            existing.update(
                relation=relation,
                weight=weight,
                doc_ids=existing_doc_ids,
                metadata=metadata or {},
            )
        else:
            self._graph.add_edge(
                source,
                target,
                relation=relation,
                weight=weight,
                doc_ids=new_doc_ids,
                metadata=metadata or {},
            )

    # ── 删除操作 ──────────────────────────────────────────

    def remove_doc_id(self, doc_id: str) -> tuple[int, int]:
        """从图谱中移除指定文档的所有实体和关系引用

        处理策略（引用计数式级联删除）：
        1. 对每条边/每个节点，从其 doc_ids 列表中移除目标 doc_id
        2. 若 doc_ids 列表变空，说明该元素仅属于被删文档 → 删除
        3. 若 doc_ids 列表非空，说明它还被其他文档引用 → 仅清理引用，保留元素
        4. 孤立节点被删除时，NetworkX 自动移除其所有关联边（级联清理）

        Args:
            doc_id: 要删除的文档 ID

        Returns:
            (removed_entity_count, removed_relation_count)
        """
        if not doc_id:
            return 0, 0

        if self.is_empty():
            logger.debug("图谱为空，跳过文档 %s 的 KG 清理", doc_id)
            return 0, 0

        # ---- 第一轮：扫描边，清理引用 + 标记空边 ----
        edges_explicit: list[tuple[str, str]] = []
        for src, dst, attrs in list(self._graph.edges(data=True)):
            doc_ids: list[str] = attrs.get("doc_ids", [])
            if doc_id not in doc_ids:
                continue
            doc_ids.remove(doc_id)
            if not doc_ids:
                edges_explicit.append((src, dst))

        # 执行显式边删除
        for src, dst in edges_explicit:
            if self._graph.has_edge(src, dst):
                self._graph.remove_edge(src, dst)

        # ---- 第二轮：扫描节点，清理引用 + 标记空节点 ----
        nodes_explicit: list[str] = []
        for node, attrs in list(self._graph.nodes(data=True)):
            doc_ids: list[str] = attrs.get("doc_ids", [])
            if doc_id not in doc_ids:
                continue
            doc_ids.remove(doc_id)
            if not doc_ids:
                nodes_explicit.append(node)

        # 统计因节点删除而被级联删除的边（这些边未被第一轮覆盖）
        edges_cascaded = 0
        for node in nodes_explicit:
            edges_cascaded += self._graph.degree(node)

        # 执行节点删除（NetworkX 自动移除该节点的所有关联边）
        for node in nodes_explicit:
            self._graph.remove_node(node)

        # ---- 持久化 ----
        self.save()

        removed_entities = len(nodes_explicit)
        removed_relations = len(edges_explicit) + edges_cascaded
        logger.info(
            "文档 %s 的 KG 数据已清理: 移除 %d 实体(节点), %d 关系(边)",
            doc_id, removed_entities, removed_relations,
        )
        return removed_entities, removed_relations

    # ── 批量导入 ──────────────────────────────────────────

    def bulk_import(
        self,
        doc_id: str,
        entities: list[dict[str, Any]],
        relations: list[dict[str, Any]],
    ) -> tuple[int, int]:
        """原子化批量导入实体和关系（单次 save，无需逐条加锁）

        跨文档名称相同的实体/关系自动合并 doc_ids 来源信息。
        关系两端如不在图中则自动创建缺失的实体节点。

        Args:
            doc_id: 来源文档 ID
            entities: [{"name": ..., "type": ..., "aliases": ..., "metadata": ...}, ...]
            relations: [{"source": ..., "target": ..., "relation": ...}, ...]

        Returns:
            (entity_count, relation_count)
        """
        entity_count = 0
        relation_count = 0

        for ent in entities:
            name = ent.get("name", "").strip()
            if not name:
                continue
            self._add_entity_impl(
                name=name,
                entity_type=ent.get("type", "unknown"),
                doc_id=doc_id,
                aliases=ent.get("aliases", []),
                metadata=ent.get("metadata", {}),
            )
            entity_count += 1

        for rel in relations:
            src = rel.get("source", "").strip()
            dst = rel.get("target", "").strip()
            rel_type = rel.get("relation", "related_to").strip()
            if not src or not dst:
                continue
            # 自动补全图上缺失的关系端节点
            if not self._graph.has_node(src):
                self._add_entity_impl(name=src, entity_type="unknown", doc_id=doc_id)
            if not self._graph.has_node(dst):
                self._add_entity_impl(name=dst, entity_type="unknown", doc_id=doc_id)
            self._add_relation_impl(
                source=src, target=dst, relation=rel_type, doc_id=doc_id,
            )
            relation_count += 1

        self.save()
        return entity_count, relation_count

    # ── 查询操作 ──────────────────────────────────────────

    def get_neighbors(
        self, entity_name: str, hops: int = 1
    ) -> dict[str, list[str]]:
        """获取指定实体的 N-hop 邻居

        Returns:
            {实体名: [邻居实体名列表]}，包含实体本身
        """
        if entity_name not in self._graph:
            return {}

        nodes: set[str] = {entity_name}
        frontier: set[str] = {entity_name}

        for _ in range(hops):
            next_frontier: set[str] = set()
            for n in frontier:
                next_frontier.update(self._graph.successors(n))
                next_frontier.update(self._graph.predecessors(n))
            nodes.update(next_frontier)
            frontier = next_frontier - nodes

        result: dict[str, list[str]] = {}
        for n in nodes:
            result[n] = list(set(
                list(self._graph.successors(n)) + list(self._graph.predecessors(n))
            ))
        return result

    def get_subgraph(
        self, entity_names: list[str], hops: int = 2
    ) -> nx.DiGraph:
        """从种子实体出发，提取 N-hop 子图

        Args:
            entity_names: 种子实体名称列表
            hops: 跳数

        Returns:
            子图 (nx.DiGraph)
        """
        if not entity_names or self.is_empty():
            return nx.DiGraph()

        nodes: set[str] = set(entity_names) & set(self._graph.nodes())
        if not nodes:
            return nx.DiGraph()

        frontier = set(nodes)
        for _ in range(hops):
            next_frontier: set[str] = set()
            for n in frontier:
                next_frontier.update(self._graph.successors(n))
                next_frontier.update(self._graph.predecessors(n))
            nodes.update(next_frontier)
            frontier = next_frontier - nodes

        return self._graph.subgraph(nodes).copy()

    def find_paths(
        self, source: str, target: str, max_len: int = 4
    ) -> list[list[str]]:
        """查找两个实体之间的所有路径（限制长度）"""
        if source not in self._graph or target not in self._graph:
            return []
        try:
            return list(
                nx.all_simple_paths(self._graph, source, target, cutoff=max_len)
            )
        except nx.NetworkXNoPath:
            return []

    def search_entities(
        self, query: str, top_k: int = 10
    ) -> list[tuple[str, float]]:
        """模糊搜索实体（精确匹配 + 别名匹配 + 子串匹配）

        Returns:
            [(实体名, 匹配分数)] 列表
        """
        if self.is_empty():
            return []

        results: list[tuple[str, float]] = []
        query_lower = query.lower()

        for node, attrs in self._graph.nodes(data=True):
            score = 0.0
            # 精确匹配
            if query_lower == node.lower():
                score = 1.0
            # 别名匹配
            elif query_lower in [a.lower() for a in attrs.get("aliases", [])]:
                score = 0.9
            # 子串匹配
            elif query_lower in node.lower():
                score = 0.7
            # 反方向子串匹配
            elif node.lower() in query_lower:
                score = 0.6
            # 部分单词匹配
            else:
                query_words = set(query_lower.split())
                node_words = set(node.lower().split())
                overlap = query_words & node_words
                if overlap:
                    score = len(overlap) / max(len(query_words), len(node_words)) * 0.5

            if score > 0:
                results.append((node, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_all_entities(self) -> list[dict]:
        """获取所有实体信息"""
        result: list[dict] = []
        for node, attrs in self._graph.nodes(data=True):
            result.append({"name": node, **attrs})
        return result

    def get_all_relations(self) -> list[dict]:
        """获取所有关系信息"""
        result: list[dict] = []
        for src, dst, attrs in self._graph.edges(data=True):
            result.append({"source": src, "target": dst, **attrs})
        return result

    # ── 持久化 ──────────────────────────────────────────

    def save(self) -> None:
        """保存图到 JSON 文件"""
        data = nx.node_link_data(self._graph, edges="links")
        with open(self._file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("图谱已保存: nodes=%d, edges=%d, path=%s",
                     self.node_count, self.edge_count, self._file_path)

    def _load(self) -> None:
        """从 JSON 文件加载图"""
        if not self._file_path.exists():
            logger.info("图谱文件不存在，初始化空图")
            return

        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._graph = nx.node_link_graph(
                data, edges="links", directed=True, multigraph=False
            )
            logger.info("图谱已加载: nodes=%d, edges=%d",
                         self.node_count, self.edge_count)
        except Exception as e:
            logger.warning("加载图谱失败: %s，使用空图", e)
            self._graph = nx.DiGraph()

    def clear(self) -> None:
        """清空图谱"""
        self._graph.clear()
        logger.info("图谱已清空")
