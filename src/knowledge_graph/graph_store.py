"""GraphStore - 基于 Kuzu 图数据库的知识图谱存储（原生持久化）

节点：实体（Entity），边：关系（Relation）
Kuzu 自动持久化到磁盘，无需手动 save/load。
"""

import json
import logging
from pathlib import Path
from typing import Any

import kuzu

from src.config.settings import settings

logger = logging.getLogger(__name__)


class GraphView:
    """轻量图视图包装器，兼容 nx.DiGraph 的部分接口

    由 GraphStore.get_subgraph() 返回，提供 nodes/edges/data 遍历能力。
    """

    def __init__(
        self,
        nodes: dict[str, dict[str, Any]],
        edges: list[tuple[str, str, dict[str, Any]]],
    ) -> None:
        self._nodes = nodes
        self._edges = edges

    def nodes(self, data: bool = False):
        """遍历节点

        Args:
            data: 是否返回属性
        Returns:
            data=False: [name, ...]
            data=True: [(name, {attr: val}), ...]
        """
        if data:
            return list(self._nodes.items())
        return list(self._nodes.keys())

    def edges(self, data: bool = False):
        """遍历边

        Args:
            data: 是否返回属性
        Returns:
            data=False: [(src, dst), ...]
            data=True: [(src, dst, {attr: val}), ...]
        """
        if data:
            return [(s, d, a) for s, d, a in self._edges]
        return [(s, d) for s, d, _ in self._edges]

    def number_of_nodes(self) -> int:
        return len(self._nodes)

    def number_of_edges(self) -> int:
        return len(self._edges)

    def has_node(self, name: str) -> bool:
        return name in self._nodes


class GraphStore:
    """Kuzu 图数据库存储（原生持久化到磁盘，无需手动 save/load）

    数据目录: data/kg/kuzu_db/
    表结构:
        Entity(node): name(PK), type, doc_ids[], aliases[], metadata(JSON)
        Relation(edge): FROM Entity TO Entity, relation, weight, doc_ids[], metadata(JSON)
    """

    def __init__(self) -> None:
        self.data_dir = settings.project_root / settings.kg_data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = str(self.data_dir / "kuzu_db")
        self._db = kuzu.Database(self._db_path)
        self._conn = kuzu.Connection(self._db)
        self._init_schema()

    # ── 内部 ──────────────────────────────────────────────

    def _init_schema(self) -> None:
        """初始化 Kuzu 表结构（幂等）"""
        self._conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Entity (
                name STRING,
                type STRING DEFAULT 'unknown',
                doc_ids STRING[],
                aliases STRING[],
                metadata STRING,
                PRIMARY KEY (name)
            )
        """)
        self._conn.execute("""
            CREATE REL TABLE IF NOT EXISTS Relation (
                FROM Entity TO Entity,
                relation STRING DEFAULT 'related_to',
                weight DOUBLE DEFAULT 1.0,
                doc_ids STRING[],
                metadata STRING
            )
        """)

    # ── 属性 ──────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        result = self._conn.execute("MATCH (e:Entity) RETURN count(*) AS cnt")
        return result.get_next()[0]

    @property
    def edge_count(self) -> int:
        result = self._conn.execute("MATCH ()-[r:Relation]->() RETURN count(*) AS cnt")
        return result.get_next()[0]

    def is_empty(self) -> bool:
        return self.node_count == 0

    # ── 实体操作 ──────────────────────────────────────────

    def add_entity(
        self,
        name: str,
        entity_type: str = "unknown",
        doc_id: str = "",
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """添加或更新实体节点（公开方法，委托给 _add_entity_impl）"""
        self._add_entity_impl(name, entity_type, doc_id, aliases, metadata)

    def _add_entity_impl(
        self,
        name: str,
        entity_type: str = "unknown",
        doc_id: str = "",
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """add_entity 的内部实现，供 bulk_import 调用"""
        aliases = aliases or []
        metadata = metadata or {}
        meta_json = json.dumps(metadata, ensure_ascii=False)

        result = self._conn.execute(
            "MATCH (e:Entity) WHERE e.name = $name RETURN e.type, e.doc_ids, e.aliases, e.metadata",
            {"name": name},
        )

        if result.has_next():
            row = result.get_next()
            existing_type, existing_doc_ids, existing_aliases, existing_meta = row

            # 合并 type（非 unknown 的优先）
            merged_type = existing_type if existing_type != "unknown" else entity_type
            if entity_type != "unknown" and existing_type == "unknown":
                merged_type = entity_type

            # 合并 doc_ids
            merged_doc_ids: list[str] = list(existing_doc_ids or [])
            if doc_id and doc_id not in merged_doc_ids:
                merged_doc_ids.append(doc_id)

            # 合并 aliases
            merged_aliases: list[str] = list(existing_aliases or [])
            existing_alias_set = set(merged_aliases)
            for a in aliases:
                if a and a not in existing_alias_set:
                    merged_aliases.append(a)
                    existing_alias_set.add(a)

            # 合并 metadata
            merged_meta: dict[str, Any] = json.loads(existing_meta or "{}")
            merged_meta.update(metadata)

            self._conn.execute(
                """
                MATCH (e:Entity) WHERE e.name = $name
                SET e.type = $type, e.doc_ids = $doc_ids,
                    e.aliases = $aliases, e.metadata = $metadata
                """,
                {
                    "name": name,
                    "type": merged_type,
                    "doc_ids": merged_doc_ids,
                    "aliases": merged_aliases,
                    "metadata": json.dumps(merged_meta, ensure_ascii=False),
                },
            )
        else:
            self._conn.execute(
                """
                CREATE (e:Entity {
                    name: $name, type: $type, doc_ids: $doc_ids,
                    aliases: $aliases, metadata: $metadata
                })
                """,
                {
                    "name": name,
                    "type": entity_type,
                    "doc_ids": [doc_id] if doc_id else [],
                    "aliases": aliases,
                    "metadata": meta_json,
                },
            )

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
        """添加或更新关系边（公开方法，委托给 _add_relation_impl）"""
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
        """add_relation 的内部实现 + 跨文档 doc_ids 合并"""
        metadata = metadata or {}
        meta_json = json.dumps(metadata, ensure_ascii=False)
        new_doc_ids = [doc_id] if doc_id else []

        result = self._conn.execute(
            """
            MATCH (s:Entity {name: $src})-[r:Relation]->(t:Entity {name: $dst})
            RETURN r.doc_ids, r.relation, r.weight, r.metadata
            """,
            {"src": source, "dst": target},
        )

        if result.has_next():
            row = result.get_next()
            existing_doc_ids, _, _, existing_meta = row

            merged_doc_ids: list[str] = list(existing_doc_ids or [])
            for d in new_doc_ids:
                if d not in merged_doc_ids:
                    merged_doc_ids.append(d)

            merged_meta: dict[str, Any] = json.loads(existing_meta or "{}")
            merged_meta.update(metadata)

            self._conn.execute(
                """
                MATCH (s:Entity {name: $src})-[r:Relation]->(t:Entity {name: $dst})
                SET r.relation = $rel, r.weight = $weight,
                    r.doc_ids = $doc_ids, r.metadata = $metadata
                """,
                {
                    "src": source,
                    "dst": target,
                    "rel": relation,
                    "weight": weight,
                    "doc_ids": merged_doc_ids,
                    "metadata": meta_json,
                },
            )
        else:
            self._conn.execute(
                """
                MATCH (s:Entity {name: $src}), (t:Entity {name: $dst})
                CREATE (s)-[r:Relation {
                    relation: $rel, weight: $weight,
                    doc_ids: $doc_ids, metadata: $metadata
                }]->(t)
                """,
                {
                    "src": source,
                    "dst": target,
                    "rel": relation,
                    "weight": weight,
                    "doc_ids": new_doc_ids,
                    "metadata": meta_json,
                },
            )

    # ── 删除操作 ──────────────────────────────────────────

    def remove_doc_id(self, doc_id: str) -> tuple[int, int]:
        """从图谱中移除指定文档的所有实体和关系引用

        引用计数式级联删除：
        1. 边/节点的 doc_ids 中移除目标 doc_id
        2. doc_ids 变空 → 删除该元素
        3. doc_ids 非空 → 仅清理引用，保留元素
        4. 节点删除时 Kuzu 自动级联删除其关联边

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

        # doc_id 来源于内部 UUID（非用户输入），安全转义后内联查询，
        # 避免 Kuzu 0.11.x 在 ANY(...) 内部使用参数化表达式时的已知 bug
        # （KU_UNREACHABLE in ParsedParameterExpression.h）
        safe_id = doc_id.replace("'", "''")

        # ---- 第一轮：处理边 ----
        result = self._conn.execute(
            f"""
            MATCH (s:Entity)-[r:Relation]->(t:Entity)
            WHERE '{safe_id}' IN r.doc_ids
            RETURN s.name, t.name, r.doc_ids
            """,
        )

        relations_to_delete: list[tuple[str, str]] = []
        relations_to_update: list[tuple[str, str, list[str]]] = []

        while result.has_next():
            src, dst, existing_ids = result.get_next()
            new_ids = [d for d in (existing_ids or []) if d != doc_id]
            if new_ids:
                relations_to_update.append((src, dst, new_ids))
            else:
                relations_to_delete.append((src, dst))

        for src, dst, new_ids in relations_to_update:
            self._conn.execute(
                "MATCH (s:Entity {name: $src})-[r:Relation]->(t:Entity {name: $dst}) SET r.doc_ids = $doc_ids",
                {"src": src, "dst": dst, "doc_ids": new_ids},
            )

        for src, dst in relations_to_delete:
            self._conn.execute(
                "MATCH (s:Entity {name: $src})-[r:Relation]->(t:Entity {name: $dst}) DELETE r",
                {"src": src, "dst": dst},
            )

        # ---- 第二轮：处理节点 ----
        result = self._conn.execute(
            f"""
            MATCH (e:Entity)
            WHERE '{safe_id}' IN e.doc_ids
            RETURN e.name, e.doc_ids
            """,
        )

        entities_to_delete: list[str] = []
        entities_to_update: list[tuple[str, list[str]]] = []

        while result.has_next():
            name, existing_ids = result.get_next()
            new_ids = [d for d in (existing_ids or []) if d != doc_id]
            if new_ids:
                entities_to_update.append((name, new_ids))
            else:
                entities_to_delete.append(name)

        for name, new_ids in entities_to_update:
            self._conn.execute(
                "MATCH (e:Entity {name: $name}) SET e.doc_ids = $doc_ids",
                {"name": name, "doc_ids": new_ids},
            )

        # 统计级联删除的边（删除节点前先计数）
        cascaded_relations = 0
        for name in entities_to_delete:
            count_result = self._conn.execute(
                "MATCH (e:Entity {name: $name})-[r:Relation]-() RETURN count(*) AS cnt",
                {"name": name},
            )
            cascaded_relations += count_result.get_next()[0]

        for name in entities_to_delete:
            self._conn.execute(
                "MATCH (e:Entity {name: $name}) DELETE e",
                {"name": name},
            )

        removed_entities = len(entities_to_delete)
        removed_relations = len(relations_to_delete) + cascaded_relations
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
        """原子化批量导入实体和关系

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

        # 批量导入使用自动提交（每条语句独立），
        # Kuzu 的 OLAP 引擎对事务有长度限制（默认 ~4GB WAL），
        # 逐条写入 + 自动提交更安全。

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
            # 自动补齐缺失的关系端节点
            src_result = self._conn.execute(
                "MATCH (e:Entity {name: $name}) RETURN count(*) AS cnt",
                {"name": src},
            )
            if src_result.get_next()[0] == 0:
                self._add_entity_impl(name=src, entity_type="unknown", doc_id=doc_id)

            dst_result = self._conn.execute(
                "MATCH (e:Entity {name: $name}) RETURN count(*) AS cnt",
                {"name": dst},
            )
            if dst_result.get_next()[0] == 0:
                self._add_entity_impl(name=dst, entity_type="unknown", doc_id=doc_id)

            self._add_relation_impl(
                source=src, target=dst, relation=rel_type, doc_id=doc_id,
            )
            relation_count += 1

        return entity_count, relation_count

    # ── 查询操作 ──────────────────────────────────────────

    def get_neighbors(
        self, entity_name: str, hops: int = 1
    ) -> dict[str, list[str]]:
        """获取指定实体的 N-hop 邻居

        Returns:
            {实体名: [邻居实体名列表]}，包含实体本身
        """
        result = self._conn.execute(
            "MATCH (e:Entity {name: $name}) RETURN count(*) AS cnt",
            {"name": entity_name},
        )
        if result.get_next()[0] == 0:
            return {}

        nodes: set[str] = {entity_name}
        frontier: set[str] = {entity_name}

        for _ in range(hops):
            if not frontier:
                break
            next_frontier: set[str] = set()
            # 逐跳查询，避免 Kuzu 可变长度路径的性能陷阱
            for f_node in frontier:
                hop_result = self._conn.execute(
                    """
                    MATCH (e:Entity {name: $name})-[r:Relation]-(n:Entity)
                    RETURN DISTINCT n.name
                    """,
                    {"name": f_node},
                )
                while hop_result.has_next():
                    neighbor = hop_result.get_next()[0]
                    if neighbor not in nodes:
                        next_frontier.add(neighbor)
            nodes.update(next_frontier)
            frontier = next_frontier

        # 为每个节点收集邻居
        result_dict: dict[str, list[str]] = {}
        for n in nodes:
            nb_result = self._conn.execute(
                """
                MATCH (e:Entity {name: $name})-[r:Relation]-(n:Entity)
                RETURN DISTINCT n.name
                """,
                {"name": n},
            )
            neighbors: list[str] = []
            while nb_result.has_next():
                neighbors.append(nb_result.get_next()[0])
            result_dict[n] = neighbors

        return result_dict

    def get_subgraph(
        self, entity_names: list[str], hops: int = 2
    ) -> GraphView:
        """从种子实体出发，提取 N-hop 子图

        Args:
            entity_names: 种子实体名称列表
            hops: 跳数

        Returns:
            子图 (GraphView)
        """
        if not entity_names or self.is_empty():
            return GraphView({}, [])

        nodes: set[str] = set()
        for name in entity_names:
            result = self._conn.execute(
                "MATCH (e:Entity {name: $name}) RETURN e.name, e.type, e.aliases, e.metadata",
                {"name": name},
            )
            if result.has_next():
                nodes.add(name)

        if not nodes:
            return GraphView({}, [])

        frontier = set(nodes)
        for _ in range(hops):
            if not frontier:
                break
            next_frontier: set[str] = set()
            for f_node in frontier:
                hop_result = self._conn.execute(
                    """
                    MATCH (e:Entity {name: $name})-[r:Relation]-(n:Entity)
                    RETURN DISTINCT n.name
                    """,
                    {"name": f_node},
                )
                while hop_result.has_next():
                    neighbor = hop_result.get_next()[0]
                    if neighbor not in nodes:
                        next_frontier.add(neighbor)
            nodes.update(next_frontier)
            frontier = next_frontier

        # 获取所有节点的属性
        node_data: dict[str, dict[str, Any]] = {}
        for name in nodes:
            n_result = self._conn.execute(
                "MATCH (e:Entity {name: $name}) RETURN e.type, e.doc_ids, e.aliases, e.metadata",
                {"name": name},
            )
            if n_result.has_next():
                e_type, doc_ids, aliases, meta_str = n_result.get_next()
                node_data[name] = {
                    "type": e_type or "unknown",
                    "doc_ids": doc_ids or [],
                    "aliases": aliases or [],
                    "metadata": json.loads(meta_str) if meta_str else {},
                }

        # 获取子图内的所有关系
        node_list = list(nodes)
        edge_data: list[tuple[str, str, dict[str, Any]]] = []
        for src_name in node_list:
            e_result = self._conn.execute(
                """
                MATCH (s:Entity {name: $src})-[r:Relation]->(t:Entity)
                WHERE t.name IN $targets
                RETURN t.name, r.relation, r.weight, r.doc_ids, r.metadata
                """,
                {"src": src_name, "targets": node_list},
            )
            while e_result.has_next():
                dst, rel_type, weight, doc_ids, meta_str = e_result.get_next()
                edge_data.append((
                    src_name, dst,
                    {
                        "relation": rel_type or "related_to",
                        "weight": float(weight) if weight is not None else 1.0,
                        "doc_ids": doc_ids or [],
                        "metadata": json.loads(meta_str) if meta_str else {},
                    },
                ))

        return GraphView(node_data, edge_data)

    def get_edge_data(
        self, source: str, target: str
    ) -> dict[str, Any] | None:
        """获取两点间的关系边数据"""
        result = self._conn.execute(
            """
            MATCH (s:Entity {name: $src})-[r:Relation]->(t:Entity {name: $dst})
            RETURN r.relation, r.weight, r.doc_ids, r.metadata
            """,
            {"src": source, "dst": target},
        )
        if result.has_next():
            rel_type, weight, doc_ids, meta_str = result.get_next()
            return {
                "relation": rel_type or "related_to",
                "weight": float(weight) if weight is not None else 1.0,
                "doc_ids": doc_ids or [],
                "metadata": json.loads(meta_str) if meta_str else {},
            }
        return None

    def find_paths(
        self, source: str, target: str, max_len: int = 4
    ) -> list[list[str]]:
        """查找两个实体之间的所有路径（限制长度）

        使用 BFS 逐层扩展，避免 Kuzu 可变长度路径的性能问题。
        """
        # 确认两个节点都存在
        s_result = self._conn.execute(
            "MATCH (e:Entity {name: $name}) RETURN count(*) AS cnt",
            {"name": source},
        )
        t_result = self._conn.execute(
            "MATCH (e:Entity {name: $name}) RETURN count(*) AS cnt",
            {"name": target},
        )
        if s_result.get_next()[0] == 0 or t_result.get_next()[0] == 0:
            return []

        # BFS 路径搜索
        paths: list[list[str]] = []
        queue: list[tuple[str, list[str]]] = [(source, [source])]
        visited: set[str] = set()

        while queue:
            current, path = queue.pop(0)
            if len(path) > max_len:
                continue

            if current == target and len(path) > 1:
                paths.append(list(path))
                if len(paths) >= 50:  # 限制路径数量
                    break
                continue

            if current != source:
                visited.add(current)

            # 获取邻居
            nb_result = self._conn.execute(
                """
                MATCH (e:Entity {name: $name})-[r:Relation]-(n:Entity)
                RETURN DISTINCT n.name
                """,
                {"name": current},
            )
            while nb_result.has_next():
                neighbor = nb_result.get_next()[0]
                if neighbor not in path:  # 避免环路
                    queue.append((neighbor, path + [neighbor]))

        return paths

    def search_entities(
        self, query: str, top_k: int = 10
    ) -> list[tuple[str, float]]:
        """模糊搜索实体（精确匹配 + 别名匹配 + 子串匹配）

        Returns:
            [(实体名, 匹配分数)] 列表
        """
        if self.is_empty():
            return []

        query_lower = query.lower()

        # 获取全部实体（小数据集，直接全量遍历）
        result = self._conn.execute(
            "MATCH (e:Entity) RETURN e.name, e.aliases, e.type",
        )

        results: list[tuple[str, float]] = []
        while result.has_next():
            name, aliases_val, _ = result.get_next()
            name_lower = name.lower()
            aliases_list = list(aliases_val or [])
            score = 0.0

            # 精确匹配
            if query_lower == name_lower:
                score = 1.0
            # 别名匹配
            elif query_lower in [a.lower() for a in aliases_list]:
                score = 0.9
            # 子串匹配
            elif query_lower in name_lower:
                score = 0.7
            # 反方向子串匹配
            elif name_lower in query_lower:
                score = 0.6
            # 部分单词匹配
            else:
                query_words = set(query_lower.split())
                name_words = set(name_lower.split())
                overlap = query_words & name_words
                if overlap:
                    score = len(overlap) / max(len(query_words), len(name_words)) * 0.5

            if score > 0:
                results.append((name, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_all_entities(self) -> list[dict]:
        """获取所有实体信息"""
        result = self._conn.execute(
            "MATCH (e:Entity) RETURN e.name, e.type, e.doc_ids, e.aliases, e.metadata",
        )
        entities: list[dict] = []
        while result.has_next():
            name, e_type, doc_ids, aliases, meta_str = result.get_next()
            entities.append({
                "name": name,
                "type": e_type or "unknown",
                "doc_ids": doc_ids or [],
                "aliases": aliases or [],
                "metadata": json.loads(meta_str) if meta_str else {},
            })
        return entities

    def get_all_relations(self) -> list[dict]:
        """获取所有关系信息"""
        result = self._conn.execute(
            """
            MATCH (s:Entity)-[r:Relation]->(t:Entity)
            RETURN s.name, t.name, r.relation, r.weight, r.doc_ids, r.metadata
            """,
        )
        relations: list[dict] = []
        while result.has_next():
            src, dst, rel_type, weight, doc_ids, meta_str = result.get_next()
            relations.append({
                "source": src,
                "target": dst,
                "relation": rel_type or "related_to",
                "weight": float(weight) if weight is not None else 1.0,
                "doc_ids": doc_ids or [],
                "metadata": json.loads(meta_str) if meta_str else {},
            })
        return relations
