"""GraphBuilder - LLM 驱动的实体关系抽取与知识图谱构建

从文档分块中抽取实体和关系，构建 NetworkX 图。
"""

import json
import logging
from typing import Any

from langchain_core.documents import Document

from src.agent.prompts import (
    KG_ENTITY_EXTRACT_SYSTEM,
    KG_ENTITY_EXTRACT_USER,
)
from src.backend.llm import create_fast_llm
from src.knowledge_graph.graph_store import GraphStore

logger = logging.getLogger(__name__)


class GraphBuilder:
    """使用 LLM 从文档中抽取实体关系，构建知识图谱

    流程:
        1. 对每个文档分块调用 LLM 抽取实体关系
        2. 解析 JSON 结果
        3. 写入 GraphStore
        4. 持久化到 JSON
    """

    def build_from_chunks(
        self,
        chunks: list[Document],
        doc_id: str,
        store: GraphStore,
        batch_size: int = 5,
    ) -> int:
        """从文档分块构建知识图谱

        Args:
            chunks: 文档分块列表
            doc_id: 文档 ID
            store: 图存储实例
            batch_size: 每次 LLM 调用的分块数量（合并上下文）

        Returns:
            抽取到的实体数量
        """
        if not chunks:
            logger.info("无分块，跳过知识图谱构建")
            return 0

        llm = create_fast_llm()
        total_entities = 0
        total_relations = 0
        store_before = store.node_count

        # 将 chunks 批量合并以减少 LLM 调用次数
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            combined_text = "\n\n---\n\n".join(
                doc.page_content for doc in batch
            )

            # 截断过长的文本（保守估计 4000 字符，避免 token 超限）
            if len(combined_text) > 4000:
                combined_text = combined_text[:4000] + "..."

            try:
                result = self._extract_entities(llm, combined_text)
                entities = result.get("entities", [])
                relations = result.get("relations", [])

                for ent in entities:
                    name = ent.get("name", "").strip()
                    if not name:
                        continue
                    store.add_entity(
                        name=name,
                        entity_type=ent.get("type", "unknown"),
                        doc_id=doc_id,
                        aliases=ent.get("aliases", []),
                        metadata=ent.get("metadata", {}),
                    )
                    total_entities += 1

                for rel in relations:
                    src = rel.get("source", "").strip()
                    dst = rel.get("target", "").strip()
                    rel_type = rel.get("relation", "related_to").strip()
                    if not src or not dst:
                        continue
                    # 确保源和目标实体存在
                    if src not in store.graph:
                        store.add_entity(name=src, entity_type="unknown", doc_id=doc_id)
                    if dst not in store.graph:
                        store.add_entity(name=dst, entity_type="unknown", doc_id=doc_id)
                    store.add_relation(
                        source=src,
                        target=dst,
                        relation=rel_type,
                        doc_id=doc_id,
                    )
                    total_relations += 1

                logger.debug(
                    "批次 %d/%d: 抽取 %d 实体, %d 关系",
                    i // batch_size + 1,
                    (len(chunks) + batch_size - 1) // batch_size,
                    len(entities),
                    len(relations),
                )

            except Exception as e:
                logger.warning("批次 %d 实体抽取失败: %s", i // batch_size + 1, e)
                continue

        store.save()
        new_nodes = store.node_count - store_before
        logger.info(
            "图谱构建完成: 新增 %d 实体, %d 关系, 文档=%s",
            total_entities, total_relations, doc_id,
        )
        return new_nodes

    def _extract_entities(self, llm, text: str) -> dict[str, Any]:
        """调用 LLM 抽取实体关系

        Returns:
            {"entities": [...], "relations": [...]}
        """
        prompt = f"{KG_ENTITY_EXTRACT_SYSTEM}\n\n{KG_ENTITY_EXTRACT_USER.format(text=text)}"
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # 尝试从 LLM 返回中提取 JSON
        return self._parse_llm_json(raw)

    @staticmethod
    def _parse_llm_json(raw: str) -> dict[str, Any]:
        """从 LLM 返回文本中解析 JSON"""
        # 尝试直接解析
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # 尝试提取 JSON 块
        import re
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        logger.warning("无法解析 LLM 实体抽取结果: %s...", raw[:200])
        return {"entities": [], "relations": []}
