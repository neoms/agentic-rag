"""GraphBuilder - LLM 驱动的实体关系抽取与知识图谱构建

从文档分块中抽取实体和关系，构建 NetworkX 图。
批次间使用 ThreadPoolExecutor 并发执行 LLM 调用以提升速度。
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.documents import Document

from src.agent.prompts import (
    KG_ENTITY_EXTRACT_SYSTEM,
    KG_ENTITY_EXTRACT_USER,
)
from src.backend.llm import create_fast_llm
from src.config.settings import settings
from src.knowledge_graph.graph_store import GraphStore

logger = logging.getLogger(__name__)


class GraphBuilder:
    """使用 LLM 从文档中抽取实体关系，构建知识图谱

    流程:
        1. 将文档分块按 batch_size 分组
        2. 多线程并发调用 LLM 抽取实体关系
        3. 串行写入 GraphStore（避免线程安全问题）
        4. 持久化到 JSON
    """

    def build_from_chunks(
        self,
        chunks: list[Document],
        doc_id: str,
        store: GraphStore,
        batch_size: int = 5,
        max_workers: int | None = None,
    ) -> int:
        """从文档分块构建知识图谱

        将分块分批后通过线程池并发调用 LLM 抽取实体关系，
        所有结果收集完成后串行写入 GraphStore。

        Args:
            chunks: 文档分块列表
            doc_id: 文档 ID
            store: 图存储实例
            batch_size: 每次 LLM 调用的分块数量（合并上下文）
            max_workers: 并发线程数，默认 3

        Returns:
            抽取到的实体数量
        """
        if not chunks:
            logger.info("无分块，跳过知识图谱构建")
            return 0

        if max_workers is None:
            max_workers = getattr(settings, "kg_max_concurrency", 3)

        total_entities = 0
        total_relations = 0
        store_before = store.node_count

        # 1. 准备批次数据
        batches: list[tuple[int, str]] = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            combined_text = "\n\n---\n\n".join(
                doc.page_content for doc in batch
            )
            # 截断过长的文本（保守估计 4000 字符，避免 token 超限）
            if len(combined_text) > 4000:
                combined_text = combined_text[:4000] + "..."
            batches.append((i // batch_size, combined_text))

        total_batches = len(batches)
        logger.info(
            "开始并发抽取 KG 实体: %d 批次, max_workers=%d, doc_id=%s",
            total_batches, max_workers, doc_id,
        )

        # 2. 并发执行 LLM 抽取（每批次独立 LLM 实例，避免线程安全问题）
        results: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self._extract_entities, create_fast_llm(), text): idx
                for idx, text in batches
            }
            for future in as_completed(future_map):
                batch_idx = future_map[future]
                try:
                    result = future.result()
                    results[batch_idx] = result
                    entities = result.get("entities", [])
                    relations = result.get("relations", [])
                    logger.debug(
                        "批次 %d/%d: 抽取 %d 实体, %d 关系",
                        batch_idx + 1, total_batches,
                        len(entities), len(relations),
                    )
                except Exception as e:
                    logger.warning("批次 %d/%d 实体抽取失败: %s",
                                   batch_idx + 1, total_batches, e)
                    results[batch_idx] = {"entities": [], "relations": []}

        # 3. 串行写入 GraphStore（按批次顺序）
        for batch_idx in sorted(results.keys()):
            result = results[batch_idx]
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

        return self._parse_llm_json(raw)

    @staticmethod
    def _parse_llm_json(raw: str) -> dict[str, Any]:
        """从 LLM 返回文本中解析 JSON"""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        logger.warning("无法解析 LLM 实体抽取结果: %s...", raw[:200])
        return {"entities": [], "relations": []}
