"""GraphBuilder - LLM 驱动的实体关系抽取与知识图谱构建

从文档分块中抽取实体和关系，写入 Kuzu 图数据库。
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
        3. 全量收集后调用 store.bulk_import() 原子化写入（单次 save）
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

        # 3. 收集全量数据，单次原子化批量导入
        all_entities: list[dict[str, Any]] = []
        all_relations: list[dict[str, Any]] = []

        for batch_idx in sorted(results.keys()):
            result = results[batch_idx]
            for ent in result.get("entities", []):
                if ent.get("name", "").strip():
                    all_entities.append(ent)
            for rel in result.get("relations", []):
                src = rel.get("source", "").strip()
                dst = rel.get("target", "").strip()
                if src and dst:
                    all_relations.append(rel)

        entity_count, relation_count = store.bulk_import(
            doc_id=doc_id,
            entities=all_entities,
            relations=all_relations,
        )
        new_nodes = store.node_count - store_before
        logger.info(
            "图谱构建完成: 新增 %d 实体, %d 关系, 文档=%s",
            entity_count, relation_count, doc_id,
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
