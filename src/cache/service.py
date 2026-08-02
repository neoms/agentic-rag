"""多级缓存服务 - 精准缓存 + 语义缓存编排

职责：
- 查询规范化 + 策略配置签名生成（缓存键）
- lookup：精准 → 语义；未命中时返回已计算的问题向量供检索复用
- store：图结束后写回答案、来源、路径、引文、幻觉结果
- replay：缓存命中时构造与真实链路一致的 SSE 事件序列
"""

import hashlib
import json
import logging
import time
import unicodedata

from src.cache.storage import CacheStorage
from src.config.settings import settings
from src.models.chat import AgenticChatRequest, StreamEvent

logger = logging.getLogger(__name__)

# 参与缓存键的模型名（切换模型即自动失效）
CACHE_MODEL_KEYS = (
    "llm_model",
    "llm_model_fast",
    "llm_model_strong",
    "embedding_model",
    "rerank_model",
)

# 缓存命中时答案的 token 分块大小（字符）
REPLAY_CHUNK_SIZE = 30


def normalize_query(query: str) -> str:
    """规范化查询：NFKC（全角→半角）+ 小写 + 折叠空白"""
    return " ".join(unicodedata.normalize("NFKC", query).lower().split())


def build_config_signature(request: AgenticChatRequest) -> str:
    """策略配置签名：8 个策略开关 + 模型名 → 排序哈希"""
    flags = {
        "web_search": request.enable_web_search,
        "reflection": request.enable_reflection,
        "rerank": request.enable_rerank,
        "grade_documents": request.enable_grade_documents,
        "transform_query": request.enable_transform_query,
        "bm25": request.enable_bm25,
        "multi_query": request.enable_multi_query,
        "kg": request.enable_kg,
    }
    parts = [f"{k}={v}" for k, v in sorted(flags.items())]
    for key in CACHE_MODEL_KEYS:
        parts.append(f"{key}={getattr(settings, key)}")
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class CacheService:
    """多级缓存编排服务"""

    def __init__(self):
        self._storage = CacheStorage(
            db_path=settings.cache_db_path_abs,
            max_entries=settings.cache_max_entries,
            ttl_seconds=settings.cache_ttl_seconds,
        )
        logger.info(
            "CacheService 初始化: threshold=%.2f, exact=%s, semantic=%s",
            settings.cache_semantic_threshold,
            settings.cache_exact_enabled,
            settings.cache_semantic_enabled,
        )

    def lookup(
        self,
        query: str,
        signature: str,
    ) -> tuple[dict | None, list[float] | None, dict]:
        """先精准后语义。

        Returns:
            (命中条目, 问题向量, 查询信息)
            - 命中：条目非 None，向量为 None
            - 未命中：条目为 None；若语义层已计算向量则返回该向量供检索复用
            - 语义命中：条目非 None，且向量非 None（供写回精准缓存复用）
            - 查询信息: {cache_type, similarity, query_norm, exact_checked,
              exact_hit, exact_ms, semantic_checked, semantic_hit, semantic_ms}
              （供前端分别展示两层状态与耗时）
        """
        info: dict = {
            "cache_type": "none",
            "similarity": None,
            "query_norm": normalize_query(query),
            "exact_checked": False,
            "exact_hit": False,
            "exact_ms": None,
            "semantic_checked": False,
            "semantic_hit": False,
            "semantic_ms": None,
        }
        if not settings.cache_enabled:
            return None, None, info
        query_norm = info["query_norm"]

        # ── 第 1 层：精准缓存 ──
        if settings.cache_exact_enabled:
            info["exact_checked"] = True
            t_exact = time.perf_counter()
            entry = self._storage.get_exact(query_norm, signature)
            info["exact_ms"] = round((time.perf_counter() - t_exact) * 1000, 1)
            if entry is not None:
                info["cache_type"] = "exact"
                info["similarity"] = 1.0
                info["exact_hit"] = True
                logger.info("[cache] 精准命中: norm='%s', hit_count=%d",
                            query_norm, entry["hit_count"])
                return entry, None, info

        # ── 第 2 层：语义缓存（需要问题向量） ──
        if not settings.cache_semantic_enabled:
            return None, None, info
        t_semantic = time.perf_counter()
        try:
            from src.backend.embedding import get_embedding_client
            vector = get_embedding_client().embed_query(query)
        except Exception as e:
            logger.warning("[cache] 语义向量计算失败，跳过语义层: %s", e)
            info["semantic_ms"] = round((time.perf_counter() - t_semantic) * 1000, 1)
            return None, None, info

        candidates = self._storage.semantic_search(
            vector,
            signature,
            top_k=1,
            min_similarity=settings.cache_semantic_threshold,
        )
        info["semantic_checked"] = True
        info["semantic_ms"] = round((time.perf_counter() - t_semantic) * 1000, 1)
        if candidates:
            best = candidates[0]
            info["similarity"] = best["similarity"]
            entry = self._storage.get_by_id(best["id"])
            if entry is not None:
                info["cache_type"] = "semantic"
                info["semantic_hit"] = True
                logger.info("[cache] 语义命中: sim=%.4f, hit_count=%d",
                            best["similarity"], entry["hit_count"])
                return entry, vector, info
            logger.info("[cache] 语义候选命中但条目已失效: id=%d", best["id"])

        best_sim = (
            f"{candidates[0]['similarity']:.4f}" if candidates else "n/a"
        )
        logger.info("[cache] 未命中: norm='%s', best_sim=%s", query_norm, best_sim)
        return None, vector, info

    def store(
        self,
        *,
        query: str,
        signature: str,
        vector: list[float] | None,
        answer: str,
        sources: list,
        agent_path: list,
        citations: dict | None = None,
        hallucination: dict | None = None,
    ) -> None:
        """写回缓存；向量缺失时（缓存被绕过）补算一次"""
        if not settings.cache_enabled:
            return
        query_norm = normalize_query(query)
        if vector is None:
            try:
                from src.backend.embedding import get_embedding_client
                vector = get_embedding_client().embed_query(query)
            except Exception as e:
                logger.warning("[cache] 写回时向量计算失败，跳过缓存: %s", e)
                return
        try:
            self._storage.upsert(
                query=query,
                query_norm=query_norm,
                vector=vector,
                signature=signature,
                answer=answer,
                sources=sources,
                agent_path=agent_path,
                citations=self._truncate_citations(citations),
                hallucination=hallucination,
            )
            logger.info("[cache] 写回成功: norm='%s', answer_len=%d",
                        query_norm, len(answer))
        except Exception as e:
            logger.warning("[cache] 写回失败: %s", e)

    @staticmethod
    def _truncate_citations(citations: dict | None) -> dict:
        """截断引文段落文本，控制缓存 DB 体积（前端引文弹窗为预览用途）"""
        if not citations:
            return {}
        max_chars = settings.cache_citation_max_chars
        truncated: dict = {}
        for key, item in citations.items():
            copy_item = dict(item)
            para = copy_item.get("paragraph_text")
            if (
                isinstance(para, str)
                and max_chars > 0
                and len(para) > max_chars
            ):
                copy_item["paragraph_text"] = para[:max_chars] + "…"
            truncated[key] = copy_item
        return truncated

    def invalidate_documents(self, doc_ids: list[str]) -> int:
        """删除文档后精确失效相关缓存条目（精准缓存 + 语义缓存一并清除）

        只清除 sources 中引用了指定 doc_id 的条目，其他条目不受影响。
        """
        if not settings.cache_enabled:
            return 0
        ids = [d for d in (doc_ids or []) if d]
        if not ids:
            return 0
        count = self._storage.invalidate_by_doc_ids(ids)
        if count:
            logger.info("[cache] 按文档失效缓存: %d 条, doc_ids=%s", count, ids)
        return count

    def replay(self, entry: dict, path: list[str]) -> list[StreamEvent]:
        """构造缓存命中的 SSE 回放事件

        顺序：token → source → citations → hallucination → path。
        其中 path 由调用方传入（本次请求的缓存专属路径，而非存储的原始路径）；
        node_start/node_step/node_data/done 由 rag_service 在调用方补发。
        """
        events: list[StreamEvent] = []
        answer = entry["answer"]
        for i in range(0, len(answer), REPLAY_CHUNK_SIZE):
            events.append(StreamEvent(event="token", data=answer[i:i + REPLAY_CHUNK_SIZE]))
        events.append(StreamEvent(
            event="source",
            data=json.dumps(entry["sources"], ensure_ascii=False),
        ))
        citations = entry.get("citations") or {}
        if citations:
            events.append(StreamEvent(
                event="citations",
                data=json.dumps(citations, ensure_ascii=False),
            ))
        if entry.get("hallucination") is not None:
            events.append(StreamEvent(
                event="hallucination",
                data=json.dumps(entry["hallucination"], ensure_ascii=False),
            ))
        events.append(StreamEvent(
            event="path",
            data=json.dumps(path, ensure_ascii=False),
        ))
        return events

    def stats(self) -> dict:
        return self._storage.stats()

    def clear(self) -> None:
        self._storage.clear()

    def close(self) -> None:
        """关闭缓存存储连接（服务关闭时调用）"""
        self._storage.close()
