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
    ) -> tuple[dict | None, list[float] | None]:
        """先精准后语义。

        Returns:
            (命中条目, 问题向量)
            - 命中：条目非 None，向量为 None
            - 未命中：条目为 None；若语义层已计算向量则返回该向量供检索复用
        """
        if not settings.cache_enabled:
            return None, None

        query_norm = normalize_query(query)

        # ── 第 1 层：精准缓存 ──
        if settings.cache_exact_enabled:
            entry = self._storage.get_exact(query_norm, signature)
            if entry is not None:
                logger.info("[cache] 精准命中: norm='%s', hit_count=%d",
                            query_norm, entry["hit_count"])
                return entry, None

        # ── 第 2 层：语义缓存（需要问题向量） ──
        if not settings.cache_semantic_enabled:
            return None, None
        try:
            from src.backend.embedding import get_embedding_client
            vector = get_embedding_client().embed_query(query)
        except Exception as e:
            logger.warning("[cache] 语义向量计算失败，跳过语义层: %s", e)
            return None, None

        candidates = self._storage.semantic_search(
            vector,
            signature,
            top_k=1,
            min_similarity=settings.cache_semantic_threshold,
        )
        if candidates:
            best = candidates[0]
            entry = self._storage.get_by_id(best["id"])
            if entry is not None:
                logger.info("[cache] 语义命中: sim=%.4f, hit_count=%d",
                            best["similarity"], entry["hit_count"])
                return entry, None
            logger.info("[cache] 语义候选命中但条目已失效: id=%d", best["id"])

        best_sim = (
            f"{candidates[0]['similarity']:.4f}" if candidates else "n/a"
        )
        logger.info("[cache] 未命中: norm='%s', best_sim=%s", query_norm, best_sim)
        return None, vector

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
                citations=citations or {},
                hallucination=hallucination,
            )
            logger.info("[cache] 写回成功: norm='%s', answer_len=%d",
                        query_norm, len(answer))
        except Exception as e:
            logger.warning("[cache] 写回失败: %s", e)

    def replay(self, entry: dict) -> list[StreamEvent]:
        """构造缓存命中的 SSE 事件序列

        顺序与真实链路一致：token → source → path → citations → hallucination
        → node_data → done。node_start/node_step 动画事件不回放，流程图按
        存储的 agent_path 显示完成节点。
        """
        events: list[StreamEvent] = []
        answer = entry["answer"]
        for i in range(0, len(answer), REPLAY_CHUNK_SIZE):
            events.append(StreamEvent(event="token", data=answer[i:i + REPLAY_CHUNK_SIZE]))
        events.append(StreamEvent(
            event="source",
            data=json.dumps(entry["sources"], ensure_ascii=False),
        ))
        events.append(StreamEvent(
            event="path",
            data=json.dumps(entry["agent_path"], ensure_ascii=False),
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
        events.append(StreamEvent(event="node_data", data="{}"))
        events.append(StreamEvent(event="done", data=""))
        return events

    def stats(self) -> dict:
        return self._storage.stats()

    def clear(self) -> None:
        self._storage.clear()
