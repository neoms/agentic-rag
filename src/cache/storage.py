"""多级缓存存储层 - SQLite 持久化 + numpy 内存向量索引

零外部依赖：
- sqlite3（标准库，WAL 模式）持久化精准缓存条目与语义缓存条目
- numpy（已有依赖）内存余弦相似度检索，与知识图谱 NumpyVectorIndex 同款模式

语义索引维护策略：
- 写入时增量追加到内存数组（避免全量重建）
- 删除（LRU 淘汰 / TTL 过期）时标记失效，失效占比过半才触发一次全量重建
- 语义检索仅在与当前 config_signature 一致的条目中执行
"""

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class CacheStorage:
    """SQLite + numpy 缓存存储"""

    def __init__(
        self,
        db_path: Path,
        max_entries: int = 5000,
        ttl_seconds: int = 0,
    ):
        self._db_path = db_path
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._lock = threading.RLock()

        # 内存语义索引
        self._vectors: np.ndarray | None = None      # (N, D) float32 L2 归一化
        self._ids: np.ndarray = np.array([], dtype=np.int64)      # 行 id
        self._created: np.ndarray = np.array([], dtype=np.float64)  # created_at
        self._valid: np.ndarray = np.array([], dtype=bool)         # 有效性掩码
        self._sig_positions: dict[str, np.ndarray] = {}  # signature -> 行位置
        self._id_pos: dict[int, int] = {}               # 行 id -> 行位置

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()
        self._load_index()
        logger.info("CacheStorage 初始化: db=%s, entries=%d, max_entries=%d",
                    db_path, self._ids.size, max_entries)

    # ==================== 建表与索引加载 ====================

    def _create_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    query_norm TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    config_signature TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources TEXT NOT NULL,
                    agent_path TEXT NOT NULL,
                    citations TEXT NOT NULL,
                    hallucination TEXT,
                    created_at REAL NOT NULL,
                    last_access_at REAL NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 1,
                    UNIQUE(query_norm, config_signature)
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_signature "
                "ON cache_entries(config_signature)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_last_access "
                "ON cache_entries(last_access_at)"
            )
            self._conn.commit()

    def _load_index(self) -> None:
        """从数据库全量重建内存语义索引"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, vector, config_signature, created_at FROM cache_entries"
            ).fetchall()
            vectors: list[np.ndarray] = []
            ids: list[int] = []
            created: list[float] = []
            sig_positions: dict[str, list[int]] = {}
            for pos, row in enumerate(rows):
                vectors.append(np.frombuffer(row["vector"], dtype=np.float32))
                ids.append(row["id"])
                created.append(row["created_at"])
                sig_positions.setdefault(row["config_signature"], []).append(pos)

            if vectors:
                self._vectors = np.stack(vectors).astype(np.float32)
                self._ids = np.array(ids, dtype=np.int64)
                self._created = np.array(created, dtype=np.float64)
                self._valid = np.ones(len(ids), dtype=bool)
                self._sig_positions = {
                    sig: np.array(poss, dtype=np.int64)
                    for sig, poss in sig_positions.items()
                }
            else:
                self._vectors = None
                self._ids = np.array([], dtype=np.int64)
                self._created = np.array([], dtype=np.float64)
                self._valid = np.array([], dtype=bool)
                self._sig_positions = {}
            self._id_pos = {
                int(rid): pos for pos, rid in enumerate(self._ids)
            }

    # ==================== 精准缓存 ====================

    def get_exact(self, query_norm: str, signature: str) -> dict | None:
        """精准命中：按 (规范化查询, 配置签名) 精确匹配"""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM cache_entries "
                "WHERE query_norm = ? AND config_signature = ?",
                (query_norm, signature),
            ).fetchone()
            if row is None:
                return None
            if self._is_expired(row["created_at"]):
                self._delete_rows([row["id"]])
                return None
            self._touch(row["id"])
            return self._row_to_entry(row)

    # ==================== 语义缓存 ====================

    def semantic_search(
        self,
        vector: list[float],
        signature: str,
        top_k: int = 1,
        min_similarity: float = 0.92,
    ) -> list[dict]:
        """余弦相似度检索（仅限同一 config_signature 的条目）"""
        if self._vectors is None or self._ids.size == 0:
            return []
        positions = self._sig_positions.get(signature)
        if positions is None or positions.size == 0:
            return []
        valid_pos = positions[self._valid[positions]]
        if valid_pos.size == 0:
            return []

        v = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm

        sims = self._vectors[valid_pos] @ v
        if self._ttl_seconds > 0:
            now = time.time()
            ttl_ok = (now - self._created[valid_pos]) <= self._ttl_seconds
            valid_pos = valid_pos[ttl_ok]
            if valid_pos.size == 0:
                return []
            sims = self._vectors[valid_pos] @ v

        order = np.argsort(-sims)[:top_k]
        return [
            {
                "id": int(self._ids[valid_pos[idx]]),
                "similarity": float(sims[idx]),
            }
            for idx in order
            if float(sims[idx]) >= min_similarity
        ]

    def get_by_id(self, entry_id: int) -> dict | None:
        """按 id 获取条目（语义命中后取完整数据），命中即更新访问统计"""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM cache_entries WHERE id = ?", (entry_id,)
            ).fetchone()
            if row is None:
                return None
            if self._is_expired(row["created_at"]):
                self._delete_rows([row["id"]])
                return None
            self._touch(row["id"])
            return self._row_to_entry(row)

    # ==================== 写入与淘汰 ====================

    def upsert(
        self,
        *,
        query: str,
        query_norm: str,
        vector: list[float],
        signature: str,
        answer: str,
        sources: list,
        agent_path: list,
        citations: dict,
        hallucination: dict | None,
    ) -> None:
        """写回条目；同 (query_norm, signature) 已存在则更新负载并保留命中统计"""
        with self._lock:
            vec = np.asarray(vector, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vec_bytes = vec.astype(np.float32).tobytes()
            now = time.time()

            self._conn.execute(
                """
                INSERT INTO cache_entries
                    (query, query_norm, vector, config_signature, answer, sources,
                     agent_path, citations, hallucination, created_at, last_access_at, hit_count)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,1)
                ON CONFLICT(query_norm, config_signature) DO UPDATE SET
                    query = excluded.query,
                    vector = excluded.vector,
                    answer = excluded.answer,
                    sources = excluded.sources,
                    agent_path = excluded.agent_path,
                    citations = excluded.citations,
                    hallucination = excluded.hallucination,
                    last_access_at = excluded.last_access_at
                """,
                (
                    query,
                    query_norm,
                    vec_bytes,
                    signature,
                    answer,
                    json.dumps(sources, ensure_ascii=False),
                    json.dumps(agent_path, ensure_ascii=False),
                    json.dumps(citations, ensure_ascii=False),
                    json.dumps(hallucination, ensure_ascii=False)
                    if hallucination is not None else None,
                    now,
                    now,
                ),
            )
            self._conn.commit()

            # 同步内存索引
            row = self._conn.execute(
                "SELECT id, created_at FROM cache_entries "
                "WHERE query_norm = ? AND config_signature = ?",
                (query_norm, signature),
            ).fetchone()
            if row is None:
                return
            existing_pos = self._id_pos.get(row["id"])
            if existing_pos is not None:
                # 更新既有向量（保持原 created_at）
                if self._vectors is not None and self._vectors.shape[1] == vec.shape[0]:
                    self._vectors[existing_pos] = vec
                    self._valid[existing_pos] = True
                else:
                    self._load_index()
            else:
                self._append_to_index(row["id"], vec, signature, row["created_at"])

            self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """LRU 淘汰：超过上限时删除最久未访问的条目"""
        with self._lock:
            count = self._conn.execute(
                "SELECT COUNT(*) AS c FROM cache_entries"
            ).fetchone()["c"]
            if count <= self._max_entries:
                return
            to_delete = count - self._max_entries
            rows = self._conn.execute(
                "SELECT id FROM cache_entries "
                "ORDER BY last_access_at ASC LIMIT ?",
                (to_delete,),
            ).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                self._delete_rows(ids)
                logger.info("CacheStorage: LRU 淘汰 %d 条 (total=%d, max=%d)",
                            len(ids), count, self._max_entries)

    # ==================== 内部工具 ====================

    def _append_to_index(
        self,
        entry_id: int,
        vector: np.ndarray,
        signature: str,
        created_at: float,
    ) -> None:
        """增量追加一条向量到内存索引"""
        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if self._vectors is None:
            self._vectors = vec
            self._ids = np.array([entry_id], dtype=np.int64)
            self._created = np.array([created_at], dtype=np.float64)
            self._valid = np.array([True])
        else:
            if self._vectors.shape[1] != vec.shape[1]:
                # 维度不一致（切换了 embedding 模型）→ 全量重建后重试
                self._load_index()
                if self._vectors is None or self._vectors.shape[1] != vec.shape[1]:
                    self._vectors = vec
                    self._ids = np.array([entry_id], dtype=np.int64)
                    self._created = np.array([created_at], dtype=np.float64)
                    self._valid = np.array([True])
                    self._sig_positions = {signature: np.array([0], dtype=np.int64)}
                    self._id_pos = {entry_id: 0}
                    return
            self._vectors = np.vstack([self._vectors, vec])
            self._ids = np.append(self._ids, entry_id)
            self._created = np.append(self._created, created_at)
            self._valid = np.append(self._valid, True)
        pos = len(self._ids) - 1
        self._id_pos[entry_id] = pos
        self._sig_positions.setdefault(signature, np.array([], dtype=np.int64))
        self._sig_positions[signature] = np.append(
            self._sig_positions[signature], pos
        )

    def _delete_rows(self, ids: list[int]) -> None:
        """删除行并标记内存索引失效"""
        with self._lock:
            self._conn.executemany(
                "DELETE FROM cache_entries WHERE id = ?",
                [(i,) for i in ids],
            )
            self._conn.commit()
            invalidated = 0
            for entry_id in ids:
                pos = self._id_pos.pop(entry_id, None)
                if pos is not None and pos < self._valid.size:
                    self._valid[pos] = False
                    invalidated += 1
            # 失效占比过半 → 触发一次全量重建，避免内存索引持续膨胀
            if (
                invalidated > 0
                and self._valid.size > 0
                and self._valid.sum() * 2 < self._valid.size
            ):
                self._load_index()

    def _touch(self, entry_id: int) -> None:
        """更新访问时间与命中次数"""
        with self._lock:
            self._conn.execute(
                "UPDATE cache_entries SET last_access_at = ?, "
                "hit_count = hit_count + 1 WHERE id = ?",
                (time.time(), entry_id),
            )
            self._conn.commit()

    def _is_expired(self, created_at: float) -> bool:
        return self._ttl_seconds > 0 and time.time() - created_at > self._ttl_seconds

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "query": row["query"],
            "query_norm": row["query_norm"],
            "config_signature": row["config_signature"],
            "answer": row["answer"],
            "sources": json.loads(row["sources"]),
            "agent_path": json.loads(row["agent_path"]),
            "citations": json.loads(row["citations"]),
            "hallucination": (
                json.loads(row["hallucination"])
                if row["hallucination"] is not None else None
            ),
            "created_at": row["created_at"],
            "last_access_at": row["last_access_at"],
            "hit_count": row["hit_count"],
        }

    # ==================== 管理接口 ====================

    def stats(self) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS total, COALESCE(SUM(hit_count), 0) AS total_hits "
                "FROM cache_entries"
            ).fetchone()
            return {
                "total": row["total"],
                "total_hits": row["total_hits"],
                "max_entries": self._max_entries,
                "ttl_seconds": self._ttl_seconds,
            }

    def clear(self) -> None:
        """清空全部缓存（测试/调试用）"""
        with self._lock:
            self._conn.execute("DELETE FROM cache_entries")
            self._conn.commit()
            self._load_index()
            logger.info("CacheStorage: 已清空全部缓存")
