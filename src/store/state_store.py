"""运行时状态持久化 - 会话历史与上传任务落盘（SQLite, WAL）

解决重启丢状态问题：
- chat_messages：按 session 持久化对话历史，MemoryManager 懒加载恢复
- 幻觉检测结果随 assistant 消息落库，前端刷新后不丢失
- upload_tasks：文档后台任务状态持久化，重启后恢复（未完成任务标记为中断）
"""

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config.settings import settings

logger = logging.getLogger(__name__)


class RuntimeStateStore:
    """SQLite 运行时状态存储（单写连接 + RLock，与 CacheStorage 同模式）"""

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path or settings.state_db_path_abs
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.RLock()
        self._create_schema()
        self._migrate_schema()
        logger.info("RuntimeStateStore 初始化: db=%s", self._db_path)

    def _create_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    hallucination TEXT
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_messages_session "
                "ON chat_messages(session_id, id)"
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS upload_tasks (
                    task_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    chunk_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self._conn.commit()

    def _migrate_schema(self) -> None:
        """存量库迁移：为 chat_messages 补充 hallucination 列（幂等）"""
        with self._lock:
            cols = {
                row[1]
                for row in self._conn.execute(
                    "PRAGMA table_info(chat_messages)"
                ).fetchall()
            }
            if "hallucination" not in cols:
                self._conn.execute(
                    "ALTER TABLE chat_messages ADD COLUMN hallucination TEXT"
                )
                self._conn.commit()
                logger.info("RuntimeStateStore: 迁移 chat_messages 增加 hallucination 列")

    # ==================== 会话历史 ====================

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """追加一条会话消息"""
        with self._lock:
            self._conn.execute(
                "INSERT INTO chat_messages (session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, role, content, time.time()),
            )
            self._conn.commit()

    def get_messages(self, session_id: str, limit: int | None = None) -> list[dict]:
        """按时间升序返回会话消息；limit 表示只取最近 N 条"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, hallucination FROM chat_messages "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit if limit is not None else -1),
            ).fetchall()
        # DESC 取最近 N 条后反转回时间升序
        result: list[dict] = []
        for r in reversed(rows):
            msg: dict = {"role": r["role"], "content": r["content"]}
            hallucination = self._parse_hallucination(r["hallucination"])
            if hallucination is not None:
                msg["hallucination"] = hallucination
            result.append(msg)
        return result

    @staticmethod
    def _parse_hallucination(raw: str | None) -> dict | None:
        """解析幻觉结果 JSON；无效/为空返回 None"""
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None

    def set_hallucination(self, session_id: str, hallucination: dict) -> None:
        """为指定会话最近一条 assistant 消息写入幻觉检测结果"""
        with self._lock:
            self._conn.execute(
                "UPDATE chat_messages SET hallucination = ? WHERE id = ("
                "SELECT id FROM chat_messages "
                "WHERE session_id = ? AND role = 'assistant' "
                "ORDER BY id DESC LIMIT 1)",
                (json.dumps(hallucination, ensure_ascii=False), session_id),
            )
            self._conn.commit()

    def list_sessions(self) -> list[dict]:
        """按最近活跃时间倒序返回全部会话摘要"""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT session_id,
                       COUNT(*) AS message_count,
                       MAX(created_at) AS updated_at,
                       (SELECT content FROM chat_messages m2
                        WHERE m2.session_id = m.session_id AND m2.role = 'user'
                        ORDER BY m2.id DESC LIMIT 1) AS preview
                FROM chat_messages m
                GROUP BY session_id
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [
            {
                "session_id": r["session_id"],
                "preview": r["preview"] or "",
                "message_count": r["message_count"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    def clear_messages(self, session_id: str) -> None:
        """删除指定会话的全部持久化消息"""
        with self._lock:
            self._conn.execute(
                "DELETE FROM chat_messages WHERE session_id = ?", (session_id,)
            )
            self._conn.commit()

    def trim_session(self, session_id: str, keep: int) -> None:
        """单会话仅保留最近 keep 条消息，删除更早的旧消息"""
        if keep <= 0:
            return
        with self._lock:
            self._conn.execute(
                "DELETE FROM chat_messages WHERE id IN ("
                "SELECT id FROM chat_messages WHERE session_id = ? "
                "ORDER BY id DESC LIMIT -1 OFFSET ?)",
                (session_id, keep),
            )
            self._conn.commit()

    # ==================== 上传任务 ====================

    def upsert_task(self, task: dict) -> None:
        """写入或更新任务状态（status 兼容 TaskStatus 枚举或字符串）"""
        status = task["status"]
        if hasattr(status, "value"):
            status = status.value
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO upload_tasks
                    (task_id, doc_id, filename, status, message, created_at, completed_at, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    status = excluded.status,
                    message = excluded.message,
                    completed_at = excluded.completed_at,
                    chunk_count = excluded.chunk_count
                """,
                (
                    task["task_id"],
                    task["doc_id"],
                    task["filename"],
                    status,
                    task["message"],
                    task["created_at"],
                    task["completed_at"],
                    task["chunk_count"],
                ),
            )
            self._conn.commit()

    def get_task(self, task_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM upload_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def list_tasks(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM upload_tasks ORDER BY created_at DESC, rowid DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_interrupted_tasks(
        self, message: str = "服务重启导致任务中断"
    ) -> int:
        """将启动前仍处于 pending/processing 的任务标记为 failed"""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE upload_tasks SET status = 'failed', message = ?, completed_at = ? "
                "WHERE status IN ('pending', 'processing')",
                (message, datetime.now(timezone.utc).isoformat()),
            )
            self._conn.commit()
            return cur.rowcount

    def prune_tasks(self, keep: int, ttl_days: int = 0) -> None:
        """清理历史任务：仅保留最新 keep 条；完成/失败超过 ttl_days 天的删除

        TTL 在 Python 侧解析 ISO 时间比较（SQLite 的 datetime 函数无法处理
        带时区偏移的 ISO 字符串）。
        """
        with self._lock:
            # 1) 条数上限：保留最新 keep 条
            self._conn.execute(
                "DELETE FROM upload_tasks WHERE task_id NOT IN ("
                "SELECT task_id FROM upload_tasks "
                "ORDER BY created_at DESC, rowid DESC LIMIT ?)",
                (keep,),
            )
            # 2) TTL：删除完成/失败超过 ttl_days 天的任务
            if ttl_days > 0:
                cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
                rows = self._conn.execute(
                    "SELECT task_id, completed_at, created_at FROM upload_tasks "
                    "WHERE status IN ('completed', 'failed')"
                ).fetchall()
                stale: list[str] = []
                for r in rows:
                    ts = r["completed_at"] or r["created_at"]
                    try:
                        dt = datetime.fromisoformat(ts)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue
                    if dt < cutoff:
                        stale.append(r["task_id"])
                if stale:
                    self._conn.executemany(
                        "DELETE FROM upload_tasks WHERE task_id = ?",
                        [(s,) for s in stale],
                    )
            self._conn.commit()

    def close(self) -> None:
        """关闭 SQLite 连接（WAL checkpoint + 释放句柄）"""
        with self._lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
            self._conn.close()
            logger.info("RuntimeStateStore: 已关闭 db=%s", self._db_path)


# 全局单例（懒加载）
_store: RuntimeStateStore | None = None


def get_runtime_state_store() -> RuntimeStateStore:
    global _store
    if _store is None:
        _store = RuntimeStateStore()
    return _store
