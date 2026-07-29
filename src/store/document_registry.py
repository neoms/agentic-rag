"""文档元数据注册表 - JSON 持久化的独立索引

维护一份文档级元数据的 JSON 文件，支持快速查询文档列表、
按哈希查重、按 doc_id 查询等操作，避免频繁访问 ChromaDB。
"""

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config.settings import settings

logger = logging.getLogger(__name__)

REGISTRY_FILENAME = "document_registry.json"


class DocumentRegistry:
    """文档元数据注册表（JSON 文件持久化）

    线程安全，写入原子化（写临时文件 → rename）。
    与 ChromaDB 向量存储平行维护，互不依赖。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}  # doc_id -> metadata
        self._loaded = False

    @property
    def _file_path(self) -> Path:
        """注册表 JSON 文件路径"""
        return settings.project_root / REGISTRY_FILENAME

    def _load(self):
        """从 JSON 文件加载注册表"""
        if self._loaded:
            return
        path = self._file_path
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    self._data = {item["doc_id"]: item for item in raw if "doc_id" in item}
                logger.info("文档注册表已加载: %s, %d 条记录", path, len(self._data))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("文档注册表加载失败: %s，使用空注册表", e)
                self._data = {}
        else:
            logger.info("文档注册表不存在 %s，使用空注册表", path)
            self._data = {}
        self._loaded = True

    def _save(self):
        """原子写入注册表 JSON 文件"""
        path = self._file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        data_list = list(self._data.values())
        fd, tmp_path = tempfile.mkstemp(
            suffix=".json",
            prefix=REGISTRY_FILENAME + ".",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data_list, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, str(path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ---- 公共 API ----

    def register(
        self,
        doc_id: str,
        filename: str,
        file_type: str,
        size_bytes: int,
        content_hash: str,
        chunk_count: int,
    ) -> dict:
        """注册一条文档元数据

        Returns:
            注册的元数据字典
        """
        record = {
            "doc_id": doc_id,
            "filename": filename,
            "file_type": file_type,
            "size_bytes": size_bytes,
            "content_hash": content_hash,
            "chunk_count": chunk_count,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._load()
            self._data[doc_id] = record
            self._save()
        logger.debug("文档注册: doc_id=%s, filename=%s", doc_id, filename)
        return record

    def remove(self, doc_id: str) -> bool:
        """删除文档注册记录

        Returns:
            是否存在并删除
        """
        with self._lock:
            self._load()
            existed = doc_id in self._data
            if existed:
                del self._data[doc_id]
                self._save()
                logger.debug("文档注销: doc_id=%s", doc_id)
            return existed

    def get(self, doc_id: str) -> Optional[dict]:
        """按 doc_id 查询文档元数据"""
        with self._lock:
            self._load()
            return self._data.get(doc_id)

    def list_all(self) -> list[dict]:
        """列出所有文档元数据（无需访问 ChromaDB）"""
        with self._lock:
            self._load()
            return list(self._data.values())

    def find_by_hash(self, content_hash: str) -> Optional[str]:
        """通过 content_hash 查找 doc_id"""
        with self._lock:
            self._load()
            for record in self._data.values():
                if record.get("content_hash") == content_hash:
                    return record["doc_id"]
            return None

    def count(self) -> int:
        """文档总数"""
        with self._lock:
            self._load()
            return len(self._data)

    def update_chunk_count(self, doc_id: str, chunk_count: int):
        """更新文档的块数"""
        with self._lock:
            self._load()
            if doc_id in self._data:
                self._data[doc_id]["chunk_count"] = chunk_count
                self._save()


# 全局单例
document_registry = DocumentRegistry()
