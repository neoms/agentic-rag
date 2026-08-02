"""文档上传：流式落盘、大小限制、队列上限、临时文件清理"""

import time
from pathlib import Path

import pytest

from src.config.settings import settings


def _temp_names():
    temp_dir = settings.project_root / "data" / "temp_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return set(p.name for p in temp_dir.iterdir())


def _wait_completed(svc, n, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        tasks = svc.list_tasks()
        if tasks and all(t.status.value == "completed" for t in tasks) and len(tasks) >= n:
            return True
        time.sleep(0.05)
    return False


def test_upload_small_file(app_client, fresh_document_service, stub_indexer):
    before = _temp_names()
    r = app_client.post(
        "/api/v1/documents/upload",
        files={"file": ("small.txt", b"hello world", "text/plain")},
    )
    assert r.status_code == 202
    assert stub_indexer[-1][0] == 11
    assert _wait_completed(fresh_document_service, 1)
    assert _temp_names() <= before


def test_upload_oversized_rejected(app_client, fresh_document_service):
    before = _temp_names()
    big = b"x" * (10 * 1024 * 1024 + 512 * 1024)
    r = app_client.post(
        "/api/v1/documents/upload",
        files={"file": ("big.txt", big, "text/plain")},
    )
    assert r.status_code == 413
    assert "超过限制" in r.json()["detail"]
    assert len(fresh_document_service.list_tasks()) == 0
    assert _temp_names() <= before


def test_upload_empty_rejected(app_client):
    r = app_client.post(
        "/api/v1/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert r.status_code == 400
    assert "为空" in r.json()["detail"]


def test_upload_large_path_handoff(app_client, fresh_document_service, stub_indexer):
    before = _temp_names()
    large = b"y" * (6 * 1024 * 1024)
    r = app_client.post(
        "/api/v1/documents/upload",
        files={"file": ("large.txt", large, "text/plain")},
    )
    assert r.status_code == 202
    # 路由流式落盘后以 Path 交接给服务，后台线程读取一次再 ingest
    assert stub_indexer[-1][0] == 6 * 1024 * 1024
    assert _wait_completed(fresh_document_service, 1)
    time.sleep(0.3)  # 等后台线程 finally 清理
    assert _temp_names() <= before


def test_queue_full_rejects_and_cleans_up(tmp_path, monkeypatch):
    """队列饱和时大文件上传返回 429，且移交的临时文件被清理"""
    import src.cache as cache_mod
    from fastapi.testclient import TestClient

    from src.api.dependencies import get_document_service
    from src.main import app
    from src.services.document_service import DocumentService
    from src.store.state_store import RuntimeStateStore

    import src.services.document_service as ds_mod
    import src.store.state_store as ss_mod

    ss_mod._store = RuntimeStateStore(tmp_path / "state.db")
    orig_cache = settings.cache_db_path
    settings.cache_db_path = str(tmp_path / "cache.db")
    cache_mod.reset_cache_service()

    class SlowIndexer:
        def ingest(self, file_bytes, filename):
            time.sleep(1.0)
            return {"doc_id": "s", "filename": filename, "chunk_count": 1, "deduplicated": False}

    monkeypatch.setattr(ds_mod, "document_indexer", SlowIndexer())
    svc = DocumentService()
    try:
        # 填满队列：workers=2 + queue=20 → 第 23 个被拒
        for i in range(22):
            svc.submit_upload_task(b"x", f"f{i}.txt")

        before = _temp_names()
        app.dependency_overrides[get_document_service] = lambda: svc
        with TestClient(app) as client:
            r = client.post(
                "/api/v1/documents/upload",
                files={"file": ("qf.txt", b"y" * (6 * 1024 * 1024), "text/plain")},
            )
        app.dependency_overrides.clear()
        assert r.status_code == 429
        assert "队列已满" in r.json()["detail"]
        assert _temp_names() <= before
    finally:
        svc.shutdown(wait=True)
        cache_mod.reset_cache_service()
        settings.cache_db_path = orig_cache
