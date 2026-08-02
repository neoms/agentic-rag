"""pytest 共享 Fixtures

复用开发验证阶段的 stub 技巧：
- temp_state_db / temp_cache_db：数据隔离到临时目录，不碰真实 data/
- stub_embedding：固定向量，避免 Embedding API 外网调用
- stub_indexer：替换文档索引，避免 embedding/KG LLM 调用
- fresh_document_service：基于临时状态库的全新服务实例
- app_client：TestClient + 依赖注入临时服务，全量隔离
"""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.config.settings import settings
from src.store.state_store import RuntimeStateStore


@pytest.fixture
def temp_state_db(tmp_path, monkeypatch):
    """将运行时状态库（模块单例）替换为临时库"""
    import src.store.state_store as ss_mod
    store = RuntimeStateStore(tmp_path / "state.db")
    monkeypatch.setattr(ss_mod, "_store", store)
    return store


@pytest.fixture
def temp_cache_db(tmp_path, monkeypatch):
    """将多级缓存库替换为临时库"""
    import src.cache as cache_mod
    from src.cache import get_cache_service
    orig_path = settings.cache_db_path
    settings.cache_db_path = str(tmp_path / "cache.db")
    cache_mod.reset_cache_service()
    try:
        yield get_cache_service()
    finally:
        cache_mod.reset_cache_service()
        settings.cache_db_path = orig_path


@pytest.fixture
def stub_embedding(monkeypatch):
    """Stub Embedding API：固定向量，避免外网调用"""
    import src.backend.embedding as emb_mod

    def fake_embed_with_retry(*args, **kwargs):
        return [{"embedding": [0.1, 0.2]}]

    monkeypatch.setattr(emb_mod, "embed_with_retry", fake_embed_with_retry)
    return emb_mod


@pytest.fixture
def stub_indexer(monkeypatch):
    """Stub 文档索引：记录输入并快速返回，避免 embedding/KG LLM 调用"""
    import src.services.document_service as ds_mod
    calls: list = []

    class StubIndexer:
        def ingest(self, file_bytes, filename):
            calls.append((
                "path" if isinstance(file_bytes, Path) else len(file_bytes),
                filename,
            ))
            time.sleep(0.05)
            return {
                "doc_id": "stub",
                "filename": filename,
                "chunk_count": 1,
                "deduplicated": False,
            }

    monkeypatch.setattr(ds_mod, "document_indexer", StubIndexer())
    return calls


@pytest.fixture
def fresh_document_service(temp_state_db, stub_indexer):
    """基于临时状态库的全新 DocumentService（不碰真实单例）"""
    from src.services.document_service import DocumentService
    svc = DocumentService()
    try:
        yield svc
    finally:
        svc.shutdown(wait=True)


@pytest.fixture
def app_client(fresh_document_service, temp_cache_db):
    """TestClient：依赖注入临时 DocumentService，state/cache 全隔离"""
    from src.main import app
    from src.api.dependencies import get_document_service
    app.dependency_overrides[get_document_service] = lambda: fresh_document_service
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
