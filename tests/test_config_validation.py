"""启动配置校验：问题收集、可行动报错、fail fast"""

import pytest

from src.config.settings import settings
from src.config.validation import validate_settings, format_issues


def test_default_settings_pass():
    """当前 .env 配置应全部通过校验"""
    assert validate_settings(settings) == []


def test_multiple_violations_collected(monkeypatch):
    monkeypatch.setattr(settings, "dashscope_api_key", "")
    monkeypatch.setattr(settings, "index_workers", 0)
    monkeypatch.setattr(settings, "llm_temperature", 3.0)
    monkeypatch.setattr(settings, "allowed_extensions", "pdf,,bad ext!")
    issues = validate_settings(settings)
    keys = {i.key for i in issues}
    assert {"DASHSCOPE_API_KEY", "INDEX_WORKERS", "LLM_TEMPERATURE"} <= keys
    for issue in issues:
        assert issue.reason and issue.location and issue.fix
    msg = format_issues(issues)
    assert "启动配置校验失败" in msg and "修复" in msg and "位置" in msg


def test_cross_field_consistency(monkeypatch):
    monkeypatch.setattr(settings, "chunk_size", 500)
    monkeypatch.setattr(settings, "chunk_overlap", 600)      # ≥ chunk_size
    monkeypatch.setattr(settings, "max_upload_size_mb", 10)
    monkeypatch.setattr(settings, "large_file_threshold_mb", 20)  # > 上限
    monkeypatch.setattr(settings, "memory_window_size", 5)
    monkeypatch.setattr(settings, "session_message_cap", 4)  # < 窗口两倍
    issues = validate_settings(settings)
    keys = {i.key for i in issues}
    assert "CHUNK_OVERLAP" in keys
    assert "LARGE_FILE_THRESHOLD_MB" in keys
    assert "SESSION_MESSAGE_CAP" in keys


def test_invalid_chunk_config_json(monkeypatch):
    monkeypatch.setattr(settings, "chunk_config_by_type", "{bad json")
    issues = validate_settings(settings)
    assert any(i.key == "CHUNK_CONFIG_BY_TYPE" for i in issues)


def test_invalid_cors_origin(monkeypatch):
    monkeypatch.setattr(settings, "cors_allowed_origins", "localhost:3000")
    issues = validate_settings(settings)
    assert any(i.key == "CORS_ALLOWED_ORIGINS" for i in issues)


def test_startup_fails_fast_without_api_key(monkeypatch):
    from fastapi.testclient import TestClient
    from src.main import app

    monkeypatch.setattr(settings, "dashscope_api_key", "")
    with pytest.raises(RuntimeError, match="启动配置校验失败"):
        with TestClient(app):
            pass
