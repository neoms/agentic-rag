"""会话 ID 强化：完整 UUID、兼容存量短 id、拒绝非法/超长输入"""

import uuid


def test_full_uuid_session_accepted(app_client):
    sid = str(uuid.uuid4())
    r = app_client.get(f"/api/v1/chat/history/{sid}")
    assert r.status_code == 200 and r.json()["total"] == 0


def test_legacy_short_session_still_works(app_client):
    """存量 8 位 hex 短会话 id 保持可用"""
    r = app_client.get("/api/v1/chat/history/abcdef12")
    assert r.status_code == 200
    r2 = app_client.delete("/api/v1/chat/history/abcdef12")
    assert r2.status_code == 200


def test_invalid_chars_rejected(app_client):
    assert app_client.get("/api/v1/chat/history/bad_sid!").status_code == 422
    assert app_client.get("/api/v1/chat/history/bad%20sid").status_code == 422


def test_overlong_session_rejected(app_client):
    r = app_client.get("/api/v1/chat/history/" + "a" * 65)
    assert r.status_code == 422


def test_stream_body_session_validation(app_client):
    """请求体中的非法 session_id 由 Pydantic 校验拒绝"""
    r = app_client.post(
        "/api/v1/chat/stream",
        json={"query": "hi", "session_id": "bad id!"},
    )
    assert r.status_code == 422
