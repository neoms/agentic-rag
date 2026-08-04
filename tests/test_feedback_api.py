"""用户反馈 API：未配置 Langfuse 时 503，配置后写回 score"""


def test_feedback_503_without_langfuse(app_client, monkeypatch):
    monkeypatch.setattr("src.eval.langfuse.langfuse_enabled", lambda: False)
    resp = app_client.post(
        "/api/v1/chat/feedback",
        json={"trace_id": "trace-123", "rating": 5},
    )
    assert resp.status_code == 503


def test_feedback_writes_score(app_client, monkeypatch):
    calls: list[tuple] = []

    def fake_record(trace_id, rating, comment=None):
        calls.append((trace_id, rating, comment))
        return True

    monkeypatch.setattr("src.eval.langfuse.record_feedback", fake_record)
    resp = app_client.post(
        "/api/v1/chat/feedback",
        json={"trace_id": "trace-123", "rating": 5, "comment": "很好"},
    )
    assert resp.status_code == 200
    assert calls == [("trace-123", 5, "很好")]


def test_feedback_validation(app_client):
    resp = app_client.post(
        "/api/v1/chat/feedback",
        json={"trace_id": "t", "rating": 6},
    )
    assert resp.status_code == 422
