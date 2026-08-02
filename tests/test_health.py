"""健康检查：组件化明细、故障降级、deep 探针"""


def test_health_all_ok(app_client):
    r = app_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok", body
    assert {"chroma", "state_db", "cache", "kg", "config"} <= set(body["checks"])


def test_health_degraded_on_component_failure(app_client, monkeypatch):
    from src.store import vector_store as vs_mod

    def boom():
        raise RuntimeError("disk broken")

    monkeypatch.setattr(vs_mod.vector_store, "get_collection_stats", boom)
    body = app_client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["checks"]["chroma"].startswith("error:")


def test_health_deep_probe_graceful(app_client):
    """deep=true 即使 Embedding 调用失败也只降级，不 500"""
    r = app_client.get("/health?deep=true")
    assert r.status_code == 200
    assert "embedding" in r.json()["checks"]
