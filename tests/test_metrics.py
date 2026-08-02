"""指标：/metrics 暴露、LLM/Embedding 计数、缓存命中计数"""

import asyncio

from src.metrics import llm_calls_total, embedding_calls_total


def test_metrics_endpoint_exposes_counters(app_client):
    text = app_client.get("/metrics").text
    for name in [
        "chat_requests_total",
        "chat_cache_hit_total",
        "chat_stream_duration_seconds",
        "llm_calls_total",
        "embedding_calls_total",
        "uploads_total",
        "uploads_failed_total",
    ]:
        assert name in text, name


def test_llm_client_counting_wrapper():
    import src.backend.llm as llm_mod

    class FakeLLM:
        model = "fake-model"

        def invoke(self, *a, **k):
            return "ok"

        async def ainvoke(self, *a, **k):
            return "ok"

        def stream(self, *a, **k):
            return iter(["a", "b"])

        async def astream(self, *a, **k):
            yield "a"

    fake = FakeLLM()
    llm_mod._install_retry_on_llm(fake)
    base = llm_calls_total.labels(model="fake-model")._value.get()
    fake.invoke("x")
    list(fake.stream("x"))
    assert llm_calls_total.labels(model="fake-model")._value.get() == base + 2


def test_embedding_counting(stub_embedding):
    import src.backend.embedding as emb_mod

    cli = emb_mod.create_embedding_client()
    base = embedding_calls_total._value.get()
    cli.embed_query("x")
    cli.embed_documents(["a", "b"])
    assert embedding_calls_total._value.get() == base + 2


def test_chat_cache_hit_counters(temp_cache_db, temp_state_db, monkeypatch):
    from src.cache.service import build_config_signature
    from src.memory.manager import memory_manager
    from src.metrics import (
        chat_requests_total,
        chat_cache_hit_total,
        chat_stream_duration_seconds,
    )
    from src.models.chat import AgenticChatRequest
    from src.services.rag_service import rag_service

    monkeypatch.setattr(memory_manager, "_store", temp_state_db)
    req = AgenticChatRequest(query="metrics-probe", session_id="m-sess")
    temp_cache_db.store(
        query="metrics-probe", signature=build_config_signature(req),
        vector=[0.1, 0.2], answer="缓存答案", sources=[], agent_path=["x"],
        citations={}, hallucination=None,
    )
    base_req = chat_requests_total._value.get()
    base_hit = chat_cache_hit_total.labels("exact")._value.get()
    base_sum = chat_stream_duration_seconds._sum.get()

    async def run():
        events = []
        async for ev in rag_service.agentic_rag_stream(req):
            events.append(ev)
        return events

    events = asyncio.run(run())
    assert any(e.event == "done" for e in events)
    assert chat_requests_total._value.get() == base_req + 1
    assert chat_cache_hit_total.labels("exact")._value.get() == base_hit + 1
    assert chat_stream_duration_seconds._sum.get() > base_sum
