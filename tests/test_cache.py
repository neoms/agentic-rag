"""多级缓存：精准/语义命中、向量复用、引文截断、文档失效、回放"""

import json

from src.cache.service import CacheService, build_config_signature
from src.models.chat import AgenticChatRequest


def _req(query):
    return AgenticChatRequest(query=query, session_id="t")


def test_exact_hit(temp_cache_db):
    req = _req("RAG是什么")
    sig = build_config_signature(req)
    temp_cache_db.store(
        query="RAG是什么", signature=sig, vector=[0.1, 0.2], answer="答案A",
        sources=[], agent_path=["x"], citations={}, hallucination=None,
    )
    entry, vector, info = temp_cache_db.lookup("RAG是什么", sig)
    assert entry is not None
    assert info["cache_type"] == "exact" and info["exact_hit"] is True
    assert vector is None  # 精准命中不计算向量


def test_semantic_hit_and_vector_reuse(temp_cache_db, stub_embedding):
    req = _req("RAG是什么")
    sig = build_config_signature(req)
    # stub 向量 [0.1, 0.2] 归一化后一致 → 语义命中
    temp_cache_db.store(
        query="RAG是什么", signature=sig, vector=[0.1, 0.2], answer="答案A",
        sources=[], agent_path=["x"], citations={}, hallucination=None,
    )
    entry, vector, info = temp_cache_db.lookup("什么是RAG", sig)
    assert entry is not None
    assert info["cache_type"] == "semantic" and info["semantic_hit"] is True
    assert vector == [0.1, 0.2]  # 问题向量返回供检索复用


def test_miss_returns_embedding(temp_cache_db, stub_embedding):
    req = _req("完全不相关的问题xyz")
    sig = build_config_signature(req)
    temp_cache_db.store(
        query="另一个问题", signature=sig, vector=[0.9, 0.1], answer="B",
        sources=[], agent_path=["x"], citations={}, hallucination=None,
    )
    entry, vector, info = temp_cache_db.lookup("完全不相关的问题xyz", sig)
    assert entry is None
    assert info["cache_type"] == "none"
    assert vector is not None


def test_citation_truncation(temp_cache_db):
    req = _req("长引文")
    sig = build_config_signature(req)
    long_para = "段" * 1000
    temp_cache_db.store(
        query="长引文", signature=sig, vector=[0.1, 0.2], answer="A",
        sources=[], agent_path=["x"],
        citations={"1": {"filename": "a.txt", "paragraph_text": long_para}},
        hallucination=None,
    )
    entry = temp_cache_db._storage.get_exact("长引文", sig)
    para = entry["citations"]["1"]["paragraph_text"]
    assert para.endswith("…") and len(para) <= 501


def test_invalidate_by_doc_ids(temp_cache_db):
    req = _req("文档失效")
    sig = build_config_signature(req)
    temp_cache_db.store(
        query="文档失效", signature=sig, vector=[0.1, 0.2], answer="A",
        sources=[{"content": "x", "metadata": {"doc_id": "doc1"}}],
        agent_path=["x"], citations={}, hallucination=None,
    )
    assert temp_cache_db.lookup("文档失效", sig)[0] is not None
    assert temp_cache_db.invalidate_documents(["doc1"]) == 1
    assert temp_cache_db.lookup("文档失效", sig)[0] is None


def test_replay_events(temp_cache_db):
    req = _req("回放")
    sig = build_config_signature(req)
    temp_cache_db.store(
        query="回放", signature=sig, vector=[0.1, 0.2], answer="你好，世界",
        sources=[{"content": "s"}], agent_path=["generate_simple"],
        citations={"1": {"filename": "a.txt", "paragraph_text": "p"}},
        hallucination={"passed": True, "faithfulness": 95.0},
    )
    entry, _, _ = temp_cache_db.lookup("回放", sig)
    events = temp_cache_db.replay(entry, ["cache_lookup", "cache_replay"])
    kinds = [e.event for e in events]
    assert "token" in kinds and "source" in kinds and "citations" in kinds
    assert "hallucination" in kinds and "path" in kinds
    assert json.loads(events[-1].data) == ["cache_lookup", "cache_replay"]


def test_semantic_focus_mismatch_miss(temp_cache_db, stub_embedding):
    """同主题、不同信息需求：新问法引入缓存问法未覆盖的焦点 → 判未命中"""
    req = _req("小象科技成立于哪一年,总部位于哪里?")
    sig = build_config_signature(req)
    temp_cache_db.store(
        query="小象科技成立于哪一年,总部位于哪里?", signature=sig,
        vector=[0.1, 0.2], answer="成立于2019年,总部在海淀",
        sources=[], agent_path=["x"], citations={}, hallucination=None,
    )
    # stub 向量相同 → 余弦相似度 1.0 会越过阈值，焦点校验应将其拦截
    entry, vector, info = temp_cache_db.lookup("小象科技ceo是谁", sig)
    assert entry is None
    assert info["cache_type"] == "none"
    assert info["semantic_hit"] is False
    assert vector is not None  # 向量仍返回供检索复用


def test_semantic_focus_covered_hit(temp_cache_db, stub_embedding):
    """新问法的焦点被缓存问法覆盖（答案应包含该信息）→ 正常命中"""
    req = _req("小象科技的ceo是谁,其他高管还有谁?")
    sig = build_config_signature(req)
    temp_cache_db.store(
        query="小象科技的ceo是谁,其他高管还有谁?", signature=sig,
        vector=[0.1, 0.2], answer="CEO是李明",
        sources=[], agent_path=["x"], citations={}, hallucination=None,
    )
    entry, vector, info = temp_cache_db.lookup("小象科技的ceo是谁", sig)
    assert entry is not None
    assert info["cache_type"] == "semantic" and info["semantic_hit"] is True


def test_should_promote_to_exact(temp_cache_db):
    """语义命中写回精准缓存：仅限与缓存问法高度一致的问法"""
    assert temp_cache_db.should_promote_to_exact(
        "小象科技的ceo是谁?", "小象科技的ceo是谁?"
    ) is True
    assert temp_cache_db.should_promote_to_exact(
        "小象科技的ceo是谁?", "小象科技ceo是谁"
    ) is True
    # 焦点差异较大的问法不写回，防止错误答案固化扩散
    assert temp_cache_db.should_promote_to_exact(
        "小象科技ceo是谁,首席执行官是谁?",
        "小象科技成立于哪一年,总部位于哪里?",
    ) is False
