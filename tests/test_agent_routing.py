"""查询重写循环止损 + 文档评估 score 模糊低分区兜底"""

from langchain_core.documents import Document

from src.agent.graph import should_continue_after_grade
from src.agent.nodes import grade_documents, rerank_documents_node
from src.config.settings import settings


class FakeLLM:
    """返回 RELEVANT 的假 LLM（用于文档评估 LLM 兜底）"""

    calls = 0

    def invoke(self, messages):
        type(self).calls += 1
        return type("R", (), {"content": "RELEVANT"})()


def _state(**overrides):
    base = {
        "query": "小象科技CEO是谁",
        "documents": [],
        "documents_relevant": False,
        "iteration_count": 0,
        "max_iterations": 3,
        "rerank_top_score": 0.0,
        "best_rerank_score": 0.0,
        "rerank_improved": True,
        "enable_web_search": False,
        "enable_transform_query": True,
    }
    base.update(overrides)
    return base


def _docs_with_scores(scores):
    return [
        Document(
            page_content=content,
            metadata={"rerank_score": score},
        )
        for content, score in scores
    ]


def test_rewrite_loop_stops_when_no_improvement():
    """重写后 top1 分无提升 → 停止重写，降级走生成"""
    result = should_continue_after_grade(_state(
        documents=[Document(page_content="x")],
        iteration_count=1,
        rerank_improved=False,
        rerank_top_score=0.2384,
    ))
    assert result == "judge_complexity"


def test_rewrite_loop_continues_when_improved():
    """首轮或重写后检索有提升 → 继续重写"""
    result = should_continue_after_grade(_state(
        documents=[Document(page_content="x")],
        iteration_count=0,
        rerank_improved=True,
    ))
    assert result == "transform_query"


def test_grade_ambiguous_low_score_falls_to_llm(monkeypatch):
    """score 模糊低分区（如 top1=0.23）不再直接判不相关，交由 LLM 兜底"""
    FakeLLM.calls = 0
    monkeypatch.setattr("src.agent.nodes.create_fast_llm", lambda: FakeLLM())
    docs = _docs_with_scores([
        ("### 创始团队 - 李明（CEO，联合创始人）：前阿里巴巴高级技术总监", 0.23),
        ("### 2.3 NexusML - 机器学习平台", 0.15),
    ])
    out = grade_documents(_state(documents=docs))
    assert out["documents_relevant"] is True
    assert FakeLLM.calls == 1  # 走了 LLM 兜底，而非 score 负判定


def test_grade_hard_low_score_irrelevant_without_llm(monkeypatch):
    """score 极低（≤ hard_min）仍直接负判，不调用 LLM"""
    FakeLLM.calls = 0
    monkeypatch.setattr("src.agent.nodes.create_fast_llm", lambda: FakeLLM())
    docs = _docs_with_scores([
        ("### 创始团队 - 李明（CEO，联合创始人）", 0.05),
        ("### 2.3 NexusML - 机器学习平台", 0.04),
    ])
    out = grade_documents(_state(documents=docs))
    assert out["documents_relevant"] is False
    assert FakeLLM.calls == 0


def _kg_doc(score=None):
    metadata = {"source": "knowledge_graph", "filename": "知识图谱"}
    if score is not None:
        metadata["rerank_score"] = score
    return Document(
        page_content="知识图谱: 小象科技 -[CEO]-> 李明",
        metadata=metadata,
    )


def test_rerank_node_reinserts_kg_at_bottom_with_score(monkeypatch):
    """KG 参与重排并拿到真实分数；被挤出 top_k 后保底插回列表底部"""
    local_docs = [
        Document(
            page_content=f"小象科技CEO相关介绍 {i}",
            metadata={"rerank_score": 0.9 - i * 0.05},
        )
        for i in range(10)
    ]
    kg_doc = _kg_doc(score=0.05)
    docs = local_docs + [kg_doc]

    def fake_rerank(query, documents, top_k=None):
        scored = sorted(
            documents,
            key=lambda d: d.metadata.get("rerank_score", 0.0),
            reverse=True,
        )
        return scored, None

    monkeypatch.setattr("src.agent.nodes.rerank_documents", fake_rerank)
    out = rerank_documents_node(_state(
        query="小象科技CEO是谁",
        documents=docs,
    ))
    result = out["documents"]
    top_k = settings.rerank_top_k
    assert len(result) == top_k + 1  # top_k 本地文档 + 保底 KG
    assert result[-1].metadata.get("source") == "knowledge_graph"
    assert result[-1].metadata.get("rerank_score") is not None
    assert out["rerank_top_score"] == result[0].metadata["rerank_score"]


def test_rerank_node_degraded_keeps_kg(monkeypatch):
    """重排 API 降级时，KG 仍保底保留（追加到结果尾部）"""
    local_docs = [
        Document(page_content=f"小象科技CEO相关介绍 {i}")
        for i in range(10)
    ]
    kg_doc = _kg_doc()
    docs = local_docs + [kg_doc]

    def fake_rerank(query, documents, top_k=None):
        return documents, "API 异常: xxx"

    monkeypatch.setattr("src.agent.nodes.rerank_documents", fake_rerank)
    out = rerank_documents_node(_state(
        query="小象科技CEO是谁",
        documents=docs,
    ))
    result = out["documents"]
    assert result[-1].metadata.get("source") == "knowledge_graph"


def test_grade_keeps_kg_in_filtered():
    """文档评估的词法过滤不丢弃 KG 上下文（结构化图谱文本 overlap 低）"""
    docs = [
        Document(
            page_content=f"小象科技CEO相关介绍 {i}",
            metadata={"rerank_score": score},
        )
        for i, score in enumerate([0.5, 0.4, 0.3, 0.2, 0.1])
    ]
    kg_doc = _kg_doc(score=0.05)
    docs.append(kg_doc)
    out = grade_documents(_state(documents=docs))
    assert out["documents_relevant"] is True
    assert any(
        d.metadata.get("source") == "knowledge_graph"
        for d in out["documents"]
    )


def test_grade_score_relevant_keeps_kg():
    """score 正判通道（返回 top3）时，KG 若不在 top3 也保底追加"""
    docs = [
        Document(
            page_content=f"小象科技CEO相关介绍 {i}",
            metadata={"rerank_score": score},
        )
        for i, score in enumerate([0.9, 0.5, 0.4])
    ]
    kg_doc = _kg_doc(score=0.1)
    docs.append(kg_doc)
    out = grade_documents(_state(documents=docs))
    assert out["documents_relevant"] is True
    assert out["documents"][-1].metadata.get("source") == "knowledge_graph"
