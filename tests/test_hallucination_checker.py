"""幻觉检测：引用文档筛选 + fail-closed 解析"""

import asyncio

from langchain_core.documents import Document

from src.services import hallucination_checker as hc


def _docs(n=3):
    return [Document(page_content=f"文档{i}内容") for i in range(1, n + 1)]


class _FakeLLM:
    def __init__(self, content):
        self.content = content
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append([m.content for m in messages])
        return type("R", (), {"content": self.content})()


def test_select_referenced_documents_filters_by_citation():
    docs = _docs(3)
    citation = {"2": {"doc_index": 2}, "5": {"doc_index": 1}}
    selected = hc._select_referenced_documents(
        docs, "答案内容 [2]。更多 [2]。", citation, 8,
    )
    assert selected == [docs[1]]  # 只保留被引用的 doc_index=2，去重


def test_select_referenced_documents_fallback_all():
    docs = _docs(3)
    # 无引用标注 → 全部文档
    assert hc._select_referenced_documents(docs, "没有编号的答案", {"1": {"doc_index": 1}}, 8) == docs
    # 无 citation_metadata → 全部文档
    assert hc._select_referenced_documents(docs, "答案 [2]。", None, 8) == docs
    # 引用映射全部失效 → 全部文档
    assert hc._select_referenced_documents(docs, "答案 [9]。", {"9": {"doc_index": 99}}, 8) == docs


def test_prompt_only_contains_cited_docs(monkeypatch):
    fake = _FakeLLM('{"passed": true, "faithfulness": 100.0}')
    monkeypatch.setattr(hc, "create_fast_llm", lambda: fake)
    docs = _docs(3)
    asyncio.run(hc.check_hallucination_async(
        docs,
        "答案引用第二个文档 [2]。",
        citation_metadata={"2": {"doc_index": 2}},
    ))
    prompt = "\n".join(fake.calls[0])
    assert "文档2内容" in prompt
    assert "文档1内容" not in prompt
    assert "文档3内容" not in prompt


def test_fail_closed_on_non_json(monkeypatch):
    fake = _FakeLLM("FAILED")  # 模型没按指令输出 JSON
    monkeypatch.setattr(hc, "create_fast_llm", lambda: fake)
    passed, faithfulness = asyncio.run(hc.check_hallucination_async(
        _docs(2), "答案 [1]。", citation_metadata={"1": {"doc_index": 1}},
    ))
    assert passed is False
    assert faithfulness == 0.0


def test_fail_closed_on_bad_json(monkeypatch):
    fake = _FakeLLM('{"passed": "maybe", "faithfulness": "很高"}')  # 字段类型非法
    monkeypatch.setattr(hc, "create_fast_llm", lambda: fake)
    passed, faithfulness = asyncio.run(hc.check_hallucination_async(
        _docs(2), "答案 [1]。", citation_metadata={"1": {"doc_index": 1}},
    ))
    assert passed is False
    assert faithfulness == 0.0
