"""重排序模块：显式超时传递与超时降级"""

from langchain_core.documents import Document

from src.backend.reranker import rerank_documents
from src.config.settings import settings


def _docs(n: int = 6) -> list[Document]:
    return [Document(page_content=f"文档{i}", metadata={"i": i}) for i in range(n)]


def test_rerank_passes_explicit_timeout(monkeypatch):
    """TextReRank.call 必须携带配置的超时，避免使用 SDK 默认 300s"""
    captured: dict = {}

    class FakeResponse:
        status_code = 200
        output = None

    def fake_call(**kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("dashscope.TextReRank.call", fake_call)
    rerank_documents("查询", _docs(6), top_k=5)
    # SDK 用 request_timeout 解析 HTTP 超时；传 timeout 不生效
    assert captured.get("request_timeout") == settings.rerank_request_timeout
    assert captured.get("request_timeout") == 10
    assert "timeout" not in captured


def test_rerank_timeout_degrades_to_original_order(monkeypatch):
    """超时/异常时降级为原始排序取前 top_k，并返回降级原因"""

    def fake_call(**kwargs):
        raise TimeoutError("TextReRank 请求超时")

    monkeypatch.setattr("dashscope.TextReRank.call", fake_call)
    docs = _docs(8)
    result, degraded = rerank_documents("查询", docs, top_k=5)
    assert degraded is not None
    assert "超时" in degraded or "失败" in degraded
    assert len(result) == 5
    assert [d.metadata["i"] for d in result] == list(range(5))
