"""安全相关回归：计算器移除、异常不泄露内部细节"""

import asyncio
import json
from pathlib import Path

import pytest
from starlette.requests import Request

from src.agent.tools import ALL_TOOLS


def test_calculator_tool_removed():
    """P0-1：计算器 eval 工具已移除，代码中不再存在 eval"""
    assert [t.name for t in ALL_TOOLS] == ["web_search_tool"]
    source = Path("src/agent/tools.py").read_text(encoding="utf-8")
    assert "eval(" not in source


def test_global_exception_handler_hides_details():
    """P0-4：全局异常处理不向客户端暴露异常详情"""
    from src.main import global_exception_handler

    fake_req = Request({
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
        "query_string": b"",
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    })
    resp = asyncio.run(
        global_exception_handler(fake_req, RuntimeError("secret /etc/passwd"))
    )
    body = json.loads(resp.body)
    assert resp.status_code == 500
    assert body == {"detail": "服务器内部错误", "error_code": "INTERNAL_ERROR"}
    assert "secret" not in json.dumps(body)


def test_sse_error_event_generic(app_client):
    """P0-4：SSE error 事件只下发通用文案"""
    from src.api.dependencies import get_rag_service
    from src.main import app

    class BoomService:
        async def agentic_rag_stream(self, request):
            raise RuntimeError("secret /etc/passwd")
            yield  # pragma: no cover

    app.dependency_overrides[get_rag_service] = lambda: BoomService()
    try:
        with app_client.stream(
            "POST", "/api/v1/chat/stream", json={"query": "hi"}
        ) as r:
            body = r.read().decode()
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 200
    assert "secret" not in body and "/etc/passwd" not in body
    assert "内部错误" in body
