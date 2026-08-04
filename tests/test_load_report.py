"""Locust 压测报告纯函数：SSE 事件切分与 token 计数解析"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "load"))
from report_helpers import iter_sse_events, parse_token_lines, percentile  # noqa: E402


def test_iter_sse_events_parses_blocks():
    raw = [
        b"event: node_start",
        b"data: cache_lookup",
        b"",
        b"event: token",
        "data: 你好".encode("utf-8"),
        b"",
        b"event: node_data",
        b'data: {"cache_lookup": {"durationMs": 1.2, "output": {"hit": true}}}',
        b"",
        b"event: done",
        b'data: {"trace_id": "abc"}',
    ]
    events = list(iter_sse_events(raw))
    assert events[0] == ("node_start", "cache_lookup")
    assert events[1] == ("token", "你好")
    assert events[2][0] == "node_data"
    assert "cache_lookup" in events[2][1]
    assert events[3] == ("done", '{"trace_id": "abc"}')


def test_parse_token_lines():
    text = (
        "# HELP llm_tokens_total LLM token usage\n"
        "# TYPE llm_tokens_total counter\n"
        'llm_tokens_total{model="m1",type="input"} 1234.0\n'
        'llm_tokens_total{model="m1",type="output"} 56.0\n'
        "other_metric 1.0\n"
    )
    counts = parse_token_lines(text)
    assert counts == {("m1", "input"): 1234.0, ("m1", "output"): 56.0}
    assert parse_token_lines("no metrics here\n") == {}


def test_percentile_empty_and_interpolation():
    assert percentile([], 0.95) == 0.0
    assert percentile([5.0], 0.50) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.50) == 2.5
