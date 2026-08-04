"""Locust 压测报告纯函数工具（不依赖 locust，便于单元测试）

包含：SSE 事件流切分、/metrics token 计数解析、百分位计算、阶段中文标签。
"""

from __future__ import annotations

import re
from typing import Iterator

# 阶段中文名（报告展示用；未收录的节点按原始 id 显示）
_STAGE_ZH = {
    "cache_lookup": "缓存查询",
    "cache_replay": "输出回放",
    "cache_store": "缓存写入",
    "analyze_kg_intent": "意图分析",
    "parallel_retrieve_merge": "并行检索合并",
    "retrieve": "检索",
    "retrieve_semantic": "语义检索",
    "retrieve_bm25": "BM25 检索",
    "retrieve_kg": "知识图谱检索",
    "transform_query": "查询重写",
    "rerank_documents": "重排序",
    "grade_documents": "文档评估",
    "web_search": "联网搜索",
    "judge_complexity": "复杂度判定",
    "generate_simple": "简单生成",
    "generate_complex": "复杂生成",
    "check_hallucination": "幻觉检测",
}
STAGE_ORDER = list(_STAGE_ZH)


def stage_label(node_id: str) -> str:
    """阶段双语标签：生成（generate_simple）"""
    zh = _STAGE_ZH.get(node_id)
    return f"{zh}（{node_id}）" if zh else node_id


def iter_sse_events(lines) -> Iterator[tuple[str, str]]:
    """把 SSE 原始行流切分为 (event, data) 块（兼容 bytes 行）"""
    event: str | None = None
    data: list[bytes] = []
    for line in lines:
        if not line:
            if event is not None:
                yield event, b"\n".join(data).decode("utf-8", "replace")
            event = None
            data = []
            continue
        line = line.rstrip(b"\r")
        if line.startswith(b"event:"):
            event = line[len(b"event:"):].strip().decode("utf-8", "replace")
        elif line.startswith(b"data:"):
            data.append(line[len(b"data:"):].strip())
    if event is not None:
        yield event, b"\n".join(data).decode("utf-8", "replace")


_TOKEN_RE = re.compile(r'^llm_tokens_total\{([^}]*)\}\s+([\d.]+)$')


def parse_token_lines(text: str) -> dict[tuple[str, str], float]:
    """从 /metrics 文本解析 llm_tokens_total{model,type} 计数（无该指标时返回空 dict）"""
    counts: dict[tuple[str, str], float] = {}
    for line in text.splitlines():
        line = line.strip()
        m = _TOKEN_RE.match(line)
        if not m:
            continue
        try:
            labels = dict(
                kv.split("=", 1) for kv in m.group(1).split(",") if "=" in kv
            )
            model = labels.get("model", "").strip('"')
            typ = labels.get("type", "").strip('"')
            counts[(model, typ)] = float(m.group(2))
        except (ValueError, AttributeError):
            continue
    return counts


def percentile(values: list[float], p: float) -> float:
    """线性插值百分位；空列表返回 0.0"""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (idx - lo)
