"""离线评估 runner：stub 流采集 + fake 分数 + 双语报告 + 门禁"""

import asyncio
import json
import logging

from src.config.settings import settings
from src.eval.runner import QUALITY_IDS, compute_scores, run_offline_eval
from src.models.chat import StreamEvent


async def _fake_stream(request):
    """测试用流式事件（不调用真实 RAG/LLM）"""
    yield StreamEvent(event="token", data="这是一个测试答案")
    yield StreamEvent(
        event="source",
        data=json.dumps(
            [{"content": "测试检索上下文", "metadata": {"filename": "t.md"}, "score": 0.9}],
            ensure_ascii=False,
        ),
    )
    yield StreamEvent(event="done", data=json.dumps({"trace_id": ""}))


def _write_dataset(tmp_path, n=2):
    p = tmp_path / "ds.jsonl"
    lines = []
    for i in range(n):
        lines.append(
            json.dumps(
                {
                    "question": f"测试问题{i}",
                    "reference": f"标准答案{i}",
                    "reference_contexts": [f"golden 上下文{i}"],
                },
                ensure_ascii=False,
            )
        )
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def test_offline_eval_fake_scores_report(tmp_path):
    ds = _write_dataset(tmp_path)
    out = tmp_path / "out"
    summary = run_offline_eval(
        ds,
        output_dir=out,
        dataset_name="smoke-test",
        run_name="smoke-test",
        flags={"kg": False},
        fake=True,
        upload=False,
        stream_fn=_fake_stream,
    )
    assert summary["samples"] == 2
    for metric_id in ("faithfulness", "answer_relevancy", "context_precision"):
        assert summary["averages"].get(metric_id) is not None
    assert out.exists()
    md_files = list(out.glob("report-*.md"))
    json_files = list(out.glob("report-*.json"))
    assert len(md_files) == 1 and len(json_files) == 1

    md = md_files[0].read_text(encoding="utf-8")
    assert "faithfulness" in md and "忠实度" in md  # 中英双语
    assert "Evaluation Report" in md

    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert payload["run"]["judge_model"]
    assert payload["averages"]["faithfulness"] > 0
    assert len(payload["samples"]) == 2


def test_offline_eval_gate(tmp_path, monkeypatch):
    ds = _write_dataset(tmp_path, n=1)
    # 阈值远高于 fake 分数 → 门禁失败
    monkeypatch.setattr(settings, "eval_gate_thresholds", '{"faithfulness": 0.99}')
    summary = run_offline_eval(
        ds,
        output_dir=tmp_path / "out1",
        dataset_name="g",
        run_name="g",
        flags={},
        fake=True,
        upload=False,
        stream_fn=_fake_stream,
    )
    assert not summary["gate_passed"]
    assert summary["gate_failures"]

    # 阈值低于 fake 分数 → 门禁通过
    monkeypatch.setattr(settings, "eval_gate_thresholds", '{"faithfulness": 0.1}')
    summary = run_offline_eval(
        ds,
        output_dir=tmp_path / "out2",
        dataset_name="g2",
        run_name="g2",
        flags={},
        fake=True,
        upload=False,
        stream_fn=_fake_stream,
    )
    assert summary["gate_passed"]


def test_compute_scores_real_path_merge(monkeypatch):
    """真实路径（非 fake）的分数合并：按 reference_contexts 分组后回填原顺序"""
    collected = [
        {
            "question": f"q{i}",
            "reference": f"r{i}",
            "reference_contexts": [f"ctx{i}"] if i % 2 == 0 else [],
            "answer": f"a{i}",
            "sources": [],
            "latency_seconds": 1.0,
            "ttft_seconds": 0.5,
            "trace_id": None,
            "error": None,
        }
        for i in range(4)
    ]

    async def fake_run_ragas(rows, with_ref, *, batch_size=4):
        n = len(rows)
        scores = {
            m: [round(0.7 + idx / 100, 3) for idx in range(n)]
            for m in QUALITY_IDS
        }
        return scores, []

    monkeypatch.setattr("src.eval.runner._run_ragas", fake_run_ragas)
    per_sample, averages, skipped, notes = asyncio.run(
        compute_scores(collected, fake=False)
    )

    assert len(per_sample) == 4
    assert "faithfulness" in averages
    assert "context_recall" in averages  # 有带 reference_contexts 的样本
    # 每个样本 5 个指标都有值（fake 全量返回），顺序与 collected 一致
    assert per_sample[0]["scores"]["faithfulness"] is not None
    assert per_sample[0]["scores"]["context_precision"] is not None
    # skipped 中不应包含 context 指标（存在 golden 上下文）
    assert not skipped
    assert notes == []


def test_compute_scores_marks_empty_answer_and_failures(monkeypatch):
    """空答案样本标注 + 全指标计算失败上报（不再静默）"""
    from src.eval.runner import compute_scores

    collected = [
        {
            "question": "q",
            "reference": "r",
            "reference_contexts": ["ctx"],
            "answer": "",
            "sources": [{"content": "x"}],
            "latency_seconds": 1.0,
            "ttft_seconds": None,
            "trace_id": None,
            "error": None,
        }
    ]

    async def fake_run_ragas(rows, with_ref, *, batch_size=4):
        return {m: [None] for m in QUALITY_IDS}, []

    monkeypatch.setattr("src.eval.runner._run_ragas", fake_run_ragas)
    per_sample, averages, skipped, _ = asyncio.run(
        compute_scores(collected, fake=False)
    )

    assert per_sample[0]["answer_empty"] is True
    assert averages == {}
    assert any("全部样本计算失败" in s["reason"] for s in skipped)


def test_ragas_note_handler_captures():
    """ragas 内部错误日志被捕获进报告（其异常对象不外露）"""
    from src.eval.runner import _RagasNoteHandler

    notes: list[str] = []
    handler = _RagasNoteHandler(notes)
    lg = logging.getLogger("ragas.executor")
    lg.addHandler(handler)
    try:
        lg.error("Exception raised in Job[0]: BadRequestError(n must be 1)")
        lg.warning("LLM returned 1 generations instead of requested 3")
    finally:
        lg.removeHandler(handler)
    assert len(notes) == 2
    assert "BadRequestError" in notes[0]
    assert "1 generations" in notes[1]


def test_metric_column_prefix_match():
    """ragas 结果列名兼容："factual_correctness(mode=f1)" 这类后缀命名必须能匹配到指标 id"""
    import pandas as pd

    from src.eval.runner import _metric_column

    df = pd.DataFrame(
        {
            "faithfulness": [1.0],
            "factual_correctness(mode=f1)": [0.8],
            "answer_relevancy": [0.5],
        }
    )
    assert _metric_column(df, "faithfulness") == "faithfulness"
    assert _metric_column(df, "factual_correctness") == "factual_correctness(mode=f1)"
    assert _metric_column(df, "answer_relevancy") == "answer_relevancy"
    assert _metric_column(df, "not_exist") is None
