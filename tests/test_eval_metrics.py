"""评估体系：双语指标注册表、门禁判定、数据集 schema 校验"""

import json

from src.eval.dataset import load_dataset, validate_dataset_file
from src.eval.metrics import ALL_METRICS, QUALITY_METRICS, METRIC_BY_ID, display_name
from src.eval.report import check_gate
from src.eval.runner import gate_requires_thresholds


def test_metric_registry_bilingual_and_unique():
    """每个指标都有完整双语字段且 id 唯一"""
    ids = set()
    for m in ALL_METRICS:
        assert m.id and m.name_zh and m.name_en
        assert m.description_zh and m.description_en
        assert m.kind in ("quality", "performance", "experience")
        assert m.id not in ids, f"重复指标 id: {m.id}"
        ids.add(m.id)
        # 双语显示名应同时包含中英文
        assert m.name_en in m.display and m.name_zh in m.display
    assert METRIC_BY_ID["faithfulness"].name_zh == "忠实度"
    assert len(QUALITY_METRICS) == 5


def test_display_name_unknown_id():
    assert display_name("unknown_metric") == "unknown_metric"


def test_gate_check():
    averages = {"faithfulness": 0.9, "context_recall": 0.5}
    passed, failures = check_gate(averages, {"faithfulness": 0.85})
    assert passed and not failures
    passed, failures = check_gate(averages, {"faithfulness": 0.95, "context_recall": 0.6})
    assert not passed
    assert len(failures) == 2
    # 空阈值恒通过
    passed, _ = check_gate(averages, {})
    assert passed


def test_gate_requires_thresholds():
    """--gate 开启但未配置阈值时必须拒绝执行，避免门禁形同虚设"""
    assert gate_requires_thresholds(gate=True, thresholds={}) is True
    assert gate_requires_thresholds(gate=True, thresholds={"faithfulness": 0.85}) is False
    assert gate_requires_thresholds(gate=False, thresholds={}) is False


def test_dataset_validation(tmp_path):
    good = tmp_path / "good.jsonl"
    good.write_text(
        json.dumps(
            {"question": "q1", "reference": "a1", "reference_contexts": ["ctx1"]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert validate_dataset_file(good) == []
    samples = load_dataset(good)
    assert len(samples) == 1 and samples[0].reference_contexts == ["ctx1"]

    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        "\n".join(
            [
                json.dumps({"question": "q1"}, ensure_ascii=False),  # 缺 reference
                json.dumps(
                    {"question": "q2", "reference": "a2", "reference_contexts": "not-list"},
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )
    errors = validate_dataset_file(bad)
    assert any("reference" in e and "1" in e for e in errors)
    assert any("reference_contexts" in e and "2" in e for e in errors)
