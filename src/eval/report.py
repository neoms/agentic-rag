"""双语评估报告 - 控制台表格 / JSON / Markdown 生成与门禁判定"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from src.eval.metrics import METRIC_BY_ID, display_name


def _tz() -> timezone:
    return timezone(timedelta(hours=8), name="CST")


def build_report_payload(
    *,
    dataset_name: str,
    dataset_path: str,
    run_name: str,
    judge_model: str,
    judge_isolated: bool,
    samples_count: int,
    strategy: dict[str, Any],
    paced: bool = False,
    per_sample: list[dict[str, Any]],
    averages: dict[str, float],
    skipped: list[dict[str, Any]],
    notes: list[str],
    langfuse_url: str | None,
) -> dict[str, Any]:
    """构造统一报告结构（双语描述内嵌）"""
    return {
        "dataset": {
            "name": dataset_name,
            "path": dataset_path,
            "samples": samples_count,
        },
        "run": {
            "name": run_name,
            "timestamp": datetime.now(_tz()).isoformat(timespec="seconds"),
            "strategy": strategy,
            "judge_model": judge_model,
            "judge_isolated": judge_isolated,
            "paced": paced,
            "langfuse_url": langfuse_url,
        },
        "averages": averages,
        "skipped_metrics": skipped,
        "eval_notes": notes,
        "samples": per_sample,
    }


def _metric_row(metric_id: str, score: float | None) -> str:
    return f"{display_name(metric_id):<40} {score if score is None else f'{score:.4f}'}"


def format_console_table(
    averages: dict[str, float],
    skipped: list[dict[str, Any]],
    per_sample: list[dict[str, Any]],
) -> str:
    """控制台双语结果表"""
    lines: list[str] = []
    header = f"{'metric（指标）':<40} {'avg（平均）':<12}"
    lines.append(header)
    lines.append("-" * len(header))
    for metric_id, score in averages.items():
        lines.append(_metric_row(metric_id, score))
    if skipped:
        lines.append("")
        lines.append("skipped（跳过）:")
        for item in skipped:
            lines.append(
                f"  - {display_name(item['metric'])}: {item.get('reason', '')}"
            )
    empty_count = sum(1 for r in per_sample if r.get("answer_empty"))
    if empty_count:
        lines.append("")
        lines.append(f"⚠ {empty_count} 个样本答案为空（answer empty），对应指标记为 0/None")
    lines.append("")
    lines.append(f"{'question（问题）':<36} {'avg（平均）'}")
    lines.append("-" * 52)
    for row in per_sample:
        q = row["question"][:34]
        avg = row.get("avg", float("nan"))
        lines.append(f"{q:<36} {'N/A' if avg is None else f'{avg:.4f}'}")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    payload: dict[str, Any],
) -> tuple[Path, Path]:
    """写 JSON + Markdown 双语报告，返回 (json_path, md_path)"""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(_tz()).strftime("%Y%m%d-%H%M%S")
    run_name = payload["run"]["name"]

    json_path = output_dir / f"report-{run_name}-{ts}.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_path = output_dir / f"report-{run_name}-{ts}.md"
    md_lines = [
        "# Agentic RAG 评估报告 / Evaluation Report",
        "",
        f"- 数据集 / Dataset: `{payload['dataset']['name']}`"
        f"（{payload['dataset']['samples']} 条）",
        f"- 运行 / Run: `{payload['run']['name']}`",
        f"- 时间 / Time: {payload['run']['timestamp']}",
        f"- judge 模型 / Judge model: `{payload['run']['judge_model']}`"
        + ("（独立评判模型 / isolated）" if payload["run"]["judge_isolated"] else "（与被测同源 / same as generator）"),
        f"- 策略 / Strategy: `{json.dumps(payload['run']['strategy'], ensure_ascii=False)}`",
        f"- 低频不并发模式 / Paced mode: "
        + ("开启（串行 + 样本间隔） / ON (serial + sample interval)" if payload["run"].get("paced") else "关闭 / OFF"),
        f"- Langfuse: {payload['run'].get('langfuse_url') or '（未配置 / not configured）'}",
        "",
        "## 指标结果 / Metric Averages",
        "",
        "| 指标 / Metric | 平均分 / Average | 说明 / Description |",
        "|------|:----:|------|",
    ]
    for metric_id, score in payload["averages"].items():
        spec = METRIC_BY_ID.get(metric_id)
        desc = f"{spec.description_zh} / {spec.description_en}" if spec else ""
        md_lines.append(f"| {display_name(metric_id)} | {score:.4f} | {desc} |")
    if payload["skipped_metrics"]:
        md_lines.extend(
            [
                "",
                "### 跳过的指标 / Skipped Metrics",
                "",
                "| 指标 / Metric | 原因 / Reason |",
                "|------|------|",
            ]
        )
        for item in payload["skipped_metrics"]:
            md_lines.append(
                f"| {display_name(item['metric'])} | {item.get('reason', '')} |"
            )
    empty_rows = [r for r in payload.get("samples", []) if r.get("answer_empty")]
    if empty_rows:
        md_lines.extend(
            [
                "",
                "### 空答案样本 / Samples with empty answers",
                "",
                "| 问题 / Question | 来源数 / Sources |",
                "|------|------|",
            ]
        )
        for r in empty_rows:
            md_lines.append(f"| {r['question'][:60]} | {r.get('sources_count', 0)} |")
    if payload.get("eval_notes"):
        md_lines.extend(
            [
                "",
                "### 评估过程日志 / Evaluation Notes",
                "",
                "```",
            ]
        )
        md_lines.extend(payload["eval_notes"])
        md_lines.append("```")
    md_lines.extend(
        [
            "",
            "## 原始结果 / Raw Results",
            "",
            "```json",
            json.dumps(payload, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path, md_path


def check_gate(
    averages: dict[str, float],
    thresholds: dict[str, float],
) -> tuple[bool, list[str]]:
    """门禁判定：任一已配置指标低于阈值即失败，返回 (通过?, 失败明细)"""
    if not thresholds:
        return True, []
    failures = [
        f"{display_name(k)}={averages.get(k, 0.0):.4f} < 阈值 {v}"
        for k, v in sorted(thresholds.items())
        if averages.get(k, 0.0) < v
    ]
    return (not failures), failures
