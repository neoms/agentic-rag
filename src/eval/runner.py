"""离线评估 runner - 标准数据集 + RAGAS 指标 + Langfuse 上传 + 双语报告 + 门禁

用法：
    uv run python -m src.eval.runner --dataset eval/datasets/smoke.jsonl
    uv run python -m src.eval.runner --dataset eval/datasets/smoke.jsonl --fake-scores
    uv run python -m src.eval.runner --dataset eval/datasets/v3.jsonl --gate

说明：
- 评估请求固定 use_cache=False（评估的是真实生成质量，缓存回放不参与评测）
- --fake-scores 为 stub 冒烟模式：不调用真实 RAG/LLM，指标为确定性假分数，
  用于验证流水线机制（采集/计算/报告/门禁/上传）
- 真实评估需要外网（DashScope + 独立 judge）；门禁失败时退出码非 0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
# 注意：ragas.metrics.collections.* 为 InstructorLLM 专用（0.4.x 中不兼容
# LangchainLLMWrapper），因此沿用 ragas.metrics 的 legacy 指标类；
# 弃用警告仅为提示，功能不受影响。
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, FactualCorrectness, Faithfulness  # noqa: E402
from ragas.run_config import RunConfig

from src.config.settings import settings
from src.eval.dataset import EvalSample, load_dataset, validate_dataset_file
from src.eval.judge import (
    get_judge_ragas_embeddings,
    get_judge_ragas_llm,
    judge_isolated,
    judge_model_name,
)
from src.eval.metrics import QUALITY_METRICS, display_name
from src.eval.report import (
    build_report_payload,
    check_gate,
    format_console_table,
    write_report,
)
from src.models.chat import AgenticChatRequest
from src.services.rag_service import FALLBACK_GENERATION_FAILED, FALLBACK_NO_DOCS, rag_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval.runner")

QUALITY_IDS = [m.id for m in QUALITY_METRICS]
_INVALID_ANSWERS = (FALLBACK_NO_DOCS, FALLBACK_GENERATION_FAILED)

StreamFn = Callable[[AgenticChatRequest], AsyncIterator[Any]]


def _metric_column(df, metric_id: str) -> str | None:
    """在 ragas 结果中查找指标列（兼容 "factual_correctness(mode=f1)" 这类后缀命名）"""
    for col in df.columns:
        if col == metric_id or col.startswith(metric_id + "("):
            return col
    return None


class _RagasNoteHandler(logging.Handler):
    """捕获 ragas 内部的错误/降级日志（其异常对象不外露），供报告使用"""

    def __init__(self, notes: list[str]):
        super().__init__(level=logging.WARNING)
        self.notes = notes

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            msg = str(record.msg)
        if msg and msg not in self.notes:
            self.notes.append(f"[{record.levelname}] {record.name}: {msg}")


# ============ 样本采集（SSE 事件消费） ============


async def _collect_one(request: AgenticChatRequest, stream_fn: StreamFn) -> dict[str, Any]:
    """消费一次流式响应，聚合 answer/sources/耗时/trace_id"""
    answer = ""
    sources: list[dict[str, Any]] = []
    ttft: float | None = None
    trace_id: str | None = None
    error: str | None = None
    t0 = time.perf_counter()
    try:
        async for event in stream_fn(request):
            if event.event == "token":
                if ttft is None:
                    ttft = round(time.perf_counter() - t0, 3)
                answer += event.data
            elif event.event == "source":
                try:
                    sources = json.loads(event.data)
                except (json.JSONDecodeError, TypeError):
                    pass
            elif event.event == "done":
                try:
                    payload = json.loads(event.data) if event.data else {}
                    trace_id = payload.get("trace_id") or None
                except (json.JSONDecodeError, TypeError):
                    pass
            elif event.event == "error":
                try:
                    error = json.loads(event.data).get("detail", "流内错误")
                except (json.JSONDecodeError, TypeError):
                    error = "流内错误"
    except Exception as e:  # noqa: BLE001 - 单样本失败不中断整轮评估
        error = f"{type(e).__name__}: {e}"
    elapsed = round(time.perf_counter() - t0, 3)
    return {
        "answer": answer,
        "sources": sources,
        "latency_seconds": elapsed,
        "ttft_seconds": ttft,
        "trace_id": trace_id,
        "error": error,
    }


def _build_request(question: str, flags: dict[str, Any]) -> AgenticChatRequest:
    return AgenticChatRequest(
        query=question,
        session_id=f"eval-{uuid.uuid4().hex[:12]}",
        use_cache=False,  # 评估真实生成质量，不测缓存回放
        enable_web_search=flags.get("web_search", False),
        enable_reflection=flags.get("reflection", True),
        enable_rerank=flags.get("rerank", True),
        enable_grade_documents=flags.get("grade_documents", True),
        enable_transform_query=flags.get("transform_query", True),
        enable_bm25=flags.get("bm25", True),
        enable_multi_query=flags.get("multi_query", False),
        enable_kg=flags.get("kg", False),
    )


def _fake_stream_fn(question: str):
    """stub 冒烟用流式事件（不调用真实 RAG/LLM）"""

    async def gen(_request: AgenticChatRequest):
        yield type("E", (), {"event": "token", "data": f"stub 答案：{question[:20]}..."})()
        yield type(
            "E", (), {
                "event": "source",
                "data": json.dumps(
                    [{"content": "stub 检索上下文", "metadata": {"filename": "stub.md"}, "score": 0.9}],
                    ensure_ascii=False,
                ),
            },
        )()
        yield type("E", (), {"event": "done", "data": json.dumps({"trace_id": ""})})()

    return gen


def collect_samples(
    dataset: list[EvalSample],
    flags: dict[str, Any],
    *,
    stream_fn: StreamFn | None = None,
    stub: bool = False,
) -> list[dict[str, Any]]:
    """逐条采集 RAG 输出；stub 模式返回确定性假数据"""
    results: list[dict[str, Any]] = []
    for i, sample in enumerate(dataset, 1):
        logger.info("[collect] %d/%d: %s", i, len(dataset), sample.question[:60])
        request = _build_request(sample.question, flags)
        if stub:
            collected = asyncio.run(_collect_one(request, _fake_stream_fn(sample.question)))
        else:
            collected = asyncio.run(_collect_one(request, stream_fn or rag_service.agentic_rag_stream))
        results.append(
            {
                "question": sample.question,
                "reference": sample.reference,
                "reference_contexts": sample.reference_contexts,
                **collected,
            }
        )
    return results


# ============ 指标计算（RAGAS） ============


def _build_ragas_metrics(with_reference_contexts: bool) -> list:
    llm = get_judge_ragas_llm()
    embeddings = get_judge_ragas_embeddings()
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        FactualCorrectness(llm=llm, language="chinese"),
    ]
    if with_reference_contexts:
        metrics.extend(
            [
                ContextPrecision(llm=llm),
                ContextRecall(llm=llm),
            ]
        )
    return metrics


def _run_ragas(
    collected: list[dict[str, Any]],
    with_ref_ctx: bool,
) -> tuple[dict[str, list[float | None]], list[str]]:
    """对一批样本跑 RAGAS，返回 ({metric_id: [score...]}, 过程日志)（顺序与 collected 一致）"""
    def _effective_answer(row: dict[str, Any]) -> str:
        """兜底/失败文案不作为有效答案参与评分（返回空串）"""
        answer = (row.get("answer") or "").strip()
        return "" if answer in _INVALID_ANSWERS else answer

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=row["question"],
                response=_effective_answer(row),
                retrieved_contexts=[s.get("content", "") for s in row.get("sources") or []],
                reference_contexts=row.get("reference_contexts") or [],
                reference=row.get("reference") or "",
            )
            for row in collected
        ]
    )
    result, notes = _evaluate_ragas(dataset, with_ref_ctx)
    if result is None:
        return {}, notes
    df = result.to_pandas()
    out: dict[str, list[float | None]] = {}
    for metric_id in QUALITY_IDS:
        col = _metric_column(df, metric_id)
        if col is not None:
            out[metric_id] = [
                (None if v is None or (isinstance(v, float) and v != v) else float(v))
                for v in df[col].tolist()
            ]
    return out, notes


def _evaluate_ragas(
    dataset: EvaluationDataset,
    with_ref_ctx: bool,
) -> tuple[Any | None, list[str]]:
    """包装 ragas.evaluate，捕获其过程日志（异常对象不外露，经日志捕获）

    返回 (EvaluationResult | None, notes)。notes 为 ragas 内部 WARNING/ERROR
    日志（如 judge 调用失败、n 代降级等），用于报告"指标失败原因"。
    """
    from ragas import evaluate

    notes: list[str] = []
    handler = _RagasNoteHandler(notes)
    logging.getLogger("ragas").addHandler(handler)
    try:
        result = evaluate(
            dataset,
            metrics=_build_ragas_metrics(with_reference_contexts=with_ref_ctx),
            llm=get_judge_ragas_llm(),
            embeddings=get_judge_ragas_embeddings(),
            raise_exceptions=False,
            show_progress=False,
            batch_size=4,
            run_config=RunConfig(max_retries=3, max_wait=10, timeout=120),
        )
        return result, notes
    except Exception as e:  # noqa: BLE001
        logger.error("RAGAS evaluate 失败: %s", e, exc_info=True)
        return None, notes
    finally:
        logging.getLogger("ragas").removeHandler(handler)


def _fake_scores(collected: list[dict[str, Any]]) -> dict[str, list[float | None]]:
    """stub 冒烟用确定性假分数（0.6~0.95 区间，可复现）"""
    out: dict[str, list[float | None]] = {}
    for metric_id in QUALITY_IDS:
        scores: list[float | None] = []
        for row in collected:
            if metric_id in ("context_precision", "context_recall") and not row.get("reference_contexts"):
                scores.append(None)
            else:
                seed = sum(ord(c) for c in row["question"])
                scores.append(round(0.62 + (seed % 34) / 100.0, 3))
        out[metric_id] = scores
    return out


def compute_scores(
    collected: list[dict[str, Any]],
    *,
    fake: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, float], list[dict[str, Any]], list[str]]:
    """计算每样本分数 + 各指标平均 + 跳过/失败说明 + ragas 过程日志"""
    notes: list[str] = []
    if fake:
        raw = _fake_scores(collected)
    else:
        with_ref = [r for r in collected if r.get("reference_contexts")]
        without_ref = [r for r in collected if not r.get("reference_contexts")]
        group_scores: dict[int, dict[str, list[float | None]]] = {}
        if with_ref:
            scores1, notes1 = _run_ragas(with_ref, True)
            group_scores[1] = scores1
            notes.extend(notes1)
        if without_ref:
            scores0, notes0 = _run_ragas(without_ref, False)
            group_scores[0] = scores0
            notes.extend(notes0)

        # 合并回原始顺序
        merged: dict[str, list[float | None]] = {m: [] for m in QUALITY_IDS}
        for row in collected:
            group = group_scores[1] if row.get("reference_contexts") else group_scores[0]
            for metric_id in QUALITY_IDS:
                scores = group.get(metric_id) or []
                merged[metric_id].append(scores.pop(0) if scores else None)
        raw = merged

    per_sample: list[dict[str, Any]] = []
    for idx, row in enumerate(collected):
        scores = {
            metric_id: (raw.get(metric_id) or [None] * len(collected))[idx]
            for metric_id in QUALITY_IDS
        }
        scored = [v for v in scores.values() if v is not None]
        per_sample.append(
            {
                "question": row["question"],
                "reference": row.get("reference", ""),
                "reference_contexts": row.get("reference_contexts", []),
                "scores": scores,
                "avg": round(sum(scored) / len(scored), 4) if scored else None,
                "latency_seconds": row.get("latency_seconds"),
                "ttft_seconds": row.get("ttft_seconds"),
                "answer_length": len(row.get("answer") or ""),
                "answer_empty": (
                    not (row.get("answer") or "").strip()
                    or (row.get("answer") or "").strip() in _INVALID_ANSWERS
                ),
                "sources_count": len(row.get("sources") or []),
                "trace_id": row.get("trace_id"),
                "error": row.get("error"),
            }
        )

    averages: dict[str, float] = {}
    for metric_id in QUALITY_IDS:
        vals = [
            row["scores"][metric_id]
            for row in per_sample
            if row["scores"][metric_id] is not None
        ]
        if vals:
            averages[metric_id] = round(sum(vals) / len(vals), 4)

    skipped: list[dict[str, Any]] = []
    for metric_id in QUALITY_IDS:
        if metric_id in ("context_precision", "context_recall") and not any(
            r.get("reference_contexts") for r in collected
        ):
            skipped.append(
                {
                    "metric": metric_id,
                    "reason": "数据集缺少 reference_contexts（golden 上下文）",
                }
            )
        elif not averages.get(metric_id):
            skipped.append(
                {
                    "metric": metric_id,
                    "reason": "全部样本计算失败（judge 调用异常或解析失败，详见评估日志）",
                }
            )
    return per_sample, averages, skipped, notes


# ============ Langfuse 上传 ============


def upload_dataset_and_scores(
    dataset_name: str,
    dataset: list[EvalSample],
    per_sample: list[dict[str, Any]],
    run_name: str,
) -> str | None:
    """上传数据集（幂等）与每 trace 得分，返回 trace url 前缀或 None"""
    from src.eval.langfuse import get_langfuse_client, score_trace

    client = get_langfuse_client()
    if client is None:
        return None
    try:
        try:
            ds = client.get_dataset(dataset_name)
            logger.info("Langfuse 数据集 '%s' 已存在（%d items）", dataset_name, len(ds.items))
        except Exception:
            ds = client.create_dataset(
                name=dataset_name,
                description="Agentic RAG 标准评估数据集（question/reference/reference_contexts）",
            )
            for s in dataset:
                client.create_dataset_item(
                    dataset_name=dataset_name,
                    input={"question": s.question},
                    expected_output={"reference": s.reference},
                    metadata={"reference_contexts": s.reference_contexts},
                )
            logger.info("Langfuse 数据集 '%s' 创建完成，共 %d 条", dataset_name, len(dataset))

        scored = 0
        for row in per_sample:
            trace_id = row.get("trace_id")
            if not trace_id:
                continue
            for metric_id, score in (row.get("scores") or {}).items():
                if score is not None:
                    if score_trace(trace_id, name=metric_id, value=score):
                        scored += 1
        logger.info("Langfuse 得分写入: %d 条（run=%s）", scored, run_name)
        client.flush()
        return client.get_trace_url(trace_id=per_sample[0]["trace_id"]) if per_sample[0].get("trace_id") else None
    except Exception as e:  # noqa: BLE001
        logger.warning("Langfuse 上传失败（不影响本地评估）: %s", e)
        return None


# ============ 主流程 ============


def run_offline_eval(
    dataset_path: Path,
    *,
    output_dir: Path,
    dataset_name: str,
    run_name: str,
    flags: dict[str, Any],
    fake: bool = False,
    gate: bool = False,
    upload: bool = True,
    stream_fn: StreamFn | None = None,
    stub: bool = False,
) -> dict[str, Any]:
    """执行一轮离线评估，返回摘要（含 gate 结果）"""
    errors = validate_dataset_file(dataset_path)
    if errors:
        raise ValueError("数据集校验失败:\n" + "\n".join(f"  - {e}" for e in errors))
    dataset = load_dataset(dataset_path)

    logger.info("加载数据集: %s（%d 条）", dataset_path, len(dataset))
    collected = collect_samples(dataset, flags, stream_fn=stream_fn, stub=stub)
    per_sample, averages, skipped, notes = compute_scores(collected, fake=fake)

    langfuse_url = None
    if upload and not fake:
        langfuse_url = upload_dataset_and_scores(dataset_name, dataset, per_sample, run_name)

    payload = build_report_payload(
        dataset_name=dataset_name,
        dataset_path=str(dataset_path),
        run_name=run_name,
        judge_model=judge_model_name(),
        judge_isolated=judge_isolated(),
        samples_count=len(dataset),
        strategy=flags,
        per_sample=per_sample,
        averages=averages,
        skipped=skipped,
        notes=notes,
        langfuse_url=langfuse_url,
    )
    json_path, md_path = write_report(output_dir, payload)

    gate_passed, failures = check_gate(averages, settings.eval_gate_thresholds_dict)
    summary = {
        "dataset": dataset_name,
        "samples": len(dataset),
        "averages": averages,
        "per_sample": per_sample,
        "skipped": skipped,
        "notes": notes,
        "gate_enabled": gate,
        "gate_passed": gate_passed,
        "gate_failures": failures,
        "report_json": str(json_path),
        "report_md": str(md_path),
        "langfuse_url": langfuse_url,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Agentic RAG 标准评估 runner")
    parser.add_argument("--dataset", required=True, help="数据集 JSONL 路径")
    parser.add_argument("--output-dir", default="eval/results", help="报告输出目录")
    parser.add_argument("--name", default="smoke", help="数据集/运行名")
    parser.add_argument("--fake-scores", action="store_true", help="stub 冒烟：确定性假分数，不调用真实 LLM")
    parser.add_argument("--gate", action="store_true", help="按 EVAL_GATE_THRESHOLDS 判定，失败退出码非 0")
    parser.add_argument("--no-langfuse", action="store_true", help="不上传 Langfuse")
    parser.add_argument("--enable-web-search", action="store_true")
    parser.add_argument("--disable-reflection", action="store_true")
    parser.add_argument("--disable-rerank", action="store_true")
    parser.add_argument("--disable-grade", action="store_true")
    parser.add_argument("--disable-transform-query", action="store_true")
    parser.add_argument("--disable-bm25", action="store_true")
    parser.add_argument("--enable-multi-query", action="store_true")
    parser.add_argument("--enable-kg", action="store_true")
    args = parser.parse_args()

    flags = {
        "web_search": args.enable_web_search,
        "reflection": not args.disable_reflection,
        "rerank": not args.disable_rerank,
        "grade_documents": not args.disable_grade,
        "transform_query": not args.disable_transform_query,
        "bm25": not args.disable_bm25,
        "multi_query": args.enable_multi_query,
        "kg": args.enable_kg,
    }
    fake = args.fake_scores or settings.eval_stub_llm

    summary = run_offline_eval(
        Path(args.dataset),
        output_dir=Path(args.output_dir),
        dataset_name=args.name,
        run_name=args.name,
        flags=flags,
        fake=fake,
        gate=args.gate,
        upload=not args.no_langfuse,
        stub=fake,
    )

    print("\n" + "=" * 60)
    print("Agentic RAG 评估完成 / Evaluation Finished")
    print(f"  数据集 / Dataset : {summary['dataset']}（{summary['samples']} 条）")
    print(f"  judge 模型       : {judge_model_name()}" + ("（独立 / isolated）" if judge_isolated() else "（与被测同源 / same as generator）"))
    print("=" * 60)
    print(format_console_table(summary["averages"], summary.get("skipped", []), summary.get("per_sample", [])))

    # gate 判定（冒烟模式同样校验机制；阈值留空时恒通过）
    if args.gate:
        if summary["gate_passed"]:
            print("\n门禁通过 / Gate passed ✔")
        else:
            print("\n门禁未通过 / Gate failed ✘")
            for f in summary["gate_failures"]:
                print(f"  - {f}")
            return 1

    print(f"\n报告 / Report: {summary['report_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
