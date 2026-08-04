"""在线评估 - 生产流量采样 + LLM-as-judge 打分写回 Langfuse

用法：
    uv run python -m src.eval.online --limit 100 --sample-rate 0.1
    uv run python -m src.eval.online --dry-run        # 只采样不打分

流程：
1. 从 Langfuse 拉取最近带 "chat" 标签的 trace（可按时段/条数过滤）
2. 按采样比例抽取，跳过缓存命中的 trace（不评估回放质量）
3. 读取每条 trace 的 "request" span（query/answer/检索上下文/耗时）
4. 对无需 reference 的指标（faithfulness/answer_relevancy/context_precision）
   用独立 judge 打分，create_score 写回对应 trace
"""

from __future__ import annotations

import argparse
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Any

from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness
from ragas.run_config import RunConfig

from src.config.settings import settings
from src.eval.judge import get_judge_ragas_embeddings, get_judge_ragas_llm, judge_model_name
from src.eval.langfuse import get_langfuse_client, score_trace
from src.eval.metrics import display_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("eval.online")

ONLINE_METRICS = ["faithfulness", "answer_relevancy", "context_precision"]
REQUEST_SPAN_NAME = "request"
CACHE_TYPES = ("exact", "semantic")


def _fetch_recent_traces(limit: int, days: int) -> list[Any]:
    """拉取最近 N 天带 chat 标签的 trace 摘要"""
    client = get_langfuse_client()
    if client is None:
        raise RuntimeError("Langfuse 未配置（LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY）")
    from_timestamp = datetime.now(timezone.utc) - timedelta(days=days)
    resp = client.api.trace.list(
        page=1,
        limit=limit,
        from_timestamp=from_timestamp,
        tags=["chat"],
    )
    traces = resp.data or []
    logger.info("Langfuse 拉取最近 %d 天 trace: %d 条", days, len(traces))
    return traces


def _extract_request_span(trace: Any) -> dict[str, Any] | None:
    """从 trace 详情中提取 request span（input/output/metadata）"""
    observations = trace.observations or []
    for obs in observations:
        if obs.name == REQUEST_SPAN_NAME:
            return {
                "query": (obs.input or {}).get("query", "") if isinstance(obs.input, dict) else "",
                "answer": (obs.output or {}).get("answer", "") if isinstance(obs.output, dict) else "",
                "sources": (obs.output or {}).get("sources", []) if isinstance(obs.output, dict) else [],
                "latency_seconds": (obs.output or {}).get("latency_seconds") if isinstance(obs.output, dict) else None,
                "cache_type": (obs.output or {}).get("cache_type"),
            }
    return None


def sample_traces(
    trace_ids: list[str],
    *,
    sample_rate: float,
    min_count: int,
    max_count: int,
) -> list[str]:
    """按比例采样（保底 min_count，最多 max_count）"""
    if not trace_ids:
        return []
    n = max(min(int(len(trace_ids) * sample_rate), max_count), min(min_count, len(trace_ids)))
    return random.sample(trace_ids, n)


def _build_online_samples(trace_ids: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """读取 request span，组装可打分样本；返回 (样本, 跳过原因)"""
    client = get_langfuse_client()
    samples: list[dict[str, Any]] = []
    skipped: list[str] = []
    for trace_id in trace_ids:
        try:
            trace = client.api.trace.get(trace_id)
            span = _extract_request_span(trace)
            if not span:
                skipped.append(f"{trace_id}: 缺少 request span")
                continue
            if span["cache_type"] in CACHE_TYPES:
                skipped.append(f"{trace_id}: 缓存命中（{span['cache_type']}）")
                continue
            if not span["answer"]:
                skipped.append(f"{trace_id}: 答案为空")
                continue
            samples.append({"trace_id": trace_id, **span})
        except Exception as e:  # noqa: BLE001
            skipped.append(f"{trace_id}: {e}")
    return samples, skipped


def _score_samples(samples: list[dict[str, Any]]) -> dict[str, list[float | None]]:
    """对在线样本批量计算 3 个无需 reference 的指标"""
    if not samples:
        return {m: [] for m in ONLINE_METRICS}
    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=s["query"],
                response=s["answer"],
                retrieved_contexts=[c.get("content", "") for c in s.get("sources") or []],
            )
            for s in samples
        ]
    )
    from ragas import evaluate

    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(llm=get_judge_ragas_llm()),
            AnswerRelevancy(llm=get_judge_ragas_llm(), embeddings=get_judge_ragas_embeddings()),
            ContextPrecision(llm=get_judge_ragas_llm()),
        ],
        llm=get_judge_ragas_llm(),
        embeddings=get_judge_ragas_embeddings(),
        raise_exceptions=False,
        show_progress=False,
        batch_size=4,
        run_config=RunConfig(max_retries=3, max_wait=10, timeout=120),
    )
    df = result.to_pandas()
    out: dict[str, list[float | None]] = {}
    for metric_id in ONLINE_METRICS:
        col = next(
            (c for c in df.columns if c == metric_id or c.startswith(metric_id + "(")),
            None,
        )
        if col is not None:
            out[metric_id] = [
                (None if v is None or (isinstance(v, float) and v != v) else float(v))
                for v in df[col].tolist()
            ]
        else:
            out[metric_id] = [None] * len(samples)
    return out


def run_online_eval(
    *,
    limit: int = 200,
    days: int = 7,
    sample_rate: float | None = None,
    min_count: int = 20,
    max_count: int = 50,
    dry_run: bool = False,
) -> dict[str, Any]:
    """执行一轮在线评估，返回摘要（打分数/跳过数/平均分）"""
    rate = sample_rate if sample_rate is not None else settings.eval_sample_rate
    traces = _fetch_recent_traces(limit=limit, days=days)
    trace_ids = [t.id for t in traces if t.id]
    sampled = sample_traces(trace_ids, sample_rate=rate, min_count=min_count, max_count=max_count)
    logger.info("采样 %d/%d 条（rate=%.2f）", len(sampled), len(trace_ids), rate)

    samples, skipped = _build_online_samples(sampled)
    logger.info("可打分样本 %d 条，跳过 %d 条", len(samples), len(skipped))

    raw = {} if dry_run else _score_samples(samples)
    written = 0
    per_trace: list[dict[str, Any]] = []
    for idx, s in enumerate(samples):
        scores = {m: ((raw.get(m) or [])[idx] if not dry_run else None) for m in ONLINE_METRICS}
        if not dry_run:
            for metric_id, score in scores.items():
                if score is not None and score_trace(s["trace_id"], name=metric_id, value=score):
                    written += 1
        per_trace.append(
            {
                "trace_id": s["trace_id"],
                "query": s["query"][:60],
                "latency_seconds": s.get("latency_seconds"),
                "scores": scores,
            }
        )

    averages = {}
    for metric_id in ONLINE_METRICS:
        vals = [row["scores"][metric_id] for row in per_trace if row["scores"][metric_id] is not None]
        if vals:
            averages[metric_id] = round(sum(vals) / len(vals), 4)

    summary = {
        "sampled": len(sampled),
        "scored": len(samples),
        "skipped": len(skipped),
        "written_scores": written,
        "averages": averages,
        "judge_model": judge_model_name(),
        "dry_run": dry_run,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Agentic RAG 在线评估（Langfuse 采样打分）")
    parser.add_argument("--limit", type=int, default=200, help="拉取 trace 上限")
    parser.add_argument("--days", type=int, default=7, help="回溯天数")
    parser.add_argument("--sample-rate", type=float, default=None, help="采样比例（默认 EVAL_SAMPLE_RATE）")
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--max-count", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="只采样不打分")
    args = parser.parse_args()

    summary = run_online_eval(
        limit=args.limit,
        days=args.days,
        sample_rate=args.sample_rate,
        min_count=args.min_count,
        max_count=args.max_count,
        dry_run=args.dry_run,
    )
    print("\n在线评估完成 / Online Evaluation Finished")
    print(f"  采样 / Sampled : {summary['sampled']}")
    print(f"  可打分 / Scored: {summary['scored']}（跳过 {summary['skipped']}）")
    print(f"  写回分数       : {summary['written_scores']}")
    print(f"  judge 模型     : {summary['judge_model']}")
    for metric_id, avg in summary["averages"].items():
        print(f"  {display_name(metric_id)}: {avg:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
