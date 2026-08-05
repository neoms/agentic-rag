"""Locust 压测 - Agentic RAG /api/v1/chat/stream（SSE 流式）

用法（冒烟，stub/缓存命中为主，控制成本）：
    cd eval/load
    locust -f locustfile.py --host http://localhost:8000 \
        --headless -u 5 -r 1 -t 60s

说明：
- 默认使用小问题池随机提问：首轮未命中并写回缓存，之后多为缓存命中，成本低
- 设置 EVAL_LOAD_UNIQUE=1 时每条请求追加随机后缀，强制未命中（真实生成压测）
- 设置 EVAL_LOAD_WAIT_MIN/MAX 可收紧用户思考间隔（默认 0.5~2.0s 模拟真实节奏；
  测缓存命中极限 QPS 时可设 0/0.05 持续打满）
- 报告指标：吞吐 QPS、端到端延迟 p50/p95/p99、首 token 延迟（TTFT）p50/p95/p99、
  缓存命中率（精准/语义/未命中分布）、分阶段耗时（节点 durationMs）、
  单请求成本估算（best-effort：测试前后拉取 /metrics 中 llm_tokens_total
  增量 × LLM_PRICE_INPUT/OUTPUT_PER_1M 单价；/metrics 不可达或未配置单价时标注 N/A）
- 阈值断言在测试结束时执行（test_stop）：
    EVAL_LOAD_P95_MAX       默认 10.0（秒）
    EVAL_LOAD_ERROR_RATE_MAX 默认 0.01
  失败时进程退出码非 0

安全频率模式（避免触发重排服务的偶发长尾惩罚）：
- EVAL_LOAD_PACED=1：QPS 视为受控频率下测得、不代表容量上限，
  报告中标注“未测容量吞吐”；其余逐请求指标（延迟/TTFT/分阶段/缓存/成本）照常输出
- 低频不并发模式推荐命令（串行 + 数秒间隔，采集“完全正常响应”下的数据）：
    EVAL_LOAD_PACED=1 EVAL_LOAD_WAIT_MIN=3 EVAL_LOAD_WAIT_MAX=6 \\
    locust -f locustfile.py --host http://localhost:8000 --headless -u 1 -r 1 -t 300s
  - 启动时会校验：并发用户 >1 或 WAIT_MIN <2s 时给出警告
  - 可用 EVAL_LOAD_QUERIES='["q1","q2",...]'（JSON 数组）追加问题池，
    让部分请求未命中缓存以采集完整流水线指标
"""

from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from locust import HttpUser, between, events, task

# 确保 eval/load 在 sys.path 上（locust -f 运行时目录已在，双保险）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_helpers import (  # noqa: E402
    STAGE_ORDER,
    iter_sse_events,
    parse_token_lines,
    percentile,
    stage_label,
)

_DEFAULT_QUERIES = [
    "Agentic RAG 系统使用什么作为向量数据库和知识图谱存储引擎？",
    "语义检索与 BM25 关键词检索各自的优势是什么？",
    "知识图谱模块中 GraphStore 的作用是什么？",
    "MMR 算法的 lambda_mult 参数取 0.7 意味着什么？",
    "小象科技成立于哪一年，总部位于哪里？",
]

# EVAL_LOAD_QUERIES：JSON 数组，追加自定义问题到问题池（配合安全频率模式
# 采集未命中缓存时的完整流水线指标）
_EXTRA_QUERIES: list[str] = []
_extra_raw = os.environ.get("EVAL_LOAD_QUERIES", "")
if _extra_raw.strip():
    try:
        _parsed = json.loads(_extra_raw)
        if not isinstance(_parsed, list) or not all(
            isinstance(q, str) and q.strip() for q in _parsed
        ):
            raise ValueError("必须是字符串数组 / expected a JSON array of strings")
        _EXTRA_QUERIES = _parsed
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[perf] EVAL_LOAD_QUERIES 解析失败 / parse failed: {e}")

# EVAL_LOAD_EXCLUDE=mmr,xxx 时按子串过滤问题池（如排除因幻觉门控未缓存的问题）
_EXCLUDE = [
    s.strip().lower()
    for s in os.environ.get("EVAL_LOAD_EXCLUDE", "").split(",")
    if s.strip()
]
QUERIES = [
    q for q in _DEFAULT_QUERIES
    if not any(part in q.lower() for part in _EXCLUDE)
] or _DEFAULT_QUERIES
QUERIES = QUERIES + _EXTRA_QUERIES

FORCE_UNIQUE = os.environ.get("EVAL_LOAD_UNIQUE", "") == "1"
# 安全频率模式：QPS 只反映受控频率，不代表容量上限
PACED = os.environ.get("EVAL_LOAD_PACED", "") == "1"

_lock = threading.Lock()
# 每条成功请求的记录：latency/ttft/cache_type/节点耗时/答案长度
_records: list[dict] = []
_start_ts = time.perf_counter()
_start_wall = datetime.now(timezone.utc)
# 测试开始时的 /metrics token 计数基线（best-effort，用于成本估算）
_start_tokens: dict[tuple[str, str], float] | None = None

# 性能测试报告目录（与质量报告同根目录，独立子目录）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(
    os.environ.get("EVAL_LOAD_REPORT_DIR", str(PROJECT_ROOT / "eval" / "results" / "perf"))
)

def _fetch_token_counts(host: str) -> dict[tuple[str, str], float] | None:
    """best-effort 拉取 /metrics 中 llm_tokens_total{model,type} 计数

    返回 None 表示端点不可达（成本估算标 N/A）；端点可达但无该指标
    （如纯缓存命中轮次未发生 LLM 调用）返回空 dict，按 0 处理。
    """
    if not host:
        return None
    url = host.rstrip("/") + "/metrics"
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=10) as resp:
            text = resp.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001 - 成本估算为可选项，失败不影响压测
        print(f"[perf] /metrics 拉取失败（{url}）: {type(e).__name__}: {e}")
        return None
    return parse_token_lines(text)


class RagChatUser(HttpUser):
    wait_time = between(
        float(os.environ.get("EVAL_LOAD_WAIT_MIN", "0.5")),
        float(os.environ.get("EVAL_LOAD_WAIT_MAX", "2.0")),
    )

    @task
    def chat_stream(self) -> None:
        query = random.choice(QUERIES)
        if FORCE_UNIQUE:
            query = f"{query}（{uuid.uuid4().hex[:6]}）"
        t0 = time.perf_counter()
        first_token_at: float | None = None
        answer_chars = 0
        cache_type = "none"
        exact_hit = semantic_hit = False
        node_durations: dict[str, float] = {}
        try:
            with self.client.post(
                "/api/v1/chat/stream",
                json={
                    "query": query,
                    "use_cache": True,
                    "enable_web_search": False,
                    "enable_reflection": True,
                    "enable_rerank": True,
                    "enable_transform_query": True,
                    "enable_bm25": True,
                    "enable_multi_query": False,
                    "enable_kg": False,
                },
                headers={"Content-Type": "application/json"},
                stream=True,
                catch_response=True,
                name="/api/v1/chat/stream",
            ) as resp:
                saw_done = False
                for ev, data in iter_sse_events(resp.iter_lines()):
                    if ev == "token":
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        answer_chars += len(data)
                    elif ev == "node_data":
                        try:
                            payload = json.loads(data)
                        except (json.JSONDecodeError, TypeError):
                            continue
                        for node_id, node in (payload or {}).items():
                            if not isinstance(node, dict):
                                continue
                            dur = node.get("durationMs")
                            if isinstance(dur, (int, float)):
                                node_durations[node_id] = float(dur)
                            if node_id == "cache_lookup":
                                out = node.get("output") or {}
                                if out.get("hit"):
                                    cache_type = out.get("cache_type") or "exact"
                                exact_hit = bool(out.get("exact_hit"))
                                semantic_hit = bool(out.get("semantic_hit"))
                    elif ev == "done":
                        saw_done = True
                        break
                latency = time.perf_counter() - t0
                if not saw_done:
                    raise RuntimeError("SSE 流未收到 done 事件")
                with _lock:
                    _records.append(
                        {
                            "latency": latency,
                            "ttft": (
                                first_token_at - t0
                                if first_token_at is not None
                                else None
                            ),
                            "cache_type": cache_type,
                            "exact_hit": exact_hit,
                            "semantic_hit": semantic_hit,
                            "node_durations": dict(node_durations),
                            "answer_chars": answer_chars,
                        }
                    )
        except Exception:  # noqa: BLE001 - 任务异常由 Locust 自动记为失败
            raise


@events.test_start.add_listener
def _on_test_start(environment, **_kwargs) -> None:  # noqa: ANN001
    global _start_tokens
    host = environment.host or ""
    _start_tokens = _fetch_token_counts(host)
    if _start_tokens is None:
        print(
            "[perf] 提示 / Note: 无法读取 /metrics 基线，"
            "成本估算将标注 N/A（cost estimate unavailable）"
        )
    if PACED:
        # 低频不并发模式约束提示：单用户 + 宽松思考间隔才能保证串行、低频
        users = (
            getattr(environment.runner, "target_user_count", None)
            or getattr(environment.runner, "user_count", None)
            or 0
        )
        wait_min = float(os.environ.get("EVAL_LOAD_WAIT_MIN", "0.5"))
        if users > 1:
            print(
                f"[perf] 警告 / WARNING: EVAL_LOAD_PACED=1 但并发用户={users}，"
                "低频不并发模式请使用 -u 1（当前仍可能触发上游限速惩罚）"
            )
        if wait_min < 2:
            print(
                f"[perf] 警告 / WARNING: EVAL_LOAD_PACED=1 但 EVAL_LOAD_WAIT_MIN={wait_min}s 过低，"
                "建议设置 EVAL_LOAD_WAIT_MIN/MAX=3/6，数秒间隔才能真正避开限速惩罚"
            )


@events.test_stop.add_listener
def _on_test_stop(environment, **_kwargs) -> None:  # noqa: ANN001
    """测试结束：计算双语汇总并执行阈值断言"""
    with _lock:
        records = list(_records)
    duration = max(time.perf_counter() - _start_ts, 1e-6)
    total = environment.runner.stats.total
    requests = total.num_requests
    failures = total.num_failures
    error_rate = failures / max(requests, 1)

    latencies = sorted(r["latency"] for r in records)
    ttfts = sorted(r["ttft"] for r in records if r["ttft"] is not None)
    qps_observed = len(records) / duration
    # 受控频率模式下不测容量吞吐：QPS 仅作为观测到的请求节奏保留，
    # 报告中标注“未测容量吞吐”
    qps = None if PACED else round(qps_observed, 3)
    qps_note = (
        "受控频率模式下测得，非容量上限 / paced mode, capacity QPS not measured"
        if PACED
        else None
    )

    p50 = percentile(latencies, 0.50)
    p95 = percentile(latencies, 0.95)
    p99 = percentile(latencies, 0.99)
    ttft_p50 = percentile(ttfts, 0.50)
    ttft_p95 = percentile(ttfts, 0.95)
    ttft_p99 = percentile(ttfts, 0.99)

    # 缓存命中分布（cache_type 来自 cache_lookup 节点输出）
    cache_exact = sum(1 for r in records if r["cache_type"] == "exact")
    cache_semantic = sum(1 for r in records if r["cache_type"] == "semantic")
    cache_miss = sum(1 for r in records if r["cache_type"] == "none")
    cache_unknown = len(records) - cache_exact - cache_semantic - cache_miss
    cache_hit_total = cache_exact + cache_semantic
    cache_hit_rate = cache_hit_total / max(len(records), 1)

    # 分阶段耗时（聚合所有请求中出现的节点 durationMs）
    stage_acc: dict[str, dict] = {}
    for r in records:
        for nid, dur_ms in r["node_durations"].items():
            acc = stage_acc.setdefault(nid, {"count": 0, "sum_ms": 0.0, "vals": []})
            acc["count"] += 1
            acc["sum_ms"] += dur_ms
            acc["vals"].append(dur_ms)
    stage_rows: list[dict] = []
    for nid, acc in stage_acc.items():
        vals = sorted(acc["vals"])
        stage_rows.append(
            {
                "stage": nid,
                "count": acc["count"],
                "mean_ms": round(acc["sum_ms"] / acc["count"], 1),
                "p95_ms": round(percentile(vals, 0.95), 1),
            }
        )
    stage_rows.sort(
        key=lambda x: (
            STAGE_ORDER.index(x["stage"]) if x["stage"] in STAGE_ORDER else 999,
            x["stage"],
        )
    )

    # TTFT 按缓存命中/未命中拆分
    ttft_hit = sorted(
        r["ttft"] for r in records if r["ttft"] is not None and r["cache_type"] != "none"
    )
    ttft_miss = sorted(
        r["ttft"] for r in records if r["ttft"] is not None and r["cache_type"] == "none"
    )

    # 成本估算（best-effort：/metrics token 增量 × 单价）
    end_tokens = _fetch_token_counts(environment.host or "")
    metrics_ok = _start_tokens is not None and end_tokens is not None
    token_delta: dict[tuple[str, str], float] = {}
    if metrics_ok:
        for key, value in end_tokens.items():  # type: ignore[union-attr]
            token_delta[key] = value - _start_tokens.get(key, 0.0)  # type: ignore[union-attr]
    price_in = float(os.environ.get("LLM_PRICE_INPUT_PER_1M", "0") or 0)
    price_out = float(os.environ.get("LLM_PRICE_OUTPUT_PER_1M", "0") or 0)
    tokens_in = sum(v for (_, typ), v in token_delta.items() if typ == "input")
    tokens_out = sum(v for (_, typ), v in token_delta.items() if typ == "output")
    cost_total = sum(
        v * (price_in if typ == "input" else price_out) / 1_000_000
        for (_, typ), v in token_delta.items()
    )
    cost_per_query = cost_total / max(len(records), 1)
    prices_configured = price_in > 0 or price_out > 0

    p95_max = float(os.environ.get("EVAL_LOAD_P95_MAX", "10.0"))
    error_max = float(os.environ.get("EVAL_LOAD_ERROR_RATE_MAX", "0.01"))
    failures_list: list[str] = []
    if p95 > p95_max:
        failures_list.append(f"p95 latency（p95 延迟）={p95:.3f}s > {p95_max}s")
    if error_rate > error_max:
        failures_list.append(f"error rate（错误率）={error_rate:.4f} > {error_max}")
    passed = not failures_list
    if failures_list:
        environment.process_exit_code = 1

    print("\n" + "=" * 64)
    print("压测结果 / Load Test Summary")
    print(f"  请求数 / Requests          : {requests}")
    print(f"  失败数 / Failures          : {failures}")
    print(f"  错误率 / Error rate        : {error_rate:.4f}")
    if PACED:
        print("  吞吐 / QPS                 : —（受控频率模式，未测容量吞吐 / paced, not measured）")
    else:
        print(f"  吞吐 / QPS                 : {qps_observed:.2f}")
    print(f"  延迟 p50 / latency p50     : {p50:.3f}s")
    print(f"  延迟 p95 / latency p95     : {p95:.3f}s")
    print(f"  延迟 p99 / latency p99     : {p99:.3f}s")
    print(
        f"  首 token 延迟 / TTFT        : p50 {ttft_p50:.3f}s / "
        f"p95 {ttft_p95:.3f}s / p99 {ttft_p99:.3f}s（{len(ttfts)} 样本）"
    )
    print(
        f"  缓存命中率 / Cache hit rate : {cache_hit_rate:.2%}"
        f"（精准 {cache_exact} / 语义 {cache_semantic} / 未命中 {cache_miss}）"
    )
    if metrics_ok and prices_configured:
        print(f"  单请求成本 / Cost per query : {cost_per_query:.6f} 元（CNY）")
    elif metrics_ok:
        print("  单请求成本 / Cost per query : —（未配置单价 / prices not configured）")
    else:
        print("  单请求成本 / Cost per query : N/A（/metrics 不可达）")
    print("=" * 64)
    if passed:
        print("阈值通过 / Threshold check PASSED ✔")
    else:
        print("阈值未通过 / Threshold check FAILED ✘")
        for f in failures_list:
            print(f"  - {f}")

    payload = {
        "timestamp": _start_wall.isoformat(),
        "duration_seconds": round(duration, 3),
        "users": _resolve_user_count(environment),
        "requests": requests,
        "failures": failures,
        "error_rate": round(error_rate, 6),
        "qps": qps,
        "qps_note": qps_note,
        "latency_p50_s": round(p50, 4),
        "latency_p95_s": round(p95, 4),
        "latency_p99_s": round(p99, 4),
        "ttft_p50_s": round(ttft_p50, 4),
        "ttft_p95_s": round(ttft_p95, 4),
        "ttft_p99_s": round(ttft_p99, 4),
        "ttft_samples": len(ttfts),
        "ttft_missing": len(records) - len(ttfts),
        "cache": {
            "total": len(records),
            "exact": cache_exact,
            "semantic": cache_semantic,
            "miss": cache_miss,
            "unknown": cache_unknown,
            "hit_rate": round(cache_hit_rate, 6),
        },
        "stage_stats": stage_rows,
        "cost": {
            "available": metrics_ok,
            "prices_configured": prices_configured,
            "tokens_input": round(tokens_in, 1),
            "tokens_output": round(tokens_out, 1),
            "total_cny": round(cost_total, 6) if prices_configured else None,
            "per_query_cny": round(cost_per_query, 6) if prices_configured else None,
            "price_input_per_1m": price_in,
            "price_output_per_1m": price_out,
        },
        "threshold_p95_max_s": p95_max,
        "threshold_error_rate_max": error_max,
        "threshold_passed": passed,
        "threshold_failures": failures_list,
        "config": {
            "force_unique": FORCE_UNIQUE,
            "paced": PACED,
            "wait_min": float(os.environ.get("EVAL_LOAD_WAIT_MIN", "0.5")),
            "wait_max": float(os.environ.get("EVAL_LOAD_WAIT_MAX", "2.0")),
            "queries_count": len(QUERIES),
            "extra_queries": len(_EXTRA_QUERIES),
        },
    }
    _write_report(environment, payload, passed, failures_list)


def _resolve_user_count(environment) -> int | None:  # noqa: ANN001
    """test_stop 时 user_count 可能已归零，优先取目标并发数"""
    runner = environment.runner
    users = getattr(runner, "target_user_count", None)
    if users is None:
        users = getattr(runner, "user_count", None)
    return users


def _write_report(
    environment,
    payload: dict,
    passed: bool,
    failures_list: list[str],
) -> None:
    """将压测结果写入 eval/results/perf/（JSON + 双语 MD）"""
    try:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        ts = _start_wall.strftime("%Y%m%d-%H%M%S")
        json_path = REPORT_DIR / f"perf-{ts}.json"
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        md_path = REPORT_DIR / f"perf-{ts}.md"
        md = _build_md(payload, passed, failures_list)
        md_path.write_text(md, encoding="utf-8")
        print(f"报告已写入 / Report written: {json_path}")
        print(f"                            {md_path}")
    except Exception as e:  # noqa: BLE001 - 报告写入失败不阻断压测结论
        print(f"报告写入失败 / Report write failed: {e}")


def _fmt_s(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}s"


def _fmt_ms(v: float) -> str:
    return f"{v:.1f}ms"


def _build_md(payload: dict, passed: bool, failures_list: list[str]) -> str:
    """双语 Markdown 报告"""
    status = "PASSED ✔" if passed else "FAILED ✘"
    cache = payload["cache"]
    cost = payload["cost"]
    ttft_samples = payload["ttft_samples"]
    ttft_label = (
        f"{_fmt_s(payload['ttft_p50_s'])} / {_fmt_s(payload['ttft_p95_s'])} / "
        f"{_fmt_s(payload['ttft_p99_s'])}（{ttft_samples} 样本）"
        if ttft_samples
        else "—（0 样本）"
    )
    if cost["available"] and cost["prices_configured"]:
        cost_label = (
            f"{cost['per_query_cny']:.6f} 元（CNY，总计 {cost['total_cny']:.6f}）"
        )
    elif cost["available"]:
        cost_label = "—（未配置 LLM_PRICE_* 单价 / prices not configured）"
    else:
        cost_label = "N/A（/metrics 不可达）"
    qps = payload["qps"]
    qps_label = (
        "—（受控频率模式，未测容量吞吐 / paced, capacity not measured）"
        if qps is None
        else f"{qps:.2f}"
    )
    lines = [
        "# 性能压测报告 / Load Test Report",
        "",
        f"- 时间 / Time: {payload['timestamp']}",
        f"- 并发用户 / Users: {payload['users']}",
        f"- 时长 / Duration: {payload['duration_seconds']:.1f}s",
        f"- 问题池 / Query pool: {payload['config']['queries_count']} 条"
        f"（force_unique={payload['config']['force_unique']}）",
        "",
        "## 核心指标 / Core metrics",
        "",
        "| 指标 / Metric | 结果 / Value |",
        "|------|------|",
        f"| 请求数 / Requests | {payload['requests']} |",
        f"| 失败 / Failures | {payload['failures']} |",
        f"| 错误率 / Error rate | {payload['error_rate']:.4f} |",
        f"| 吞吐 / QPS | {qps_label} |",
        f"| 端到端延迟 p50 / E2E latency p50 | {payload['latency_p50_s']:.3f}s |",
        f"| 端到端延迟 p95 / E2E latency p95 | {payload['latency_p95_s']:.3f}s |",
        f"| 端到端延迟 p99 / E2E latency p99 | {payload['latency_p99_s']:.3f}s |",
        f"| 首 token 延迟 / TTFT（p50/p95/p99） | {ttft_label} |",
        f"| 缓存命中率 / Cache hit rate | {cache['hit_rate']:.2%} |",
        f"| 单请求成本 / Cost per query | {cost_label} |",
        f"| 阈值判定 / Threshold | {status} |",
        "",
    ]

    lines += [
        "## 缓存命中分布 / Cache hit distribution",
        "",
        "| 类型 / Type | 数量 / Count |",
        "|------|------|",
        f"| 精准命中 / Exact hit | {cache['exact']} |",
        f"| 语义命中 / Semantic hit | {cache['semantic']} |",
        f"| 未命中 / Miss | {cache['miss']} |",
        f"| 未知 / Unknown | {cache['unknown']} |",
        "",
    ]

    # TTFT 分场景（缓存命中 / 未命中）——从记录中二次计算
    with _lock:
        records = list(_records)
    ttft_hit = sorted(
        r["ttft"] for r in records if r["ttft"] is not None and r["cache_type"] != "none"
    )
    ttft_miss = sorted(
        r["ttft"] for r in records if r["ttft"] is not None and r["cache_type"] == "none"
    )
    hit_p50 = percentile(ttft_hit, 0.50) if ttft_hit else None
    hit_p95 = percentile(ttft_hit, 0.95) if ttft_hit else None
    miss_p50 = percentile(ttft_miss, 0.50) if ttft_miss else None
    miss_p95 = percentile(ttft_miss, 0.95) if ttft_miss else None
    lines += [
        "## 首 token 延迟分场景 / TTFT by scenario",
        "",
        "| 场景 / Scenario | 样本 / Samples | p50 | p95 |",
        "|------|------|------|------|",
        f"| 缓存命中 / Cache hit | {len(ttft_hit)} | "
        f"{_fmt_s(hit_p50)} | {_fmt_s(hit_p95)} |",
        f"| 未命中 / Miss | {len(ttft_miss)} | "
        f"{_fmt_s(miss_p50)} | {_fmt_s(miss_p95)} |",
        "",
    ]

    lines += [
        "## 分阶段耗时 / Per-stage latency",
        "",
        "| 阶段 / Stage | 样本 / Samples | 平均 / Mean | p95 |",
        "|------|------|------|------|",
    ]
    for row in payload["stage_stats"]:
        lines.append(
            f"| {stage_label(row['stage'])} | {row['count']} | "
            f"{_fmt_ms(row['mean_ms'])} | {_fmt_ms(row['p95_ms'])} |"
        )
    if not payload["stage_stats"]:
        lines.append("| （无节点耗时数据 / no stage data） | — | — | — |")
    lines.append("")

    if cost["available"]:
        lines += [
            "## 成本估算 / Cost estimate（best-effort，基于 /metrics token 增量）",
            "",
            f"- input tokens（输入）：{cost['tokens_input']:.0f}",
            f"- output tokens（输出）：{cost['tokens_output']:.0f}",
            f"- 单价（元/百万 token）：输入 {cost['price_input_per_1m']}，"
            f"输出 {cost['price_output_per_1m']}",
        ]
        if cost["prices_configured"]:
            lines += [
                f"- 总成本 / Total cost：{cost['total_cny']:.6f} 元（CNY）",
                f"- 单请求成本 / Cost per query：{cost['per_query_cny']:.6f} 元（CNY）",
            ]
        else:
            lines.append(
                "- 未配置单价，仅统计 token 用量 / "
                "prices not configured, token usage only"
            )
        lines.append("")

    lines += [
        f"阈值 / Thresholds: p95 ≤ {payload['threshold_p95_max_s']}s，"
        f"error rate ≤ {payload['threshold_error_rate_max']}",
    ]
    if failures_list:
        lines.append("")
        lines.append("未通过项 / Failures:")
        lines += [f"- {f}" for f in failures_list]
    return "\n".join(lines) + "\n"
