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
- 阈值断言在测试结束时执行（test_stop）：
    EVAL_LOAD_P95_MAX       默认 10.0（秒）
    EVAL_LOAD_ERROR_RATE_MAX 默认 0.01
  失败时进程退出码非 0
"""

from __future__ import annotations

import os
import random
import threading
import time
import uuid

from locust import HttpUser, between, events, task

_DEFAULT_QUERIES = [
    "Agentic RAG 系统使用什么作为向量数据库和知识图谱存储引擎？",
    "语义检索与 BM25 关键词检索各自的优势是什么？",
    "知识图谱模块中 GraphStore 的作用是什么？",
    "MMR 算法的 lambda_mult 参数取 0.7 意味着什么？",
    "小象科技成立于哪一年，总部位于哪里？",
]

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

FORCE_UNIQUE = os.environ.get("EVAL_LOAD_UNIQUE", "") == "1"

_lock = threading.Lock()
_latencies: list[float] = []
_start_ts = time.perf_counter()


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
        try:
            with self.client.post(
                "/api/v1/chat/stream",
                json={
                    "query": query,
                    "use_cache": True,
                    "enable_web_search": False,
                    "enable_reflection": True,
                    "enable_rerank": True,
                    "enable_grade_documents": True,
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
                for line in resp.iter_lines():
                    if line and line.startswith(b"event: done"):
                        saw_done = True
                        break
                latency = time.perf_counter() - t0
                if not saw_done:
                    raise RuntimeError("SSE 流未收到 done 事件")
                with _lock:
                    _latencies.append(latency)
        except Exception:  # noqa: BLE001 - 任务异常由 Locust 自动记为失败
            raise


@events.test_stop.add_listener
def _on_test_stop(environment, **_kwargs) -> None:  # noqa: ANN001
    """测试结束：计算双语汇总并执行阈值断言"""
    with _lock:
        latencies = sorted(_latencies)
    duration = max(time.perf_counter() - _start_ts, 1e-6)
    total = environment.runner.stats.total
    requests = total.num_requests
    failures = total.num_failures
    error_rate = failures / max(requests, 1)

    p50 = _percentile(latencies, 0.50)
    p95 = _percentile(latencies, 0.95)
    p99 = _percentile(latencies, 0.99)
    qps = len(latencies) / duration

    print("\n" + "=" * 60)
    print("压测结果 / Load Test Summary")
    print(f"  请求数 / Requests      : {requests}")
    print(f"  失败数 / Failures      : {failures}")
    print(f"  错误率 / Error rate    : {error_rate:.4f}")
    print(f"  吞吐 / QPS             : {qps:.2f}")
    print(f"  延迟 p50 / latency     : {p50:.3f}s")
    print(f"  延迟 p95 / latency     : {p95:.3f}s")
    print(f"  延迟 p99 / latency     : {p99:.3f}s")
    print("=" * 60)

    p95_max = float(os.environ.get("EVAL_LOAD_P95_MAX", "10.0"))
    error_max = float(os.environ.get("EVAL_LOAD_ERROR_RATE_MAX", "0.01"))
    failures_list: list[str] = []
    if p95 > p95_max:
        failures_list.append(f"p95 latency（p95 延迟）={p95:.3f}s > {p95_max}s")
    if error_rate > error_max:
        failures_list.append(f"error rate（错误率）={error_rate:.4f} > {error_max}")
    if failures_list:
        print("阈值未通过 / Threshold check FAILED ✘")
        for f in failures_list:
            print(f"  - {f}")
        environment.process_exit_code = 1
    else:
        print("阈值通过 / Threshold check PASSED ✔")


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (idx - lo)
