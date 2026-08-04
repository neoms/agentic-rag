"""评估指标注册表 - 中英双语标准指标定义

设计约定：
- id：稳定英文标识，用作 Langfuse score 名 / Prometheus 指标名 / 报告键名
- name_zh / name_en / description_zh / description_en：人类可读双语标签
- 所有报告、控制台输出、门禁摘要均以「英文（中文）」双语展示
- 质量指标全部采用 RAGAS 标准实现，不保留自研 judge
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    """单一指标的定义（双语）"""

    id: str                    # 稳定英文 id（Langfuse/Prometheus/报告键名）
    name_zh: str               # 中文名
    name_en: str               # 英文名
    description_zh: str        # 中文说明
    description_en: str        # 英文说明
    kind: str = "quality"      # quality（质量）| performance（性能）| experience（体验）
    requires_reference: bool = False            # 需要标准答案
    requires_reference_contexts: bool = False   # 需要 golden 上下文（用于检索类指标）

    @property
    def display(self) -> str:
        """双语显示名：faithfulness（忠实度）"""
        return f"{self.name_en}（{self.name_zh}）"


# ============ 质量指标（RAGAS 标准实现） ============

QUALITY_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        id="faithfulness",
        name_zh="忠实度",
        name_en="faithfulness",
        description_zh="答案中的每条事实陈述是否都能在检索上下文中找到依据（反幻觉核心指标）",
        description_en="Whether every factual claim in the answer is grounded in the retrieved context",
    ),
    MetricSpec(
        id="answer_relevancy",
        name_zh="答案相关性",
        name_en="answer_relevancy",
        description_zh="答案是否直接、有效回应用户问题",
        description_en="Whether the answer directly and appropriately addresses the user's question",
    ),
    MetricSpec(
        id="factual_correctness",
        name_zh="事实正确性",
        name_en="factual_correctness",
        description_zh="答案中的事实性陈述与标准答案的一致性（需标准答案）",
        description_en="Whether the factual claims in the answer match the reference answer",
        requires_reference=True,
    ),
    MetricSpec(
        id="context_precision",
        name_zh="上下文精度",
        name_en="context_precision",
        description_zh="检索结果中相关文档是否排在前面、噪音比例（需 golden 上下文）",
        description_en="Whether relevant retrieved items are ranked higher than irrelevant ones",
        requires_reference_contexts=True,
    ),
    MetricSpec(
        id="context_recall",
        name_zh="上下文召回",
        name_en="context_recall",
        description_zh="标准答案所需信息是否都被检索到（漏检检测，需 golden 上下文）",
        description_en="How much of the reference answer is covered by the retrieved context",
        requires_reference_contexts=True,
    ),
)

# ============ 性能指标（Prometheus / 压测） ============

PERFORMANCE_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        id="ttft",
        name_zh="首 token 延迟",
        name_en="TTFT",
        description_zh="从请求发出到首个输出 token 的时间（秒）",
        description_en="Time from request start to the first output token (seconds)",
        kind="performance",
    ),
    MetricSpec(
        id="e2e_latency",
        name_zh="端到端延迟",
        name_en="e2e_latency",
        description_zh="完整请求耗时（秒），含 p50/p95/p99 分位",
        description_en="End-to-end request duration (seconds), including p50/p95/p99",
        kind="performance",
    ),
    MetricSpec(
        id="cost_per_query",
        name_zh="单请求成本",
        name_en="cost_per_query",
        description_zh="单次请求的 LLM token 费用估算（元）",
        description_en="Estimated LLM token cost per request (CNY)",
        kind="performance",
    ),
    MetricSpec(
        id="cache_hit_rate",
        name_zh="缓存命中率",
        name_en="cache_hit_rate",
        description_zh="多级缓存命中比例（精准/语义合计）",
        description_en="Multi-level cache hit ratio (exact + semantic)",
        kind="performance",
    ),
    MetricSpec(
        id="error_rate",
        name_zh="错误率",
        name_en="error_rate",
        description_zh="请求失败（非 200 / 流内错误）占比",
        description_en="Ratio of failed requests (non-200 or stream errors)",
        kind="performance",
    ),
    MetricSpec(
        id="throughput",
        name_zh="吞吐量",
        name_en="throughput",
        description_zh="单位时间成功请求数（QPS）",
        description_en="Successful requests per second (QPS)",
        kind="performance",
    ),
)

# ============ 在线体验指标 ============

EXPERIENCE_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        id="user_feedback",
        name_zh="用户反馈",
        name_en="user_feedback",
        description_zh="用户在对话气泡上的 👍/👎 评分（1-5）",
        description_en="User thumbs up/down rating (1-5) on the chat bubble",
        kind="experience",
    ),
    MetricSpec(
        id="online_scores",
        name_zh="在线采样打分",
        name_en="online_scores",
        description_zh="对生产流量采样进行的 LLM-as-judge 质量评分",
        description_en="LLM-as-judge quality scores sampled from production traffic",
        kind="experience",
    ),
)

ALL_METRICS: tuple[MetricSpec, ...] = QUALITY_METRICS + PERFORMANCE_METRICS + EXPERIENCE_METRICS
METRIC_BY_ID: dict[str, MetricSpec] = {m.id: m for m in ALL_METRICS}


def display_name(metric_id: str) -> str:
    """按 id 取双语显示名；未知 id 原样返回"""
    spec = METRIC_BY_ID.get(metric_id)
    return spec.display if spec else metric_id
