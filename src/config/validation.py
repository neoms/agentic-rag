"""启动配置校验 - 输出可行动的报错信息

每条问题都包含四要素：当前值、原因、修改位置、修复方式，
确保运维/开发者不用翻源码就能定位并修复配置。
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from src.config.settings import Settings

_HAS_SCHEME = re.compile(r"^https?://", re.IGNORECASE)
_EXT_RE = re.compile(r"^[A-Za-z0-9]+$")

ENV_LOCATION = "项目根目录 .env 文件（或运行环境的同名环境变量；Docker 部署时为 compose 读取的宿主 .env）"


@dataclass
class ConfigIssue:
    """一条配置问题"""

    key: str        # 环境变量名
    value: str      # 当前值
    reason: str     # 为什么错
    location: str   # 在哪里修改
    fix: str        # 怎么改才对

    def format(self) -> str:
        return (
            f"[{self.key}]\n"
            f"  当前值: {self.value}\n"
            f"  原因: {self.reason}\n"
            f"  位置: {self.location}\n"
            f"  修复: {self.fix}"
        )


def _fmt(value) -> str:
    s = str(value)
    return s if len(s) <= 80 else s[:77] + "..."


def _issue(key: str, value, reason: str, fix: str) -> ConfigIssue:
    return ConfigIssue(
        key=key,
        value=_fmt(value),
        reason=reason,
        location=ENV_LOCATION,
        fix=fix,
    )


def _check_int(
    issues: list[ConfigIssue],
    key: str,
    value,
    min_v: int | None,
    max_v: int | None,
    label: str,
    example: str,
) -> None:
    """整数范围检查，错误信息包含正确示例"""
    if isinstance(value, bool):
        issues.append(_issue(
            key, value,
            f"{label} 不能是布尔值",
            f"改为整数，例如 {key}={example}",
        ))
        return
    ok = True
    if min_v is not None and value < min_v:
        ok = False
    if max_v is not None and value > max_v:
        ok = False
    if not ok:
        bound = (
            f"≥ {min_v}" if max_v is None
            else f"{min_v} ~ {max_v}" if min_v is not None
            else f"≤ {max_v}"
        )
        issues.append(_issue(
            key, value,
            f"{label}，取值范围应为 {bound}",
            f"修改为 {key}={example}",
        ))


def _check_float(
    issues: list[ConfigIssue],
    key: str,
    value,
    min_v: float,
    max_v: float,
    label: str,
    example: str,
) -> None:
    if value < min_v or value > max_v:
        issues.append(_issue(
            key, value,
            f"{label}，取值范围应为 {min_v} ~ {max_v}",
            f"修改为 {key}={example}",
        ))


def _check_writable(
    issues: list[ConfigIssue],
    key: str,
    path: Path,
    purpose: str,
) -> None:
    """数据目录可写性检查（目录不存在则创建）"""
    try:
        path.mkdir(parents=True, exist_ok=True)
        if not os.access(path, os.W_OK):
            raise OSError("目录无写权限")
    except OSError as e:
        issues.append(ConfigIssue(
            key=key,
            value=str(path),
            reason=f"{purpose}目录不可写（{e}），服务会在首次写入时失败",
            location="部署环境（容器挂载卷 / 目录权限 / 磁盘空间）",
            fix=(
                "确保该路径存在且运行用户（Docker 容器内为 appuser，uid=10001）"
                f"有写权限；检查挂载：docker compose 中数据卷应挂载到 {path}"
            ),
        ))


def validate_settings(s: Settings) -> list[ConfigIssue]:
    """校验全部值得启动检查的配置项，返回问题列表（空 = 全部通过）"""
    issues: list[ConfigIssue] = []

    # ── 必填项 ──
    if not s.dashscope_api_key.strip():
        issues.append(_issue(
            "DASHSCOPE_API_KEY", "（空）",
            "必须配置阿里云百炼 API Key，否则所有 LLM/Embedding/重排序调用都会失败",
            "在 .env 中填写 DASHSCOPE_API_KEY=sk-xxxxxxxx；Key 可在百炼控制台获取",
        ))

    for key, value, label in [
        ("LLM_MODEL", s.llm_model, "默认 LLM 模型名"),
        ("LLM_MODEL_FAST", s.llm_model_fast, "快速评估模型名"),
        ("LLM_MODEL_STRONG", s.llm_model_strong, "强生成模型名"),
        ("EMBEDDING_MODEL", s.embedding_model, "Embedding 模型名"),
        ("APP_HOST", s.app_host, "服务监听地址"),
        ("LOG_FILE", s.log_file, "日志文件路径"),
    ]:
        if not str(value).strip():
            issues.append(_issue(
                key, "（空）", f"{label}不能为空",
                f"在 .env 中填写 {key}=<正确值>",
            ))

    if s.rerank_enabled and not s.rerank_model.strip():
        issues.append(_issue(
            "RERANK_MODEL", "（空）",
            "RERANK_ENABLED=true 时必须指定重排序模型",
            "填写 RERANK_MODEL=gte-rerank-v2（或百炼支持的其他重排序模型名）",
        ))

    if not _HAS_SCHEME.match(s.llm_base_url):
        issues.append(_issue(
            "LLM_BASE_URL", s.llm_base_url,
            "必须是 http:// 或 https:// 开头的 API 地址",
            "填写 LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1",
        ))

    # ── CORS ──
    origins = s.cors_allowed_origins_list
    if not origins:
        issues.append(_issue(
            "CORS_ALLOWED_ORIGINS", s.cors_allowed_origins,
            "解析后为空（不能全部是空项或纯逗号）",
            "填写至少一个来源，例如 CORS_ALLOWED_ORIGINS=http://localhost:3000",
        ))
    for origin in origins:
        if origin != "*" and not _HAS_SCHEME.match(origin):
            issues.append(_issue(
                "CORS_ALLOWED_ORIGINS", origin,
                "每个来源必须是 http(s):// 开头的地址，或使用通配符 *",
                "修改为 CORS_ALLOWED_ORIGINS=http(s)://你的前端域名",
            ))

    # ── 允许上传的扩展名 ──
    exts = s.allowed_extensions_list
    if not exts:
        issues.append(_issue(
            "ALLOWED_EXTENSIONS", s.allowed_extensions,
            "解析后为空（不能全部是空项）",
            "填写 ALLOWED_EXTENSIONS=pdf,md,txt,docx,csv",
        ))
    for ext in exts:
        if not _EXT_RE.match(ext):
            issues.append(_issue(
                "ALLOWED_EXTENSIONS", ext,
                "扩展名只能包含字母和数字（不带点、不带空格），如 pdf",
                "修改为 ALLOWED_EXTENSIONS=pdf,md,txt,docx,csv",
            ))

    # ── 数值范围（key, 值, 最小, 最大, 说明, 正确示例） ──
    int_rules = [
        ("LLM_MAX_TOKENS", s.llm_max_tokens, 1, None, "LLM 最大输出 token 数", "2048"),
        ("CHUNK_SIZE", s.chunk_size, 1, None, "分块大小", "500"),
        ("CHUNK_OVERLAP", s.chunk_overlap, 0, None, "分块重叠", "100"),
        ("RETRIEVAL_TOP_K", s.retrieval_top_k, 1, None, "检索候选数", "12"),
        ("RERANK_TOP_K", s.rerank_top_k, 1, None, "重排序保留数", "5"),
        ("MEMORY_WINDOW_SIZE", s.memory_window_size, 1, None, "对话记忆窗口轮数", "20"),
        ("MAX_UPLOAD_SIZE_MB", s.max_upload_size_mb, 1, None, "上传大小上限", "10"),
        ("LARGE_FILE_THRESHOLD_MB", s.large_file_threshold_mb, 1, None, "大文件落盘阈值", "5"),
        ("MIN_CONTENT_CHARS", s.min_content_chars, 0, None, "文档最少有效字符数", "20"),
        ("MULTI_QUERY_NUM_VARIATIONS", s.multi_query_num_variations, 1, None, "Multi-Query 变体数", "3"),
        ("KG_MAX_HOPS", s.kg_max_hops, 1, None, "KG 子图最大跳数", "2"),
        ("KG_MAX_ENTITIES", s.kg_max_entities, 1, None, "KG 单次查询最多实体数", "10"),
        ("KG_MAX_CONCURRENCY", s.kg_max_concurrency, 1, None, "KG 构建并发数", "3"),
        ("LLM_MAX_RETRIES", s.llm_max_retries, 0, None, "LLM 最大重试次数", "3"),
        ("LLM_RETRY_MIN_WAIT", s.llm_retry_min_wait, 0, None, "LLM 重试最小等待秒数", "2"),
        ("LLM_RETRY_MAX_WAIT", s.llm_retry_max_wait, 0, None, "LLM 重试最大等待秒数", "30"),
        ("LLM_REQUEST_TIMEOUT", s.llm_request_timeout, 1, None, "LLM 单次请求超时秒数", "120"),
        ("APP_PORT", s.app_port, 1, 65535, "服务监听端口", "8000"),
        ("CACHE_MAX_ENTRIES", s.cache_max_entries, 1, None, "缓存条目上限", "5000"),
        ("CACHE_TTL_SECONDS", s.cache_ttl_seconds, 0, None, "缓存过期秒数", "0"),
        ("CACHE_CITATION_MAX_CHARS", s.cache_citation_max_chars, 1, None, "缓存引文段落最大长度", "500"),
        ("SESSION_MESSAGE_CAP", s.session_message_cap, 1, None, "单会话消息保留上限", "500"),
        ("TASK_HISTORY_KEEP", s.task_history_keep, 1, None, "上传任务保留条数", "100"),
        ("TASK_HISTORY_TTL_DAYS", s.task_history_ttl_days, 0, None, "任务保留天数", "7"),
        ("INDEX_WORKERS", s.index_workers, 1, None, "后台索引并发 worker 数", "2"),
        ("INDEX_QUEUE_MAX", s.index_queue_max, 0, None, "索引排队任务上限", "20"),
    ]
    for key, value, min_v, max_v, label, example in int_rules:
        _check_int(issues, key, value, min_v, max_v, label, example)

    float_rules = [
        ("LLM_TEMPERATURE", s.llm_temperature, 0.0, 2.0, "生成温度", "0.0"),
        ("RETRIEVAL_SIMILARITY_THRESHOLD", s.retrieval_similarity_threshold, 0.0, 1.0, "语义检索过滤阈值", "0.5"),
        ("GRADE_SCORE_IRRELEVANT_MAX", s.grade_score_irrelevant_max, 0.0, 1.0, "文档负判定阈值", "0.25"),
        ("GRADE_SCORE_RELEVANT_MIN", s.grade_score_relevant_min, 0.0, 1.0, "文档正判定阈值", "0.70"),
        ("GRADE_SCORE_RELEVANT_GAP", s.grade_score_relevant_gap, 0.0, 1.0, "文档正判定断层差", "0.10"),
        ("MIN_READABLE_RATIO", s.min_readable_ratio, 0.0, 1.0, "文档可读字符比例", "0.3"),
        ("CACHE_SEMANTIC_THRESHOLD", s.cache_semantic_threshold, 0.0, 1.0, "语义缓存命中阈值", "0.92"),
    ]
    for key, value, min_v, max_v, label, example in float_rules:
        _check_float(issues, key, value, min_v, max_v, label, example)

    # ── 跨字段一致性 ──
    if s.chunk_overlap >= s.chunk_size:
        issues.append(_issue(
            "CHUNK_OVERLAP", s.chunk_overlap,
            "分块重叠必须小于分块大小（CHUNK_OVERLAP < CHUNK_SIZE），否则分块退化为纯重叠",
            "例如 CHUNK_SIZE=500、CHUNK_OVERLAP=100",
        ))

    if s.llm_retry_max_wait < s.llm_retry_min_wait:
        issues.append(_issue(
            "LLM_RETRY_MAX_WAIT", s.llm_retry_max_wait,
            "重试最大等待不能小于最小等待（LLM_RETRY_MAX_WAIT ≥ LLM_RETRY_MIN_WAIT）",
            "例如 LLM_RETRY_MIN_WAIT=2、LLM_RETRY_MAX_WAIT=30",
        ))

    if s.large_file_threshold_mb > s.max_upload_size_mb:
        issues.append(_issue(
            "LARGE_FILE_THRESHOLD_MB", s.large_file_threshold_mb,
            "大文件落盘阈值不能超过上传大小上限（阈值应 ≤ MAX_UPLOAD_SIZE_MB）",
            "例如 MAX_UPLOAD_SIZE_MB=10、LARGE_FILE_THRESHOLD_MB=5",
        ))

    if s.session_message_cap < s.memory_window_size * 2:
        issues.append(_issue(
            "SESSION_MESSAGE_CAP", s.session_message_cap,
            f"单会话消息上限不能小于记忆窗口的两倍（当前窗口 {s.memory_window_size} 轮 = {s.memory_window_size * 2} 条），否则最近 N 轮历史会被截断",
            f"设置 SESSION_MESSAGE_CAP ≥ {s.memory_window_size * 2}",
        ))

    # ── chunk_config_by_type JSON 合法性 ──
    if s.chunk_config_by_type.strip():
        try:
            raw = json.loads(s.chunk_config_by_type)
            for ext, pair in raw.items():
                if (
                    not isinstance(pair, (list, tuple))
                    or len(pair) != 2
                    or not all(isinstance(v, int) for v in pair)
                    or pair[0] < 1
                    or pair[1] < 0
                    or pair[1] >= pair[0]
                ):
                    issues.append(_issue(
                        "CHUNK_CONFIG_BY_TYPE", f"{ext}: {pair}",
                        "每个扩展名的值必须是 [chunk_size, chunk_overlap] 且 0 ≤ overlap < size",
                        '例如 {"pdf": [800, 200]}',
                    ))
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            issues.append(_issue(
                "CHUNK_CONFIG_BY_TYPE", s.chunk_config_by_type,
                f"不是合法的 JSON（{e}），当前会静默回退为默认分块参数",
                '填写形如 {"pdf": [800, 200], "md": [500, 100]} 的 JSON',
            ))

    # ── 数据目录可写性 ──
    _check_writable(issues, "CHROMA_PERSIST_DIR", s.chroma_persist_dir_path, "向量库")
    _check_writable(issues, "KG_DATA_DIR", s.project_root / s.kg_data_dir, "知识图谱")
    _check_writable(issues, "CACHE_DB_PATH", s.cache_db_path_abs.parent, "缓存库")
    _check_writable(issues, "STATE_DB_PATH", s.state_db_path_abs.parent, "运行时状态库")
    _check_writable(issues, "LOG_FILE", (s.project_root / s.log_file).parent, "日志")

    return issues


def format_issues(issues: list[ConfigIssue]) -> str:
    """把问题列表格式化为一份完整、可照做的启动错误说明"""
    lines = [f"启动配置校验失败，共 {len(issues)} 项问题，请逐项修复后重启服务："]
    for i, issue in enumerate(issues, 1):
        lines.append(f"\n{i}. " + issue.format())
    return "\n".join(lines)
