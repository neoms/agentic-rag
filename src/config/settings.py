"""应用配置中心 - 基于 Pydantic Settings 管理所有配置项"""

import json
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# 显式加载 .env 到 os.environ，确保 LangSmith 等非 Settings 字段的环境变量也能被第三方库读取
load_dotenv()


class Settings(BaseSettings):
    """全局配置，从 .env 文件和环境变量自动加载"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ========== 百炼 API ==========
    dashscope_api_key: str = ""
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # LLM 模型配置（模型名一律从 .env 读取，代码中不写死具体模型）
    llm_model: str = ""  # 环境变量 LLM_MODEL
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.0

    # 不同场景下的模型选择（环境变量 LLM_MODEL_FAST / LLM_MODEL_STRONG）
    llm_model_fast: str = ""    # 快速场景（检索评估等）
    llm_model_strong: str = ""  # 高质量场景（最终生成）

    # 流式生成（generate_simple/complex）额外请求体参数（JSON）。
    # 思考模型在长 RAG prompt 下可能把 token 预算耗尽导致空答案，
    # 可填 {"enable_thinking": false} 关闭思考，保证最终内容必出。
    generation_extra_body: str = ""  # 环境变量 GENERATION_EXTRA_BODY

    # Embedding 模型（环境变量 EMBEDDING_MODEL）
    embedding_model: str = ""

    # ========== ChromaDB ==========
    chroma_persist_dir: str = "data/chroma"

    # ========== 文档分块 ==========
    chunk_size: int = 500
    chunk_overlap: int = 100
    # 按文件类型差异化分块参数 (JSON: {"扩展名": [chunk_size, chunk_overlap]})
    chunk_config_by_type: str = '{"pdf": [800, 200], "md": [500, 100], "txt": [500, 100], "docx": [500, 100], "csv": [500, 100]}'
    # Tokenizer 编码（用于 token 计数分块，设为空字符串则回退到 len()）
    tokenizer_encoding: str = "cl100k_base"

    @property
    def chunk_params_by_type(self) -> dict[str, tuple[int, int]]:
        """解析 chunk_config_by_type JSON 为 {ext: (size, overlap)} 字典"""
        try:
            raw = json.loads(self.chunk_config_by_type)
            return {k: tuple(v) for k, v in raw.items()}
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}

    # ========== 检索 ==========
    retrieval_top_k: int = 12
    retrieval_similarity_threshold: float = 0.5

    # ========== 重排序 ==========
    rerank_enabled: bool = True
    rerank_model: str = ""  # 百炼文本重排序模型（环境变量 RERANK_MODEL）
    rerank_top_k: int = 5  # 重排序后保留的文档数

    # ========== 文档评估（grade_documents score 预筛） ==========
    # 阈值标定建议：先跑一轮真实数据，观察 rerank_score 分布再微调
    grade_score_irrelevant_max: float = 0.25  # 全体文档最高分 ≤ 此值 → 直接 IRRELEVANT（负判定，0 LLM）
    grade_score_relevant_min: float = 0.70    # top1 分 ≥ 此值 → 候选正判定（还需满足断层）
    grade_score_relevant_gap: float = 0.10    # top1 与 top2 的最小分差（断层检测，规避绝对阈值误判）

    # ========== 对话记忆 ==========
    memory_window_size: int = 20

    # ========== 文件上传 ==========
    max_upload_size_mb: int = 10
    allowed_extensions: str = "pdf,md,txt,docx,csv"
    large_file_threshold_mb: int = 5  # 超过此大小（MB）的文件写入临时文件后处理，减少内存占用

    # ========== 内容校验 ==========
    min_content_chars: int = 20          # 解析后最少有效字符数
    min_readable_ratio: float = 0.3      # 可读字符（字母+中文）最低比例

    # ========== Multi-Query 多角度查询 ==========
    multi_query_num_variations: int = 3  # 生成的查询变体数量

    # ========== 知识图谱 ==========
    kg_data_dir: str = "data/kg"
    kg_max_hops: int = 2           # 子图提取最大跳数
    kg_max_entities: int = 10      # 单次查询最多提取实体数
    kg_max_concurrency: int = 3    # KG 构建 LLM 并发调用数

    # ========== LLM 重试 ==========
    llm_max_retries: int = 3             # 最大重试次数
    llm_retry_min_wait: int = 2          # 重试最小等待秒数 (exponential backoff)
    llm_retry_max_wait: int = 30         # 重试最大等待秒数
    llm_request_timeout: int = 120       # 单次请求超时秒数

    # ========== 服务 ==========
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_file: str = "log/app.log"   # 滚动日志文件（10MB × 5）

    # CORS 允许来源（逗号分隔；生产环境需显式配置前端地址，不能使用 * 配合 credentials）
    cors_allowed_origins: str = "http://localhost:3000"

    # ========== 多级缓存（精准 + 语义） ==========
    cache_enabled: bool = True
    cache_exact_enabled: bool = True
    cache_semantic_enabled: bool = True
    cache_semantic_threshold: float = 0.92   # 语义缓存命中阈值（余弦相似度）
    cache_max_entries: int = 5000            # 缓存条目上限（LRU 淘汰）
    cache_ttl_seconds: int = 0               # 0 = 不过期（仅按 LRU 淘汰）
    cache_db_path: str = "data/cache/cache.db"
    cache_citation_max_chars: int = 500      # 缓存引文段落最大长度（防 DB 膨胀）

    # ========== 运行时状态（会话历史 / 上传任务） ==========
    state_db_path: str = "data/state/state.db"

    # ========== 数据保留策略（防无界增长） ==========
    session_message_cap: int = 500    # 单会话最多保留的消息条数（内存 + 库，超出删除最旧）
    task_history_keep: int = 100      # 上传任务保留条数（超出删除最旧）
    task_history_ttl_days: int = 7    # 已完成/失败任务的保留天数（0 = 仅按条数限制）

    # ========== 评估与可观测性（Langfuse / 评测） ==========
    # Langfuse Cloud 追踪与评估平台（密钥必须成对配置，未配置时全链路优雅降级）
    langfuse_public_key: str = ""     # 环境变量 LANGFUSE_PUBLIC_KEY
    langfuse_secret_key: str = ""     # 环境变量 LANGFUSE_SECRET_KEY
    langfuse_host: str = "https://cloud.langfuse.com"  # 环境变量 LANGFUSE_HOST

    # LLM-as-judge 评判模型（独立强评判模型，避免被测模型自己评自己；
    # 留空则回退 LLM_MODEL_STRONG，报告中会标注"judge 与被测同源"）
    eval_judge_model: str = ""        # 环境变量 EVAL_JUDGE_MODEL
    eval_judge_base_url: str = ""     # 环境变量 EVAL_JUDGE_BASE_URL（可选，默认同 LLM_BASE_URL）
    eval_judge_api_key: str = ""      # 环境变量 EVAL_JUDGE_API_KEY（可选，默认同 DASHSCOPE_API_KEY）
    # judge 额外请求体参数（JSON；如思考模型与 RAGAS n=3 冲突时填 {"enable_thinking": false}）
    eval_judge_extra_body: str = ""   # 环境变量 EVAL_JUDGE_EXTRA_BODY

    # 离线评估发布门禁阈值（JSON：{"metric_id": 阈值}，如 {"faithfulness": 0.85}）
    eval_gate_thresholds: str = "{}"  # 环境变量 EVAL_GATE_THRESHOLDS
    # 在线评估采样比例（0 < rate <= 1，最少 20 条）
    eval_sample_rate: float = 0.1     # 环境变量 EVAL_SAMPLE_RATE
    # 压测/评估 stub 模式（不调用真实 LLM，仅验证流水线机制）
    eval_stub_llm: bool = False       # 环境变量 EVAL_STUB_LLM

    # LLM 成本估算单价（元/百万 token；0 表示不估算成本指标）
    llm_price_input_per_1m: float = 0.0    # 环境变量 LLM_PRICE_INPUT_PER_1M
    llm_price_output_per_1m: float = 0.0   # 环境变量 LLM_PRICE_OUTPUT_PER_1M

    # ========== 文档索引后台队列 ==========
    index_workers: int = 2            # 后台索引并发 worker 数
    index_queue_max: int = 20         # 排队任务上限（超出拒绝上传，防成本/资源滥用）

    @property
    def eval_gate_thresholds_dict(self) -> dict[str, float]:
        """解析门禁阈值 JSON 为 {metric_id: threshold}"""
        try:
            raw = json.loads(self.eval_gate_thresholds)
            return {str(k): float(v) for k, v in raw.items()}
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}

    @property
    def eval_judge_extra_body_dict(self) -> dict | None:
        """解析 judge 额外请求体 JSON；非法或空返回 None"""
        raw = self.eval_judge_extra_body.strip()
        if not raw:
            return None
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None

    @property
    def generation_extra_body_dict(self) -> dict | None:
        """解析生成节点额外请求体 JSON；非法或空返回 None"""
        raw = self.generation_extra_body.strip()
        if not raw:
            return None
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None

    @property
    def allowed_extensions_list(self) -> list[str]:
        return [ext.strip().lower() for ext in self.allowed_extensions.split(",")]

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    @property
    def chroma_persist_dir_path(self) -> Path:
        p = Path(self.chroma_persist_dir)
        if not p.is_absolute():
            p = self.project_root / p
        return p

    @property
    def cache_db_path_abs(self) -> Path:
        p = Path(self.cache_db_path)
        if not p.is_absolute():
            p = self.project_root / p
        return p

    @property
    def state_db_path_abs(self) -> Path:
        p = Path(self.state_db_path)
        if not p.is_absolute():
            p = self.project_root / p
        return p

    @property
    def cors_allowed_origins_list(self) -> list[str]:
        """解析 CORS 允许来源为列表（过滤空项）"""
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]


# 全局单例
settings = Settings()
