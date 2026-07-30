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

    # LLM 模型配置
    llm_model: str = "qwen-plus"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.0

    # 不同场景下的模型选择
    llm_model_fast: str = "qwen-turbo"    # 快速场景（检索评估等）
    llm_model_strong: str = "qwen-max"    # 高质量场景（最终生成）

    # Embedding 模型
    embedding_model: str = "text-embedding-v4"

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
    rerank_model: str = "gte-rerank"  # 百炼文本重排序模型
    rerank_top_k: int = 5  # 重排序后保留的文档数

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


# 全局单例
settings = Settings()
