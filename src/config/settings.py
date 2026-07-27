"""应用配置中心 - 基于 Pydantic Settings 管理所有配置项"""

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
    chroma_persist_dir: str = "chroma_data"

    # ========== 文档分块 ==========
    chunk_size: int = 500
    chunk_overlap: int = 100

    # ========== 检索 ==========
    retrieval_top_k: int = 20
    retrieval_similarity_threshold: float = 0.5

    # ========== 重排序 ==========
    rerank_enabled: bool = True
    rerank_model: str = "gte-rerank"  # 百炼文本重排序模型
    rerank_top_k: int = 5  # 重排序后保留的文档数

    # ========== 对话记忆 ==========
    memory_window_size: int = 20

    # ========== 文件上传 ==========
    max_upload_size_mb: int = 10
    allowed_extensions: str = "pdf,md,txt"

    # ========== HyDE 假设文档嵌入 ==========
    hyde_model: str = "qwen-turbo"  # 用于生成假设答案的模型

    # ========== Multi-Query 多角度查询 ==========
    multi_query_num_variations: int = 3  # 生成的查询变体数量

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
