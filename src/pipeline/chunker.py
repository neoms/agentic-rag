"""文本分块器 - 基于 RecursiveCharacterTextSplitter

使用 tiktoken 进行 Token 计数，使 chunk_size 真正代表"最大 Token 数"，
而非字符数。对中英文混合内容的分块精度显著优于 len()。
"""

import logging
from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_token_length_function() -> callable:
    """获取 token 计数函数

    优先使用 tiktoken 编码，若编码名称为空或加载失败则回退到 len()。
    结果缓存在模块级，避免重复加载 tokenizer。
    """
    encoding_name = settings.tokenizer_encoding
    if not encoding_name:
        logger.info("tokenizer_encoding 为空，回退到字符计数 len()")
        return len

    try:
        import tiktoken
        encoding = tiktoken.get_encoding(encoding_name)
        logger.info("使用 tiktoken 编码: %s", encoding_name)
        return lambda text: len(encoding.encode(text))
    except Exception as e:
        logger.warning("加载 tiktoken 编码 %s 失败: %s，回退到 len()", encoding_name, e)
        return len


def create_chunker(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """创建文本分块器

    Args:
        chunk_size: 块大小（Token 数），默认 500
        chunk_overlap: 重叠大小（Token 数），默认 100

    Returns:
        RecursiveCharacterTextSplitter 实例
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        length_function=_get_token_length_function(),
        separators=["\n\n", "\n", "。", ".", " ", ""],
        is_separator_regex=False,
    )


def chunk_texts(
    texts: list[str],
    metadata: dict | None = None,
    file_type: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """将文本列表分块为 Document 列表

    支持按文件类型差异化分块参数。若指定了 file_type 且未显式传入
    chunk_size/chunk_overlap，则自动从 settings.chunk_params_by_type 中
    查找对应参数；未匹配时回退到 settings.chunk_size / chunk_overlap。

    Args:
        texts: 文本片段列表
        metadata: 每个文档块的元数据
        file_type: 文件扩展名（含点号，如 ".pdf"），用于选择分块参数
        chunk_size: 显式指定块大小（优先级最高）
        chunk_overlap: 显式指定重叠大小（优先级最高）

    Returns:
        Document 对象列表
    """
    if file_type and chunk_size is None and chunk_overlap is None:
        params = settings.chunk_params_by_type.get(file_type)
        if params:
            chunk_size, chunk_overlap = params

    chunker = create_chunker(chunk_size, chunk_overlap)
    base_metadata = metadata or {}

    all_docs: list[Document] = []
    for text in texts:
        chunks = chunker.split_text(text)
        for chunk in chunks:
            if not chunk or not chunk.strip():
                continue
            doc = Document(page_content=chunk, metadata=dict(base_metadata))
            all_docs.append(doc)

    # 增强元数据：块序号和总数
    total = len(all_docs)
    for i, doc in enumerate(all_docs):
        doc.metadata["chunk_index"] = i
        doc.metadata["total_chunks"] = total

    logger.info("分块完成：%d 段文本 → %d 个文档块", len(texts), total)
    return all_docs
