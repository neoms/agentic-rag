"""文本分块器 - 基于 RecursiveCharacterTextSplitter"""

import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config.settings import settings

logger = logging.getLogger(__name__)


def create_chunker(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """创建文本分块器

    Args:
        chunk_size: 块大小，默认 500
        chunk_overlap: 重叠大小，默认 100

    Returns:
        RecursiveCharacterTextSplitter 实例
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", ".", " ", ""],
        is_separator_regex=False,
    )


def chunk_texts(
    texts: list[str],
    metadata: dict | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """将文本列表分块为 Document 列表

    Args:
        texts: 文本片段列表
        metadata: 每个文档块的元数据
        chunk_size: 块大小
        chunk_overlap: 重叠大小

    Returns:
        Document 对象列表
    """
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

    logger.info("分块完成：%d 段文本 → %d 个文档块", len(texts), len(all_docs))
    return all_docs
