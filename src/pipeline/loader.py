"""文档加载器 - 支持 PDF、Markdown、TXT 格式解析"""

import io
import logging
from pathlib import Path
from langchain_core.documents import Document
from PyPDF2 import PdfReader
import markdown
import re

logger = logging.getLogger(__name__)

HTML_CLEANER = re.compile(r"<[^>]+>")


def load_pdf(file_bytes: bytes, filename: str) -> list[str]:
    """解析 PDF 文件，按页返回文本列表"""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append(text.strip())
    logger.info("PDF %s 解析出 %d 页文本", filename, len(pages))
    return pages


def load_markdown(file_bytes: bytes, filename: str) -> list[str]:
    """解析 Markdown 文件，返回纯文本"""
    md_text = file_bytes.decode("utf-8", errors="replace")
    html = markdown.markdown(md_text)
    plain_text = HTML_CLEANER.sub("", html)
    # 按双换行分割段落
    paragraphs = [p.strip() for p in plain_text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [plain_text]
    logger.info("Markdown %s 解析出 %d 段落", filename, len(paragraphs))
    return paragraphs


def load_txt(file_bytes: bytes, filename: str) -> list[str]:
    """解析纯文本文件，按段落分割"""
    text = file_bytes.decode("utf-8", errors="replace")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    logger.info("TXT %s 解析出 %d 段落", filename, len(paragraphs))
    return paragraphs


LOADER_MAP = {
    ".pdf": load_pdf,
    ".md": load_markdown,
    ".txt": load_txt,
}


def load_document(file_bytes: bytes, filename: str) -> list[str]:
    """根据文件扩展名自动选择解析器

    Args:
        file_bytes: 文件二进制内容
        filename: 文件名（用于判断类型）

    Returns:
        文本片段列表
    """
    suffix = Path(filename).suffix.lower()
    loader = LOADER_MAP.get(suffix)
    if loader is None:
        raise ValueError(f"不支持的文件格式: {suffix}，支持的格式: {list(LOADER_MAP.keys())}")
    return loader(file_bytes, filename)
