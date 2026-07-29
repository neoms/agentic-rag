"""文档加载器 - 支持 PDF、Markdown、TXT 格式解析"""

import io
import logging
from pathlib import Path
from typing import Optional
from langchain_core.documents import Document
from PyPDF2 import PdfReader
import markdown
import re

logger = logging.getLogger(__name__)

HTML_CLEANER = re.compile(r"<[^>]+>")

# 文件魔数签名映射
MAGIC_SIGNATURES: dict[str, bytes] = {
    ".pdf": b"%PDF-",
    ".docx": b"PK\x03\x04",  # ZIP 格式，DOCX 是 ZIP 容器
}


def detect_file_type(file_bytes: bytes) -> Optional[str]:
    """通过文件头魔数检测真实文件类型

    Args:
        file_bytes: 文件二进制内容（至少前 8 字节）

    Returns:
        检测到的扩展名（含点号），若无法识别则返回 None
    """
    if not file_bytes or len(file_bytes) < 8:
        return None
    for ext, magic in MAGIC_SIGNATURES.items():
        if file_bytes[:len(magic)] == magic:
            return ext
    return None


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

    Note:
        通过魔数检测真实文件类型，若与扩展名不一致仅 warning 不阻断，
        保持以扩展名 LOADER_MAP 为准的解析策略以确保向后兼容。
    """
    suffix = Path(filename).suffix.lower()

    # 魔数检测：交叉校验扩展名与真实文件类型
    detected = detect_file_type(file_bytes)
    if detected and detected != suffix:
        logger.warning(
            "文件扩展名与真实类型不匹配: 扩展名=%s, 魔数检测=%s, 文件名=%s, 将以扩展名解析",
            suffix, detected, filename,
        )
    elif detected:
        logger.debug("魔数检测通过: %s → %s", filename, detected)

    loader = LOADER_MAP.get(suffix)
    if loader is None:
        raise ValueError(f"不支持的文件格式: {suffix}，支持的格式: {list(LOADER_MAP.keys())}")
    return loader(file_bytes, filename)
