"""文档加载器 - 支持 PDF、Markdown、TXT、DOCX、CSV 格式解析"""

import csv
import io
import logging
import re
from pathlib import Path
from typing import Optional
from langchain_core.documents import Document
from docx import Document as DocxDocument
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LAParams

from src.config.settings import settings

logger = logging.getLogger(__name__)

# Markdown 内联标记清理
_INLINE_FORMAT = re.compile(r"\*\*\*(.+?)\*\*\*|\*\*(.+?)\*\*|\*(.+?)\*|~~(.+?)~~")
_INLINE_CODE = re.compile(r"(?<!`)`(?!``)([^`]+)`")  # 仅匹配单反引号，避免匹配 ``` 代码 fence
_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_HR_PATTERN = re.compile(r"^-{3,}\s*$|^\*{3,}\s*$", re.MULTILINE)

# Unicode 范围：CJK 统一表意文字
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")

# Markdown 标题检测（行首 # 标记）
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s", re.MULTILINE)


def _is_readable_char(ch: str) -> bool:
    """判断字符是否属于可读字符（拉丁字母或 CJK 汉字）"""
    return (ch.isascii() and ch.isalpha()) or bool(_CJK_PATTERN.match(ch))

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
    """使用 pdfminer.six 解析 PDF 文件，按页返回文本列表

    通过布局分析保留页面内文本的阅读顺序，特别适合含有多栏、
    CJK 文字或复杂排版的 PDF。
    """
    laparams = LAParams(
        detect_vertical=True,       # 检测竖排文字（CJK 需要）
        all_texts=True,             # 提取所有文本层
        line_margin=0.5,            # 行间距阈值
        char_margin=2.0,            # 字符间距阈值
        word_margin=0.1,            # 词间距阈值
    )
    pages: list[str] = []
    try:
        for page_layout in extract_pages(io.BytesIO(file_bytes), laparams=laparams):
            texts: list[str] = []
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    text = element.get_text().strip()
                    if text:
                        texts.append(text)
            page_text = "\n".join(texts).strip()
            if page_text:
                pages.append(page_text)
    except Exception as e:
        logger.warning("pdfminer 解析失败 %s: %s，返回空列表", filename, e)
        return []

    logger.info("PDF %s 解析出 %d 页文本", filename, len(pages))
    return pages


def _refine_md_text(raw: str) -> str:
    """清理 Markdown 内联标记，保留结构标记

    保留标题、列表、代码块、引用等结构标记（#、-、```、>），
    仅去除内联格式（粗体/斜体/删除线/行内代码）并转换链接和图片。
    """
    # 去除行内代码标记
    text = _INLINE_CODE.sub(r"\1", raw)
    # 去除粗体/斜体/删除线（保留内容）
    text = _INLINE_FORMAT.sub(
        lambda m: next(g for g in m.groups() if g is not None), text
    )
    # 图片 → [image: alt]
    text = _IMAGE_PATTERN.sub(r"[image: \1]", text)
    # 链接 → text (url)
    text = _LINK_PATTERN.sub(r"\1 (\2)", text)
    return text


def _split_by_headings(text: str) -> list[str]:
    """按标题边界分割 Markdown 文本，确保每个章节连续

    标题行（# / ## / ### 等）作为章节边界，正文跟随最近的标题。
    每个返回的段落都是一个完整章节（标题 + 内容），不会被跨节断开。

    Returns:
        章节文本列表，按文档顺序排列
    """
    lines = text.split("\n")
    sections: list[str] = []
    current: list[str] = []

    for line in lines:
        if _HEADING_PATTERN.match(line):
            # 新的标题行 → 结束上一节
            if current:
                sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("\n".join(current).strip())

    return [s for s in sections if s]


def load_markdown(file_bytes: bytes, filename: str) -> list[str]:
    """解析 Markdown 文件，保留结构信息

    按标题边界分割为章节（语义分块），确保 chunk 不会跨越章节。
    同时保留标题、列表、代码块、引用等结构标记。
    """
    md_text = file_bytes.decode("utf-8", errors="replace")
    # 先清理内联标记
    plain = _refine_md_text(md_text)
    # 将水平线作为段落分隔符
    plain = _HR_PATTERN.sub("\n\n---\n\n", plain)

    # 按标题边界分割（语义分块，标题保留在章节开头）
    sections = _split_by_headings(plain)
    if not sections:
        sections = [plain]

    logger.info("Markdown %s 解析出 %d 个章节（按标题边界）", filename, len(sections))
    return sections


def load_txt(file_bytes: bytes, filename: str) -> list[str]:
    """解析纯文本文件，按段落分割"""
    text = file_bytes.decode("utf-8", errors="replace")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    logger.info("TXT %s 解析出 %d 段落", filename, len(paragraphs))
    return paragraphs


def load_docx(file_bytes: bytes, filename: str) -> list[str]:
    """解析 DOCX 文件，按段落返回文本列表"""
    try:
        doc = DocxDocument(io.BytesIO(file_bytes))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
        if not paragraphs:
            logger.warning("DOCX %s 未解析到段落文本", filename)
        logger.info("DOCX %s 解析出 %d 段落", filename, len(paragraphs))
        return paragraphs
    except Exception as e:
        logger.warning("DOCX 解析失败 %s: %s，返回空列表", filename, e)
        return []


def load_csv(file_bytes: bytes, filename: str) -> list[str]:
    """解析 CSV 文件，将每行转为文本，跳过空行

    首行作为表头，后续每行按列拼接为 "列名: 值" 格式。
    """
    try:
        text = file_bytes.decode("utf-8", errors="replace")
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            logger.warning("CSV %s 为空文件", filename)
            return []

        header = rows[0]
        lines: list[str] = []

        # 表头单独作为一行
        header_text = " | ".join(cell.strip() for cell in header if cell.strip())
        if header_text:
            lines.append(f"[表头] {header_text}")

        # 数据行：每行转 "列名: 值" 格式
        for row in rows[1:]:
            cols = [cell.strip() for cell in row]
            if not any(cols):
                continue
            parts = []
            for i, val in enumerate(cols):
                col_name = header[i].strip() if i < len(header) else f"列{i}"
                if val:
                    parts.append(f"{col_name}: {val}")
            if parts:
                lines.append(" | ".join(parts))

        logger.info("CSV %s 解析出 %d 行", filename, len(lines))
        return lines
    except Exception as e:
        logger.warning("CSV 解析失败 %s: %s，返回空列表", filename, e)
        return []


LOADER_MAP = {
    ".pdf": load_pdf,
    ".md": load_markdown,
    ".txt": load_txt,
    ".docx": load_docx,
    ".csv": load_csv,
}


def validate_content(texts: list[str], filename: str) -> None:
    """校验解析结果的内容完整性

    Args:
        texts: 解析后的文本片段列表
        filename: 文件名（用于日志）

    Raises:
        ValueError: 内容为空或有效字符数不足最低阈值
    """
    if not texts:
        raise ValueError(f"文件 {filename} 解析后内容为空")

    total_chars = sum(len(t) for t in texts)
    if total_chars < settings.min_content_chars:
        raise ValueError(
            f"文件 {filename} 解析后有效字符数 {total_chars} 不足最低阈值 {settings.min_content_chars}"
        )

    # 计算可读字符比例
    all_text = "".join(texts)
    readable_count = sum(1 for ch in all_text if _is_readable_char(ch))
    readable_ratio = readable_count / total_chars if total_chars > 0 else 0.0

    if readable_ratio < settings.min_readable_ratio:
        logger.warning(
            "文件 %s 可读字符比例 %.1f%% 低于阈值 %.0f%%，内容可能异常",
            filename, readable_ratio * 100, settings.min_readable_ratio * 100,
        )
    else:
        logger.debug("内容完整性校验通过: %s (chars=%d, readable=%.1f%%)",
                     filename, total_chars, readable_ratio * 100)


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
    texts = loader(file_bytes, filename)

    # 内容完整性预检
    validate_content(texts, filename)

    return texts
