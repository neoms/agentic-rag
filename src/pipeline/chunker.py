"""文本分块器 - 基于 RecursiveCharacterTextSplitter

使用 tiktoken 进行 Token 计数，使 chunk_size 真正代表"最大 Token 数"，
而非字符数。对中英文混合内容的分块精度显著优于 len()。
"""

import logging
import re
from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import settings

logger = logging.getLogger(__name__)

# 标题检测（与 loader.py 保持一致）
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s", re.MULTILINE)

# 分块标题前缀格式（唯一事实源）：chunker 生成与 grade 关键词剥离共用，
# 修改格式只需改这一处
TITLE_PREFIX_TEMPLATE = "【文档主题：{title}】\n"
_TITLE_PREFIX_STRIP_RE = re.compile(r"^【文档主题：.*?】\s*")


def strip_title_prefix(text: str) -> str:
    """剥离块文本开头的文档标题前缀（用于关键词 overlap 计算）

    标题前缀本身对语义检索 / 重排 / 幻觉检测有补全价值，但会虚增
    grade 的关键词 overlap（同文档所有块都带标题词 → 全部保留 → 答案过长），
    因此在关键词打分时剥离。
    """
    return _TITLE_PREFIX_STRIP_RE.sub("", text, count=1)


def _extract_section_title(text: str) -> str:
    """从文本中提取首个标题行作为章节名称"""
    for line in text.split("\n"):
        if _HEADING_PATTERN.match(line):
            return line.strip()
    return ""


def _extract_document_title(texts: list[str]) -> str:
    """提取文档级标题：优先首个 H1，其次任意标题行，最后为空串

    用于"分块上下文补全"：文档标题拼进每个块，避免块的实体/事实
    因缺少文档级主题上下文而无法被检索与判定（如"创始团队"块缺少
    公司名，导致 rerank 低分、幻觉检测无法溯源"小象科技的CEO是李明"）。
    """
    h1 = ""
    any_heading = ""
    for text in texts:
        for line in text.split("\n"):
            m = _HEADING_PATTERN.match(line)
            if not m:
                continue
            title = line[m.end():].strip()
            if not title:
                continue
            if not any_heading:
                any_heading = title
            if m.group(1) == "#" and not h1:
                h1 = title
    return h1 or any_heading


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
    title_prefix = ""
    if settings.chunk_title_context:
        doc_title = _extract_document_title(texts)
        if doc_title:
            title_prefix = TITLE_PREFIX_TEMPLATE.format(title=doc_title)

    all_docs: list[Document] = []
    for text in texts:
        section_title = _extract_section_title(text)
        chunks = chunker.split_text(text)
        for chunk in chunks:
            if not chunk or not chunk.strip():
                continue
            meta = dict(base_metadata)
            if section_title:
                meta["section_title"] = section_title
            content = title_prefix + chunk if title_prefix else chunk
            doc = Document(page_content=content, metadata=meta)
            all_docs.append(doc)

    # 增强元数据：块序号和总数
    total = len(all_docs)
    for i, doc in enumerate(all_docs):
        doc.metadata["chunk_index"] = i
        doc.metadata["total_chunks"] = total

    logger.info("分块完成：%d 段文本 → %d 个文档块", len(texts), total)
    return all_docs
