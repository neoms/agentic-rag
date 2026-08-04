"""分块器：文档标题上下文补全"""

from src.config.settings import settings
from src.pipeline.chunker import (
    TITLE_PREFIX_TEMPLATE,
    _extract_document_title,
    chunk_texts,
    strip_title_prefix,
)


def test_title_prefix_prepended_to_all_chunks():
    texts = [
        "# 小象科技生态系统与产品矩阵\n\n## 1. 公司概况\n\n小象科技成立于2019年。",
        "## 2. 创始团队\n\n李明（CEO，联合创始人）。",
    ]
    docs = chunk_texts(texts, metadata={"filename": "04.md"}, chunk_size=120, chunk_overlap=10)
    assert docs
    assert all(
        d.page_content.startswith("【文档主题：小象科技生态系统与产品矩阵】")
        for d in docs
    )
    # 标题前缀之后仍是原内容（golden 上下文子串不受影响）
    assert any("小象科技成立于2019年" in d.page_content for d in docs)
    assert any("创始团队" in d.page_content for d in docs)


def test_title_prefix_disabled(monkeypatch):
    monkeypatch.setattr(settings, "chunk_title_context", False)
    docs = chunk_texts(
        ["# 标题\n\n内容"], metadata={}, chunk_size=120, chunk_overlap=10,
    )
    assert not docs[0].page_content.startswith("【文档主题")


def test_no_heading_no_prefix():
    docs = chunk_texts(
        ["纯文本内容没有标题"], metadata={}, chunk_size=120, chunk_overlap=10,
    )
    assert not docs[0].page_content.startswith("【文档主题")


def test_extract_document_title_prefers_h1():
    assert _extract_document_title(["### 二级标题\nx", "# 一级标题\n内容"]) == "一级标题"
    assert _extract_document_title(["### 只有二级\nx"]) == "只有二级"
    assert _extract_document_title(["没有标题的文本"]) == ""


def test_strip_title_prefix_uses_same_format_as_generation():
    """剥离逻辑与生成逻辑共用同一前缀格式（单一事实源）"""
    prefixed = TITLE_PREFIX_TEMPLATE.format(title="小象科技生态系统与产品矩阵") + "内容"
    assert prefixed.startswith("【文档主题：小象科技生态系统与产品矩阵】")
    assert strip_title_prefix(prefixed) == "内容"
    assert strip_title_prefix("### 创始团队") == "### 创始团队"  # 无前缀原样返回
