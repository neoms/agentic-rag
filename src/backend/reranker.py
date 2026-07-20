"""重排序模块 - 使用百炼 DashScope text-rerank API 对检索结果做精排

在混合检索（语义 + MMR）合并去重后，通过专有重排序模型对文档
列表做二次精排，提升最终传给 LLM 的文档质量。
"""

import logging

from langchain_core.documents import Document

from src.config.settings import settings

logger = logging.getLogger(__name__)


def rerank_documents(
    query: str,
    documents: list[Document],
    top_k: int | None = None,
) -> list[Document]:
    """使用百炼文本重排序模型对检索结果做精排

    Args:
        query: 用户查询
        documents: 待重排序的文档列表
        top_k: 返回的文档数量，默认使用 settings.rerank_top_k

    Returns:
        重排序后的文档列表（按相关性得分从高到低排列，截断到 top_k）
    """
    if not documents:
        return documents

    top_k = top_k or settings.rerank_top_k

    if len(documents) <= top_k:
        logger.info("文档数 %d ≤ top_k=%d，跳过重排序", len(documents), top_k)
        return documents

    logger.info("开始重排序: query='%s', 文档数=%d, top_k=%d", query[:80], len(documents), top_k)

    try:
        # 使用百炼 DashScope 的 TextReRank API
        from dashscope import TextReRank

        doc_texts = [doc.page_content for doc in documents]

        response = TextReRank.call(
            model=settings.rerank_model,
            query=query,
            documents=doc_texts,
            top_n=min(top_k, len(doc_texts)),
            return_documents=False,
        )

        if response.status_code != 200 or not response.output:
            logger.warning(
                "重排序 API 返回异常: status=%s, code=%s, message=%s",
                response.status_code,
                getattr(response, "code", "N/A"),
                getattr(response, "message", "N/A"),
            )
            return documents[:top_k]

        # API 已按 relevance_score 降序排列
        results = response.output.results
        reranked = [
            documents[r.index]
            for r in results
            if r.index < len(documents)
        ]

        # 附加 relevance_score 到 metadata
        for i, doc in enumerate(reranked):
            doc.metadata["rerank_score"] = round(results[i].relevance_score, 4)

        top_score = results[0].relevance_score if results else 0
        bottom_score = results[-1].relevance_score if results else 0
        logger.info(
            "重排序完成: %d → %d, relevance_score 范围 [%.4f, %.4f]",
            len(documents), len(reranked),
            bottom_score, top_score,
        )

        return reranked

    except ImportError:
        logger.warning("dashscope 未安装或版本不支持 TextReRank，跳过重排序")
        return documents[:top_k]
    except Exception as e:
        logger.error("重排序失败: %s，降级为原始排序取前 %d 个", e, top_k)
        return documents[:top_k]
