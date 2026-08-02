"""Tool Calling 工具定义 - 联网搜索等可调用的外部工具"""

import logging
from langchain_core.tools import tool
from ddgs import DDGS

logger = logging.getLogger(__name__)


def _duckduckgo_search(query: str, max_results: int = 5) -> list[dict]:
    """通过 DuckDuckGo 执行网页搜索

    Args:
        query: 搜索关键词
        max_results: 最大结果数

    Returns:
        [{"title": ..., "url": ..., "snippet": ...}, ...]
    """
    results: list[dict] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
    except Exception as e:
        logger.warning("DuckDuckGo 搜索失败: %s", e)

    logger.info("DuckDuckGo 搜索 '%s' → %d 条结果", query, len(results))
    return results


@tool
def web_search_tool(query: str) -> str:
    """联网搜索获取最新信息。当本地知识库无法回答问题时使用。

    Args:
        query: 搜索关键词

    Returns:
        搜索结果摘要（含来源链接）
    """
    logger.info("联网搜索请求: %s", query)
    results = _duckduckgo_search(query, max_results=5)
    if not results:
        logger.warning("联网搜索无结果: %s", query)
        return f"未找到与 '{query}' 相关的搜索结果。"
    logger.info("联网搜索成功: %d 条结果", len(results))
    lines = [f'联网搜索 "{query}" 结果：']
    for i, r in enumerate(results, 1):
        lines.append(f"\n[{i}] {r['snippet']}")
        if r["url"]:
            lines.append(f"    链接: {r['url']}")
    return "\n".join(lines)


# 工具列表
ALL_TOOLS = [web_search_tool]
