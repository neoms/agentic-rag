"""Tool Calling 工具定义 - 联网搜索、计算器等可调用的外部工具"""

import math
import logging
from langchain_core.tools import tool
from ddgs import DDGS
from src.config.settings import settings

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
def calculator(expression: str) -> str:
    """执行数学计算。支持基本算术运算（加减乘除、幂、取余等）。

    Args:
        expression: 数学表达式，如 "2 + 2 * 5"、"sqrt(16)"、"pow(3, 4)"

    Returns:
        计算结果
    """
    try:
        safe_dict = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "log10": math.log10,
            "pi": math.pi, "e": math.e,
        }
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算出错：{str(e)}"


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
        return f"未找到与 '{query}' 相关的搜索结果。"

    lines = [f'联网搜索 "{query}" 结果：']
    for i, r in enumerate(results, 1):
        lines.append(f"\n[{i}] {r['snippet']}")
        if r["url"]:
            lines.append(f"    链接: {r['url']}")
    return "\n".join(lines)


# 工具列表
ALL_TOOLS = [calculator, web_search_tool]
