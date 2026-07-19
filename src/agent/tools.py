"""Tool Calling 工具定义 - 联网搜索、计算器等可调用的外部工具"""

import math
import logging
from langchain_core.tools import tool
from src.config.settings import settings

logger = logging.getLogger(__name__)


@tool
def calculator(expression: str) -> str:
    """执行数学计算。支持基本算术运算（加减乘除、幂、取余等）。

    Args:
        expression: 数学表达式，如 "2 + 2 * 5"、"sqrt(16)"、"pow(3, 4)"

    Returns:
        计算结果
    """
    try:
        # 允许使用的安全函数
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
        搜索结果摘要
    """
    # 注意：生产环境应接入真正的搜索 API（如 SERP API / Bing API）
    # 此处提供一个提示性的占位实现
    logger.info("联网搜索请求: %s", query)
    return (
        f'[联网搜索结果] 关于 "{query}" 的搜索：\n'
        "（提示：生产环境请配置 SERP API Key 或 Bing Search API 以启用真实联网搜索）"
    )


# 工具列表
ALL_TOOLS = [calculator, web_search_tool]
