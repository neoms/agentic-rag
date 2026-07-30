"""LangSmith RAG 评估流水线

使用方法:
    uv run python eval/run_eval.py --version v1
    uv run python eval/run_eval.py --version v2 --enable-kg
    uv run python eval/run_eval.py --version v2 --enable-kg --enable-multi-query

目录结构:
    eval/
    ├── v1/                          # 版本目录
    │   ├── sample_docs/             # 测试文档
    │   ├── dataset.jsonl            # 测试数据集（Q&A）
    │   └── results/                 # 评估结果（自动下载保存）
    ├── v2/                          # v2 版本（含知识图谱评估）
    │   ├── sample_docs/
    │   ├── dataset.jsonl
    │   └── results/
    └── run_eval.py                  # 本脚本

评估指标（8 个）：
  correctness / faithfulness / answer_relevance / completeness
  context_precision / retrieval_relevance / answer_length / latency
"""

import sys
import json
import time
import asyncio
import logging
import re
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run

from src.services.rag_service import rag_service
from src.models.chat import AgenticChatRequest, SourceDocument
from src.services.document_service import document_service
from src.store.vector_store import vector_store
from src.backend.llm import create_fast_llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# LangSmith 数据集前缀
DATASET_PREFIX = "agentic-rag-eval"

# ============================================================
# 数据集加载
# ============================================================


def load_dataset(version: str) -> list[dict]:
    """从 eval/{version}/dataset.jsonl 加载测试数据集"""
    dataset_path = Path(__file__).resolve().parent / version / "dataset.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")

    examples = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if not examples:
        raise ValueError(f"数据集为空: {dataset_path}")

    logger.info("从 %s 加载了 %d 条测试数据", dataset_path, len(examples))
    return examples


# ============================================================
# 步骤 1：索引测试文档
# ============================================================


def index_sample_docs(version: str):
    """将 eval/{version}/sample_docs/ 下未索引的文档入库"""
    docs_dir = Path(__file__).resolve().parent / version / "sample_docs"
    md_files = sorted(docs_dir.glob("*.md"))

    if not md_files:
        logger.warning("%s/ 目录下没有 .md 文件", docs_dir)
        return

    existing = vector_store.list_documents()
    existing_names = {d["filename"] for d in existing}

    new_files = [f for f in md_files if f.name not in existing_names]
    if not new_files:
        logger.info("所有测试文档已索引（%d 个）", len(existing))
        return

    logger.info("索引 %d 个新文档...", len(new_files))
    for fp in new_files:
        content = fp.read_text(encoding="utf-8")
        result = document_service.upload_document(
            file_bytes=content.encode("utf-8"),
            filename=fp.name,
        )
        logger.info("  %s → %d chunks (doc_id=%s)", fp.name, result.chunk_count, result.doc_id)


# ============================================================
# 步骤 2：创建 LangSmith 数据集
# ============================================================


def create_langsmith_dataset(version: str, examples: list[dict]) -> str:
    """在 LangSmith 中创建评估数据集（幂等：已存在则跳过）"""
    client = Client()
    dataset_name = f"{DATASET_PREFIX}-{version}"

    existing = list(client.list_datasets(dataset_name=dataset_name))
    if existing:
        ds = existing[0]
        logger.info("LangSmith 数据集 '%s' 已存在（%d 条）", dataset_name, ds.example_count)
        return dataset_name

    ds = client.create_dataset(
        dataset_name=dataset_name,
        description=f"Agentic RAG 评估数据集 — {version} 版本，{len(examples)} 条 Q&A",
    )
    for ex in examples:
        client.create_example(
            inputs={"question": ex["question"]},
            outputs={"answer": ex["answer"]},
            dataset_id=ds.id,
        )
    logger.info("LangSmith 数据集 '%s' 创建完成，共 %d 条", dataset_name, len(examples))
    return dataset_name


# ============================================================
# 步骤 3：目标函数
# ============================================================


# 可通过命令行参数覆盖的全局评估配置
_eval_config: dict = {}


async def _consume_stream(request: AgenticChatRequest) -> dict:
    """消费流式事件，聚合为结构化评估结果"""
    answer = ""
    sources: list[SourceDocument] = []
    agent_path: list[str] = []
    reflection_count = 0

    async for event in rag_service.agentic_rag_stream(request):
        if event.event == "token":
            answer += event.data
        elif event.event == "source":
            try:
                raw = json.loads(event.data)
                sources = [SourceDocument(**s) for s in raw]
            except Exception:
                pass
        elif event.event == "path":
            try:
                agent_path = json.loads(event.data)
            except Exception:
                pass
        elif event.event == "hallucination":
            try:
                hr = json.loads(event.data)
                if not hr.get("passed", True):
                    reflection_count += 1
            except Exception:
                pass

    return {
        "answer": answer,
        "sources": sources,
        "agent_path": agent_path,
        "reflection_count": reflection_count,
    }


def target_fn(inputs: dict) -> dict:
    """评估目标函数 — 调用 Agentic RAG 流式接口，记录延迟"""
    question = inputs["question"]
    logger.info("RAG 请求: %s", question[:60])

    request = AgenticChatRequest(
        query=question,
        session_id=f"eval-{abs(hash(question)) % 10000}",
        enable_web_search=False,
        enable_reflection=True,
        enable_kg=_eval_config.get("enable_kg", False),
        enable_multi_query=_eval_config.get("enable_multi_query", False),
        enable_bm25=_eval_config.get("enable_bm25", True),

    t_start = time.perf_counter()
    result = asyncio.run(_consume_stream(request))
    elapsed = round(time.perf_counter() - t_start, 3)

    sources = [
        {
            "content": s.content,
            "filename": s.metadata.get("filename", "unknown"),
            "rerank_score": s.metadata.get("rerank_score"),
        }
        for s in result["sources"]
    ]

    return {
        "answer": result["answer"],
        "sources": json.dumps(sources, ensure_ascii=False),
        "agent_path": json.dumps(result["agent_path"], ensure_ascii=False),
        "reflection_count": result["reflection_count"],
        "latency_seconds": elapsed,
    }


# ============================================================
# 评估器（8 个指标）
# ============================================================


def _llm_judge(prompt: str, scale: int = 5) -> float:
    llm = create_fast_llm()
    resp = llm.invoke(prompt)
    text = resp.content.strip()
    try:
        nums = re.findall(r"\d+\.?\d*", text)
        if nums:
            score = float(nums[0])
            return min(max(score / scale, 0.0), 1.0)
    except (ValueError, IndexError):
        pass
    return 0.5


def _parse_sources(sources_str: str) -> list[dict]:
    try:
        return json.loads(sources_str)
    except (json.JSONDecodeError, TypeError):
        return []


# 1. 答案正确性
def evaluate_answer_correctness(run: Run, example: Example) -> dict:
    prediction = run.outputs.get("answer", "")
    reference = example.outputs.get("answer", "")
    question = example.inputs.get("question", "")
    if not prediction:
        return {"key": "correctness", "score": 0.0, "comment": "空答案"}

    prompt = f"""你是一位严格的评估专家。请对比"标准答案"和"系统生成答案"，综合评分（1-5）：
- 准确性：是否准确包含了标准答案中的关键事实和概念？
- 完整性：是否覆盖了标准答案的核心要点？

只输出一个数字（1-5）。

问题：{question}
标准答案：{reference}
系统答案：{prediction}
评分（1-5）:"""
    score = _llm_judge(prompt)
    return {"key": "correctness", "score": score, "comment": f"正确性: {score:.2f}"}


# 2. 忠实度/反幻觉
def evaluate_faithfulness(run: Run, example: Example) -> dict:
    answer = run.outputs.get("answer", "")
    sources = _parse_sources(run.outputs.get("sources", "[]"))
    if not answer:
        return {"key": "faithfulness", "score": 0.0, "comment": "空答案"}
    if not sources:
        return {"key": "faithfulness", "score": 0.5, "comment": "无检索文档"}

    docs_text = "\n---\n".join(
        f"[文档{i+1}] {s['content'][:500]}" for i, s in enumerate(sources[:5])
    )
    prompt = f"""你是一位严格的 RAG 忠实度评估专家。请判断"系统答案"中的每一条事实性陈述是否能从"检索文档"中找到依据。

评分标准（1-5）：
5 = 所有事实性陈述都能在文档中找到明确依据，完全忠实
4 = 绝大部分事实有文档依据，仅极少量合理推断
3 = 主要事实有依据，但存在一些文档中未明确说明的补充信息
2 = 存在明显与文档不符或文档中未提及的重要声明
1 = 大量编造、凭空捏造，与文档严重不符

只输出一个数字（1-5）。

检索文档：
{docs_text}

系统答案：
{answer}

忠实度评分（1-5）:"""
    score = _llm_judge(prompt)
    return {"key": "faithfulness", "score": score, "comment": f"忠实度: {score:.2f}"}


# 3. 答案相关性
def evaluate_answer_relevance(run: Run, example: Example) -> dict:
    answer = run.outputs.get("answer", "")
    question = example.inputs.get("question", "")
    if not answer:
        return {"key": "answer_relevance", "score": 0.0, "comment": "空答案"}

    prompt = f"""你是一位答案质量评估专家。请评估"系统答案"是否直接、有效地回答了"用户问题"。

评分标准（1-5）：
5 = 精准直接回答，没有任何偏离
4 = 基本直接回答，仅少量不相关内容
3 = 部分相关，但有不少偏离或泛泛而谈
2 = 回答与问题关系较弱，大量内容与问题无关
1 = 答非所问，完全偏离主题

只输出一个数字（1-5）。

用户问题：{question}
系统答案：{answer}
相关性评分（1-5）:"""
    score = _llm_judge(prompt)
    return {"key": "answer_relevance", "score": score, "comment": f"答案相关性: {score:.2f}"}


# 4. 完整性
def evaluate_completeness(run: Run, example: Example) -> dict:
    answer = run.outputs.get("answer", "")
    question = example.inputs.get("question", "")
    if not answer:
        return {"key": "completeness", "score": 0.0, "comment": "空答案"}

    prompt = f"""你是一位答案完整性评估专家。请评估"系统答案"是否完整覆盖了"用户问题"的所有方面。

评分标准（1-5）：
5 = 完整覆盖问题所有方面，没有遗漏
4 = 覆盖了主要方面，仅缺少次要细节
3 = 覆盖部分方面，有明显的遗漏
2 = 只回答了问题的很小一部分
1 = 几乎没有回答问题的核心要点

只输出一个数字（1-5）。

用户问题：{question}
系统答案：{answer}
完整性评分（1-5）:"""
    score = _llm_judge(prompt)
    return {"key": "completeness", "score": score, "comment": f"完整性: {score:.2f}"}


# 5. 上下文精度
def evaluate_context_precision(run: Run, example: Example) -> dict:
    answer = run.outputs.get("answer", "")
    sources = _parse_sources(run.outputs.get("sources", "[]"))
    question = example.inputs.get("question", "")
    if not sources:
        return {"key": "context_precision", "score": 0.0, "comment": "无检索文档"}
    if not answer:
        return {"key": "context_precision", "score": 0.0, "comment": "空答案"}

    docs_text = "\n---\n".join(
        f"[文档{i+1} - {s.get('filename','?')}]\n{s['content'][:300]}"
        for i, s in enumerate(sources[:5])
    )
    prompt = f"""你是一位检索质量评估专家。给定"用户问题"和"系统生成的答案"，请评估"检索文档"中有多大比例的信息对回答问题是有用的。

评分标准（1-5）：
5 = 几乎所有检索文档都直接有助于回答问题（高精度，低噪音）
3 = 约一半文档有用
1 = 几乎所有检索文档都与问题无关

只输出一个数字（1-5）。

用户问题：{question}
系统答案：{answer}
检索文档：
{docs_text}
上下文精度评分（1-5）:"""
    score = _llm_judge(prompt)
    return {"key": "context_precision", "score": score, "comment": f"上下文精度: {score:.2f}"}


# 6. 检索相关性
def evaluate_retrieval_relevance(run: Run, example: Example) -> dict:
    sources = _parse_sources(run.outputs.get("sources", "[]"))
    question = example.inputs.get("question", "")
    if not sources:
        return {"key": "retrieval_relevance", "score": 0.0, "comment": "无文档"}

    sources_text = "\n---\n".join(
        f"[{i+1} - {s.get('filename', '?')}]\n{s.get('content', '')[:300]}"
        for i, s in enumerate(sources[:3])
    )
    prompt = f"""请评估以下检索到的文档内容与用户问题的相关程度。

评分标准（1-5）：5=高度相关，3=部分相关，1=完全无关
只输出一个数字（1-5）。

用户问题：{question}
检索文档：{sources_text}
相关性评分（1-5）:"""
    score = _llm_judge(prompt)
    return {"key": "retrieval_relevance", "score": score, "comment": f"检索相关性: {score:.2f}"}


# 7. 答案长度
def evaluate_answer_length(run: Run, example: Example) -> dict:
    answer = run.outputs.get("answer", "")
    length = len(answer)
    if length == 0:
        return {"key": "answer_length", "score": 0.0, "comment": "答案为空"}
    if length < 20:
        return {"key": "answer_length", "score": 0.3, "comment": f"过短: {length}字"}
    if length > 2000:
        return {"key": "answer_length", "score": 0.7, "comment": f"较长: {length}字"}
    return {"key": "answer_length", "score": 1.0, "comment": f"合适: {length}字"}


# 8. 延迟
def evaluate_latency(run: Run, example: Example) -> dict:
    latency = run.outputs.get("latency_seconds", -1)
    if latency < 0:
        return {"key": "latency", "score": 0.0, "comment": "无法获取延迟"}
    if latency < 3:
        score = 1.0
    elif latency < 6:
        score = 0.8
    elif latency < 10:
        score = 0.5
    else:
        score = 0.2
    return {"key": "latency", "score": score, "comment": f"延迟: {latency:.2f}s"}


ALL_EVALUATORS = [
    evaluate_answer_correctness,
    evaluate_faithfulness,
    evaluate_answer_relevance,
    evaluate_completeness,
    evaluate_context_precision,
    evaluate_retrieval_relevance,
    evaluate_answer_length,
    evaluate_latency,
]

EVAL_NAMES = [
    "correctness", "faithfulness", "answer_relevance", "completeness",
    "context_precision", "retrieval_relevance", "answer_length", "latency",
]

# ============================================================
# 格式化输出
# ============================================================


def _format_table_header() -> str:
    header = f"{'问题':<36} " + " ".join(f"{n:<6}" for n in EVAL_NAMES) + "  avg"
    return f"\n{header}\n{'-' * len(header)}"


def _format_row(question: str, scores: dict[str, float]) -> str:
    q = question[:34] + ".." if len(question) > 36 else question
    cells = " ".join(f"{scores.get(name, 0):.2f}  " for name in EVAL_NAMES)
    avg = sum(scores.values()) / max(len(scores), 1)
    return f"{q:<36} {cells} {avg:.2f}"


def _format_averages(all_scores: list[dict[str, float]]) -> str:
    cols = []
    for name in EVAL_NAMES:
        vals = [s[name] for s in all_scores if name in s]
        cols.append(f"{sum(vals)/max(len(vals),1):.2f}  ")
    overall = sum(
        sum(s.values()) / max(len(s), 1) for s in all_scores
    ) / max(len(all_scores), 1)
    return f"{'【平均】':<36} {''.join(cols)} {overall:.2f}"


# ============================================================
# 结果下载（从 LangSmith 拉取并保存到本地）
# ============================================================


def download_results(version: str, experiment_name: str, all_scores: list[dict[str, float]]):
    """下载 LangSmith 评估结果并保存到 eval/{version}/results/"""
    results_dir = Path(__file__).resolve().parent / version / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    tz_beijing = timezone(timedelta(hours=8))
    timestamp = datetime.now(tz_beijing).strftime("%Y%m%d-%H%M%S")

    client = Client()

    # 计算各指标平均值
    averages = {}
    for name in EVAL_NAMES:
        vals = [s[name] for s in all_scores if name in s]
        if vals:
            averages[name] = round(sum(vals) / len(vals), 4)
    overall_avg = round(
        sum(averages.values()) / max(len(averages), 1), 4
    )

    # 构建详细结果
    summary = {
        "version": version,
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "num_examples": len(all_scores),
        "averages": averages,
        "overall_average": overall_avg,
        "metric_descriptions": {
            "correctness": "答案与标准答案的事实一致性",
            "faithfulness": "答案是否忠实于检索文档（反幻觉检测）",
            "answer_relevance": "答案是否直接回应用户问题",
            "completeness": "答案是否完整覆盖问题要点",
            "context_precision": "检索文档中真正有用的比例（去噪音）",
            "retrieval_relevance": "检索文档与问题的语义相关性",
            "answer_length": "答案长度合理性",
            "latency": "端到端响应延迟（秒）",
        },
    }

    # 保存 JSON 结果
    result_path = results_dir / f"result-{timestamp}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 也生成一份可读的 Markdown 报告
    md_path = results_dir / f"report-{timestamp}.md"
    md_lines = [
        f"# Agentic RAG 评估报告",
        f"",
        f"- **版本**: {version}",
        f"- **实验名称**: {experiment_name}",
        f"- **评估时间**: {timestamp}",
        f"- **测试样本数**: {len(all_scores)}",
        f"",
        f"## 指标结果",
        f"",
        f"| 指标 | 得分 | 说明 |",
        f"|------|:----:|------|",
    ]
    for name, score in averages.items():
        desc = summary["metric_descriptions"].get(name, "")
        md_lines.append(f"| {name} | {score:.2f} | {desc} |")
    md_lines.append(f"| **综合** | **{overall_avg:.2f}** | |")
    md_lines.append(f"")
    md_lines.append(f"## 原始结果")
    md_lines.append(f"")
    md_lines.append(f"```json")
    md_lines.append(json.dumps(summary, ensure_ascii=False, indent=2))
    md_lines.append(f"```")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    logger.info("结果已保存: %s", result_path)
    logger.info("报告已保存: %s", md_path)


# ============================================================
# 主流程
# ============================================================


def main():
    global _eval_config

    parser = argparse.ArgumentParser(description="Agentic RAG LangSmith 评估流水线")
    parser.add_argument(
        "--version", "-v",
        required=True,
        help="评估版本目录（如 v1, v2），对应 eval/{version}/",
    )
    parser.add_argument(
        "--enable-kg",
        action="store_true",
        help="启用知识图谱检索",
    )
    parser.add_argument(
        "--enable-multi-query",
        action="store_true",
        help="启用 Multi-Query 多角度检索",
    )
    parser.add_argument(
        "--enable-bm25",
        action="store_true",
        help="启用 BM25 关键词检索",
    )
    args = parser.parse_args()
    version = args.version

    _eval_config["enable_kg"] = args.enable_kg
    _eval_config["enable_multi_query"] = args.enable_multi_query
    _eval_config["enable_bm25"] = args.enable_bm25

    version_dir = Path(__file__).resolve().parent / version
    if not version_dir.is_dir():
        logger.error("版本目录不存在: %s", version_dir)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  Agentic RAG — LangSmith 评估 (版本: %s, 策略: kg=%s mq=%s bm25=%s)",
                version, _eval_config["enable_kg"], _eval_config["enable_multi_query"],
                _eval_config["enable_bm25"])
    logger.info("=" * 60)

    # 加载数据集
    examples = load_dataset(version)

    # 1. 索引文档
    logger.info("\n>>> [1/4] 索引测试文档")
    index_sample_docs(version)
    stats = vector_store.get_collection_stats()
    logger.info("向量库: name=%s, count=%d", stats["name"], stats["count"])
    if stats["count"] == 0:
        logger.error("向量库为空！请检查 eval/%s/sample_docs/", version)
        sys.exit(1)

    # 2. 创建 LangSmith 数据集
    logger.info("\n>>> [2/4] 创建 LangSmith 数据集")
    dataset_name = create_langsmith_dataset(version, examples)

    # 3. 运行评估
    logger.info("\n>>> [3/4] 运行评估（%d 题 × %d 评估器）", len(examples), len(ALL_EVALUATORS))
    results = evaluate(
        target_fn,
        data=dataset_name,
        evaluators=ALL_EVALUATORS,
        experiment_prefix=f"agentic-rag-{version}",
        max_concurrency=1,
    )

    # 输出详细结果表
    all_scores: list[dict[str, float]] = []
    logger.info(_format_table_header())
    if hasattr(results, "_results"):
        for r in results._results:
            q = r["run"].inputs.get("question", "?")
            feedbacks = r["evaluation_results"]["results"]
            scores = {f.key: f.score for f in feedbacks}
            all_scores.append(scores)
            logger.info(_format_row(q, scores))
    logger.info(_format_averages(all_scores))

    if not all_scores:
        logger.warning("未获取到评分数据，跳过结果保存")

    # 4. 下载并保存结果
    logger.info("\n>>> [4/4] 下载并保存评估结果到 eval/%s/results/", version)
    download_results(version, results.experiment_name, all_scores)

    # 总览
    logger.info("\n" + "=" * 60)
    logger.info("  评估完成！")
    logger.info("  版本: %s", version)
    logger.info("  实验名称: %s", results.experiment_name)
    logger.info("  LangSmith: %s", results.url)
    logger.info("  本地结果: eval/%s/results/", version)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
