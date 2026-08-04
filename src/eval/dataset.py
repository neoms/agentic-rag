"""评估数据集工程 - 标准 schema 校验与加载

标准 schema（JSONL，每行一个对象）：
    {
      "question": "用户问题（必填）",
      "reference": "标准答案（必填）",
      "reference_contexts": ["golden 上下文1", "..."]  // 可选；缺失时跳过
      //   context_precision / context_recall 两个检索指标
    }

旧版 v1/v2 数据集（仅 question+answer）不迁移，不参与新体系。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalSample:
    """单条评估样本"""

    question: str
    reference: str
    reference_contexts: list[str] = field(default_factory=list)


def validate_dataset_file(path: Path) -> list[str]:
    """校验数据集文件，返回错误信息列表（空 = 合法）"""
    errors: list[str] = []
    if not path.exists():
        return [f"数据集文件不存在: {path}"]
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        return [f"无法读取数据集: {e}"]

    if not any(line.strip() for line in lines):
        return ["数据集为空"]

    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"第 {i} 行不是合法 JSON: {e}")
            continue
        if not isinstance(obj, dict):
            errors.append(f"第 {i} 行必须是 JSON 对象")
            continue
        question = obj.get("question")
        reference = obj.get("reference")
        if not isinstance(question, str) or not question.strip():
            errors.append(f"第 {i} 行缺少非空字符串字段 question")
        if not isinstance(reference, str) or not reference.strip():
            errors.append(f"第 {i} 行缺少非空字符串字段 reference")
        if "reference_contexts" in obj:
            rcs = obj["reference_contexts"]
            if not isinstance(rcs, list) or not all(
                isinstance(x, str) and x.strip() for x in rcs
            ):
                errors.append(
                    f"第 {i} 行 reference_contexts 必须是字符串数组（每项非空）"
                )
    return errors


def load_dataset(path: Path) -> list[EvalSample]:
    """加载数据集；schema 非法时抛出 ValueError（含行号错误）"""
    errors = validate_dataset_file(path)
    if errors:
        raise ValueError("数据集校验失败:\n" + "\n".join(f"  - {e}" for e in errors))

    samples: list[EvalSample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        samples.append(
            EvalSample(
                question=obj["question"],
                reference=obj["reference"],
                reference_contexts=obj.get("reference_contexts") or [],
            )
        )
    return samples
