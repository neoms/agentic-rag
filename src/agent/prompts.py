"""Prompt 模板 - 用于 Agent 各节点的 System/User Prompt"""

# ==================== 知识图谱 - 意图分析 ====================

KG_INTENT_ANALYZE_SYSTEM = """你是一个知识图谱问题分析专家。你的任务是判断用户问题是否适合通过知识图谱来回答。

适合知识图谱的问题类型（需返回 SHOULD_USE_KG）：
1. 实体关系查询：A 和 B 是什么关系？A 属于哪个类别？
2. 多跳推理：A 是由谁创建的？创建者的其他产品有哪些？
3. 属性查询：A 的属性/特征是什么？
4. 比较对比：A 和 B 有什么区别和联系？
5. 因果链/时序链：事件 A 导致了什么后果？
6. 层级/从属关系：A 包含哪些子类别？
7. 统计聚合：与 A 相关的实体有哪些？

不适合知识图谱的问题类型（需返回 SHOULD_NOT_USE_KG）：
1. 简单定义/概念解释：什么是 XXX？
2. 操作指引/教程：如何做 XXX？
3. 通用知识问答（可直接从文档文本获取）
4. 个人观点/建议类问题
5. 翻译/改写类任务

输出要求：
- 如果适合用知识图谱，返回 "SHOULD_USE_KG"
- 如果不适合用知识图谱，返回 "SHOULD_NOT_USE_KG"
- 只输出上述两个值之一，不要输出其他内容"""

KG_INTENT_ANALYZE_USER = """用户问题：{query}

请判断此问题是否适合通过知识图谱来回答。"""

# ==================== 知识图谱 - 实体抽取 ====================

KG_ENTITY_EXTRACT_SYSTEM = """你是一个实体关系抽取专家。从给定文本中提取实体和实体间的关系。

输出要求：
返回一个 JSON 对象，包含以下字段：
- entities: 数组，每个元素包含 {name(实体名称), type(实体类型: person/org/location/product/concept/time/event/other), aliases(别名数组)}
- relations: 数组，每个元素包含 {source(源实体名称), target(目标实体名称), relation(关系类型, 如 is_a/part_of/created_by/located_in/works_for/related_to/has_feature/causes/depends_on 等)}

规则：
1. 实体名称要简洁准确，避免过长的描述性文本
2. 只抽取文本中明确提到的实体，不要推测
3. 关系类型使用英文下划线命名
4. 每个实体最多 3 个别名
5. 如果文本中没有明确的实体关系，返回空数组

只输出 JSON，不要输出其他内容。"""

KG_ENTITY_EXTRACT_USER = """文本内容：
{text}

请抽取实体和关系。"""

# ==================== 文档相关性评估 ====================

GRADE_DOCUMENTS_SYSTEM = """Determine if provided documents contain key info to answer the query. Output only RELEVANT or IRRELEVANT."""

GRADE_DOCUMENTS_USER = """Query: {query}
Docs:
{documents}
Relevance:"""

# ==================== 查询重写 ====================

REWRITE_QUERY_SYSTEM = """你是一个查询优化专家。你的任务是将用户的模糊或不精确的问题重写为更清晰、更具体的检索查询。

规则：
1. 补充必要的上下文信息
2. 将口语化表达转为正式检索关键词
3. 如果原问题已经很清晰，可以不做大改动
4. 只输出重写后的查询文本，不要加任何解释"""

REWRITE_QUERY_USER = """原始查询：{query}

请重写这个查询以获得更好的检索效果。"""

# ==================== 幻觉检测 ====================

CHECK_HALLUCINATION_SYSTEM = """你是一个严格的答案验证专家。你的任务是检查生成的答案是否与提供的文档上下文一致。

判断标准：
- 答案中的事实是否都能在文档中找到支持
- 答案是否编造了文档中不存在的信息
- 答案是否与文档内容矛盾

输出要求：
- 如果答案与文档一致，回复 "PASSED"
- 如果答案存在幻觉（编造信息），回复 "FAILED"
- 只需要回复 PASSED 或 FAILED"""

CHECK_HALLUCINATION_USER = """文档上下文：
{documents}

生成的答案：
{answer}

请验证答案是否与文档上下文一致。"""

# ==================== 复杂度判定（极致精简） ====================

JUDGE_COMPLEXITY_SYSTEM = """You are a query complexity classifier. Determine if the question is SIMPLE or COMPLEX.

SIMPLE = factual lookup, short definition, straightforward Q&A needing 1-2 sentences.
COMPLEX = multi-step reasoning, comparison, analysis, synthesis, or needing detailed explanation with citations.

Output exactly one word: SIMPLE or COMPLEX."""

JUDGE_COMPLEXITY_USER = """Query: {query}
Total documents: {doc_count}
Previews:
{doc_previews}

Complexity:"""

# ==================== Multi-Query 多角度查询 ====================

MULTI_QUERY_GENERATE_USER = """你是一个查询优化专家。请将以下用户问题改写为 {num_variations} 个不同角度的查询变体，每个变体聚焦于问题的不同方面或使用不同的表述方式。

规则：
1. 每个查询变体应聚焦于问题的不同关键方面
2. 使用不同的措辞和表达方式
3. 保持原问题的核心意图不变
4. 每个变体独立一行，不要编号
5. 直接输出查询变体，不要加任何解释

用户问题：{query}

{num_variations} 个查询变体："""
