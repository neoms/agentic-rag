# Agentic RAG 系统架构详解

## 1. 项目概述

Agentic RAG 是一个基于 LangGraph 构建的多策略智能检索增强生成系统，采用"可配置开关 + 并行检索 + 自反思"的架构设计。系统使用阿里云百炼（DashScope）作为大语言模型和嵌入模型的后端，ChromaDB 作为向量数据库，NetworkX 作为知识图谱存储引擎。

## 2. 核心技术栈

| 组件 | 技术选型 | 用途 |
|------|---------|------|
| LLM | qwen-plus (阿里百炼) | 答案生成、实体抽取、意图分析 |
| Embedding | text-embedding-v3 (阿里百炼) | 文本向量化 |
| 向量存储 | ChromaDB (PersistentClient) | 文档语义检索 |
| 知识图谱 | NetworkX + JSON | 结构化知识存储 |
| Agent 框架 | LangGraph StateGraph | 多节点编排与路由 |
| Web 框架 | FastAPI | API 服务 |
| 前端 | Vue 3 + TypeScript | 交互界面 |

## 3. Agent 工作流

系统的核心是一个有向状态图，包含15个节点：

### 3.1 入口节点：analyze_kg_intent

知识图谱意图分析节点，在检索之前执行。它通过 LLM 分析用户问题的特征，判断是否需要调用知识图谱。如果问题涉及实体关系查询、多跳推理、比较对比等结构化的知识需求，则标记 `kg_intent=True`；否则标记为 `kg_intent=False`，走纯文本检索路径。

### 3.2 语义检索节点：retrieve

执行基于向量的语义相似度检索，同时结合 MMR（最大边际相关性）算法混合检索。语义检索捕获语义层面的相似性，MMR 则通过多样性重排避免结果冗余，lambda_mult=0.7 在相似度和多样性之间取得平衡。

### 3.3 增强检索策略

系统支持四种可选的高级检索策略，通过 Send API 实现并行 fan-out：

- **BM25 检索**：基于词频-逆文档频率的关键词匹配，适合精确术语查找。对中文分词通过 jieba 实现。
- **HyDE 检索**：先让 LLM 生成一个虚拟答案，再用该虚拟答案的向量去检索，缩小语义鸿沟。
- **Multi-Query 检索**：从多角度自动改写查询，生成3-5个不同视角的子查询分别检索，提高覆盖面。
- **知识图谱检索**：从 GraphStore 中提取实体关系，通过子图拓展和多跳路径推理得到结构化知识。

### 3.4 结果合并：merge_retrieval

所有并行的检索策略结果在此节点集中收敛（fan-in）。去重策略基于文档 page_content 的哈希值，相同内容只保留一条。知识图谱的检索结果以特殊 Document 形式插入到列表最前面，确保 LLM 优先参考结构化的知识。

### 3.5 重排序：rerank_documents

使用 LLM 对候选文档重新排序，通过 Cross-Encoder 方式评估每个文档与查询的相关性，返回带排序的结果。该节点可通过 `enable_rerank` 开关控制。

### 3.6 文档评估：grade_documents

LLM 对文档进行二元相关性评估（相关/不相关）。如果所有文档都不相关，根据配置可能触发查询重写或联网搜索降级。

### 3.7 联网搜索：web_search

当本地文档都不相关时，通过 DuckDuckGo 搜索互联网获取信息，作为降级方案。

### 3.8 查询重写：transform_query

当检索质量不佳时，LLM 自动优化查询语句，改进检索效果。

### 3.9 答案生成：generate

基于最终的文档上下文和对话历史，由 LLM 生成结构化的最终答案。

### 3.10 幻觉检测：check_hallucination

对生成答案进行自反思检测，判断答案中的信息是否都能在提供文档中找到依据。如果检测到幻觉，触发重新生成。

## 4. Fan-out/Fan-in 并行模式

系统使用 LangGraph 的 `Send` API 实现动态并行检索：

```
retrieve 节点 → conditionEdges 函数
  ├── [enable_bm25=True] → Send to bm25_retrieve
  ├── [enable_hyde=True] → Send to hyde_retrieve  
  ├── [enable_multi_query=True] → Send to multi_query_retrieve
  ├── [enable_kg=True and kg_intent=True] → Send to kg_retrieve
  └── [全部关闭] → 直达 merge_retrieval

各并行节点 → merge_retrieval (fan-in 收敛)
```

这种设计允许根据场景灵活组合检索策略，关闭的节点完全不消耗计算资源。

## 5. 配置系统

所有运行时参数通过环境变量或 .env 文件配置，由 pydantic-settings 管理。关键配置项包括：

- LLM 模型名称、Base URL、API Key
- Embedding 模型名称
- 文档分块大小（默认500字符）和重叠量（默认100字符）
- 检索 Top-K（默认8）和相似度阈值
- 知识图谱最大跳数（默认2）和最大实体数（默认10）
- 知识图谱数据目录（默认 kg_data）
