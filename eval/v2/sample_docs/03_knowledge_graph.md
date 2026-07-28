# 知识图谱模块设计与实现

## 1. 模块架构

知识图谱模块位于 `src/knowledge_graph/`，包含四个核心组件：

| 组件 | 文件 | 职责 |
|------|------|------|
| GraphStore | graph_store.py | 图数据存储与操作 |
| GraphBuilder | graph_builder.py | 从文档抽取实体关系自动建图 |
| GraphRetriever | graph_retriever.py | 图谱检索与推理 |
| KGIntentAnalyzer | kg_intent.py | 问题类型分析与路由决策 |

## 2. GraphStore 图存储

GraphStore 基于 NetworkX 的有向图实现，使用 JSON 进行本地持久化：

### 2.1 核心数据结构

- **节点（实体）**：每个实体包含 name、type、aliases、doc_ids、properties 等属性
- **边（关系）**：连接两个实体，包含 relation 类型、weight 权重、doc_id 来源

### 2.2 实体类型体系

支持的实体分类：
- **person**：人物
- **org**：组织/公司
- **location**：地点
- **product**：产品/服务
- **concept**：概念/术语
- **time**：时间点/时期
- **event**：事件
- **other**：其他

### 2.3 关系类型体系

| 关系类型 | 说明 | 示例 |
|---------|------|------|
| is_a | 从属关系 | Python is_a 编程语言 |
| part_of | 组成关系 | 向量检索 part_of 检索系统 |
| created_by | 创建者关系 | NexusML created_by 张小伟 |
| located_in | 地理位置 | 总部 located_in 北京 |
| works_for | 雇佣关系 | 李明 works_for 小象科技 |
| related_to | 关联关系 | 知识图谱 related_to 检索系统 |
| has_feature | 属性特征 | 系统 has_feature 多策略检索 |
| causes | 因果关系 | 文档不相关 causes 联网搜索 |
| depends_on | 依赖关系 | 知识图谱 depends_on NetworkX |

### 2.4 核心操作

- **add_entity / add_relation**：添加单个实体/关系
- **add_entities_batch / add_relations_batch**：批量添加
- **search_entities**：模糊搜索实体（支持子串匹配）
- **get_subgraph**：以种子实体为中心的多跳子图提取
- **find_paths**：两实体间的多跳路径查找
- **save / load**：JSON 持久化读写

## 3. GraphBuilder 图谱构建

文档入库时自动触发图谱构建流程：

### 3.1 抽取流程

1. 对文档的每个 chunk 进行 LLM 实体关系抽取（使用 qwen-turbo 以降低成本）
2. 将 JSON 格式的抽取结果解析为实体列表和关系列表
3. 批量写入 GraphStore（自动去重：同名实体合并，同一实体对同名关系跳过）
4. JSON 持久化到 `kg_data/knowledge_graph.json`

### 3.2 Prompt 设计

实体抽取 Prompt 要求 LLM 返回结构化 JSON：
- entities 数组：name（实体名称）、type（类型）、aliases（别名）
- relations 数组：source（源实体）、target（目标实体）、relation（关系类型）

严格控制输出为纯 JSON，要求只抽取文本中明确提到的实体，不推测。

## 4. KGIntentAnalyzer 意图分析

在 Agent 工作流的入口节点执行，判断用户问题是否需要调用知识图谱：

### 4.1 适用 KG 的问题类型

- **实体关系查询**：A 和 B 是什么关系？
- **多跳推理**：A 的创建者还创建了哪些产品？
- **属性查询**：A 有哪些特性？
- **比较对比**：A 和 B 有什么区别？
- **层级/从属**：A 包含哪些子类别？
- **因果链**：事件 A 导致了什么？

### 4.2 不适用 KG 的问题类型

- 简单定义/概念解释
- 操作指引/教程
- 个人观点/建议
- 翻译/改写任务

### 4.3 降级保护

三重降级保护确保系统稳定性：
1. 图谱为空 → 自动标记 `kg_intent=False`
2. LLM 意图分析异常 → 默认 `kg_intent=False`
3. `enable_kg=False` → 完全跳过 KG 路径

## 5. GraphRetriever 图谱检索

处理 KG 适用问题的检索流程：

### 5.1 检索步骤

1. **实体抽取**：LLM 从 query 提取关键实体名词（最多10个）
2. **实体链接**：在 GraphStore 中搜索匹配实体
   - 优先精确名称匹配
   - 次选别名匹配
   - 兜底：语义相似度匹配（使用 Embedding）
3. **子图提取**：BFS 广度优先搜索拓展关联实体（最大2跳）
4. **路径推理**：查找种子实体间的多跳连接路径
5. **上下文生成**：将实体信息、关系、路径转换为自然语言文本

### 5.2 检索结果缓存

检索结果以结构化文本形式存储在 `kg_context` 字段中，在 merge_retrieval 节点作为特殊 Document（标记 source="knowledge_graph"）插入到文档列表最前面。

这确保了知识图谱的结构化知识在生成阶段被优先使用。

## 6. 与原有 RAG 流程的无缝集成

### 6.1 Agent 图修改点

- 新增 `analyze_kg_intent` 作为入口节点（原入口为 `retrieve`）
- 新增 `kg_retrieve` 作为并行检索策略之一
- route_retrieval_strategies 函数增加 KG 条件分支
- merge_retrieval_node 增加 KG 上下文合并逻辑

### 6.2 数据流变化

```
原始流程: START → retrieve → [并行策略] → merge → ... → END
新流程:   START → analyze_kg_intent → retrieve → [并行策略+KG] → merge → ... → END
```

### 6.3 前端集成

- ChatInput 增加橙色「知识图谱」开关按钮（Network 图标）
- AgentPathBadge 新增 analyze_kg_intent 和 kg_retrieve 节点标签
- Sidebar SVG 流程图扩展为五路并行检索
- enable_kg 参数通过 API 请求传递给后端

## 7. 存储与持久化

- 图谱数据保存在 `kg_data/knowledge_graph.json`
- 启动时自动加载已有数据
- 每次文档入库后自动保存
- 支持增量构建：新文档的实体/关系自动合并到现有图谱
