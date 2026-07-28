# 开源 AI 工具链与框架生态

## 1. 向量数据库生态

### 1.1 ChromaDB

ChromaDB 是轻量级开源向量数据库，支持嵌入存储和语义检索。由 Chroma 团队开发，采用 Apache 2.0 许可证。

核心特性：
- 支持本地持久化和客户端-服务器两种模式
- 内置多种 Embedding 函数（OpenAI、HuggingFace）
- 元数据过滤支持
- 自动距离计算（余弦相似度、欧氏距离等）

ChromaDB 是本项目使用的向量数据库，数据存储在 chroma_data 目录下。

### 1.2 其他主流向量数据库

| 数据库 | 开发者 | 核心特点 |
|--------|-------|---------|
| Pinecone | Pinecone Inc. | 全托管云服务，超高并发 |
| Milvus | Zilliz | 云原生分布式，支持 GPU 索引 |
| Weaviate | SeMI Technologies | 内置向量化模块，GraphQL 接口 |
| Qdrant | Qdrant Team | Rust 实现，高性能过滤 |
| FAISS | Meta AI | 纯库模式，极致本地性能 |

## 2. LLM 应用框架

### 2.1 LangChain

LangChain 由 Harrison Chase 创建，是 LLM 应用开发的事实标准框架：

- 链式组合（Chains）：串联多个处理步骤
- 工具调用：LLM 调用外部 API 和函数
- 检索模块：集成多种向量数据库和检索策略
- Agent 模块：自主规划和执行任务

### 2.2 LangGraph

LangGraph 是 LangChain 团队推出的状态图框架，用于构建复杂的多步骤 Agent：

- StateGraph：基于状态机的有向图编排
- 条件边：根据状态动态路由
- Send API：fan-out/fan-in 并行执行
- Checkpointing：状态持久化和重放

本项目基于 LangGraph 构建整个 Agent 工作流。

### 2.3 其他框架对比

| 框架 | 开发者 | 定位 | 适合场景 |
|------|--------|------|---------|
| LlamaIndex | Jerry Liu | 数据索引与检索 | 文档问答 |
| Haystack | deepset | NLP 管道 | 企业搜索 |
| DSPy | Stanford | 声明式编程 | 复杂 NLP 管道 |
| AutoGen | Microsoft | 多Agent协作 | 对话式任务 |

## 3. 模型服务与部署

### 3.1 推理引擎

- **vLLM**：UC Berkeley 开发的高性能推理引擎，PagedAttention 实现显存高效管理
- **TensorRT-LLM**：NVIDIA 推出的 LLM 推理优化框架，GPU 利用率极高
- **TGI (Text Generation Inference)**：HuggingFace 官方的生产级推理服务
- **SGLang**：结构化生成语言，通过 RadixAttention 优化批量推理

### 3.2 量化技术

- **GPTQ**：基于训练后量化的方法，适合 GPU 部署
- **AWQ (Activation-aware Weight Quantization)**：保留重要通道精度，效果优于 GPTQ
- **GGUF**：llama.cpp 使用的格式，适合 CPU 推理
- **bitsandbytes**：HuggingFace 集成，支持 8bit/4bit 量化

## 4. 评估与测试工具

### 4.1 RAG 评估

- **RAGAS**：专为 RAG 设计的评估框架，评估忠实度、相关性、上下文召回等指标
- **DeepEval**：LLM 评估框架，支持 RAG 和通用 LLM 任务
- **TruLens**：可观测性和评估工具，追踪 RAG 管道的每一步
- **LangSmith**：LangChain 官方的调试和评估平台

### 4.2 测试集构建

高质量的 RAG 测试集应包含：
- 简单的事实查询（验证基础检索）
- 多跳推理问题（验证高级检索和 KG）
- 比较类问题（验证实体关系抽取）
- 定义类问题（验证语义检索准确性）
- 不相关问题（验证降级和过滤机制）

## 5. 监控与可观测性

### 5.1 LLM 可观测性

- **LangSmith**：追踪 LLM 调用的完整链路
- **Phoenix (Arize)**：开源 LLM 可观测性平台
- **Weights & Biases**：模型训练和 LLM 应用监控
- **OpenTelemetry**：通用可观测性标准

### 5.2 关键监控指标

- Token 消耗和成本追踪
- 检索延迟和吞吐量
- Agent 路径分布（哪些节点被调用）
- 幻觉检测率和重试次数
- 文档引用率和答案相关性得分
