# Agentic RAG

基于 LangGraph 构建的智能检索增强生成（Agentic RAG）系统。支持多策略并行检索、知识图谱推理、重排序精排、自反思评估、联网搜索降级、工具调用、流式输出等高级能力。前后端分离架构，后端 FastAPI + 前端 Vue 3。

## 架构特点

- **多策略并行检索**：语义检索 + MMR 多样性、BM25 关键词、HyDE 假设文档嵌入、Multi-Query 多角度查询，通过 LangGraph Send API 实现 fan-out/fan-in 并行执行，可独立开关控制
- **知识图谱模块**：Kuzu 图数据库（原生持久化）+ numpy .npz 二进制向量索引，LLM 自动抽取实体关系构建图谱，支持意图分析路由、实体链接、子图提取、多跳路径推理，不适用时平滑降级到原有 RAG 流程
- **重排序精排**：LLM Cross-Encoder 对检索结果二次排序，提升 Top-K 文档质量
- **Agent 智能体**：自反思能力（文档评估 → 查询重写 → 幻觉检测）；支持 Tool Calling（联网搜索）
- **联网搜索降级**：向量库无匹配时自动走 DuckDuckGo 网页搜索，结果带来源 URL
- **LangSmith 评估**：8 维度自动化评估流水线（正确性/忠实度/相关性/完整性/上下文精度/延迟等），版本化目录管理（v1/v2）
- **百炼平台统一接入**：LLM 使用 OpenAI 兼容协议，Embedding 使用官方 DashScope SDK
- **文档分块**：`RecursiveCharacterTextSplitter`，chunk_size=500、chunk_overlap=100
- **FastAPI + SSE 流式输出**：自动生成 Swagger 文档
- **ChromaDB 本地持久化**：零外部依赖，数据保存在 `data/chroma/` 目录，文档元数据（doc_id/filename/hash）随块 metadata 一并存储，无需独立注册表
- **Vue 3 前端**：Vite + TypeScript + TailwindCSS，支持多策略切换、会话历史、拖拽上传、Agent 流程图可视化

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的百炼 API Key
```

必填环境变量：

| 变量 | 说明 | 示例 |
|------|------|------|
| `DASHSCOPE_API_KEY` | 百炼 API Key | `sk-xxx` |

可选环境变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_MODEL` | 默认 LLM 模型 | `qwen-plus` |
| `LLM_MODEL_FAST` | 快速评估模型 | `qwen-turbo` |
| `LLM_MODEL_STRONG` | 强生成模型 | `qwen-max` |
| `EMBEDDING_MODEL` | 嵌入模型 | `text-embedding-v4` |
| `CHROMA_PERSIST_DIR` | ChromaDB 数据目录 | `data/chroma` |
| `RETRIEVAL_TOP_K` | 检索候选数 | `20` |
| `RETRIEVAL_SIMILARITY_THRESHOLD` | 语义检索过滤阈值 | `0.5` |
| `RERANK_ENABLED` | 是否启用重排序 | `true` |
| `RERANK_MODEL` | 重排序模型 | `gte-rerank` |
| `RERANK_TOP_K` | 重排序后保留数 | `5` |
| `GRADE_SCORE_IRRELEVANT_MAX` | 文档评估负判定：全体文档最高分 ≤ 此值直接不相关（0 LLM） | `0.25` |
| `GRADE_SCORE_RELEVANT_MIN` | 文档评估正判定：top1 分数下限 | `0.70` |
| `GRADE_SCORE_RELEVANT_GAP` | 文档评估正判定：top1 与 top2 最小分差（断层检测） | `0.10` |
| `MEMORY_WINDOW_SIZE` | 对话记忆窗口 | `20` |
| `MAX_UPLOAD_SIZE_MB` | 上传文件大小限制 | `10` |
| `KG_DATA_DIR` | 知识图谱数据目录 | `kg_data` |
| `KG_MAX_HOPS` | KG 子图最大跳数 | `2` |
| `KG_MAX_ENTITIES` | 单次查询最多实体数 | `10` |
| `CORS_ALLOWED_ORIGINS` | CORS 允许来源（逗号分隔多个） | `http://localhost:3000` |
| `SESSION_MESSAGE_CAP` | 单会话最多保留消息条数（内存 + 库） | `500` |
| `TASK_HISTORY_KEEP` | 上传任务保留条数（超出删除最旧） | `100` |
| `TASK_HISTORY_TTL_DAYS` | 已完成/失败任务保留天数（0 = 仅按条数限制） | `7` |
| `INDEX_WORKERS` | 后台索引并发 worker 数 | `2` |
| `INDEX_QUEUE_MAX` | 排队任务上限（超出拒绝上传） | `20` |
| `CACHE_ENABLED` | 是否启用多级缓存 | `true` |
| `CACHE_EXACT_ENABLED` | 是否启用精准缓存 | `true` |
| `CACHE_SEMANTIC_ENABLED` | 是否启用语义缓存 | `true` |
| `CACHE_SEMANTIC_THRESHOLD` | 语义缓存命中阈值（余弦相似度） | `0.92` |
| `CACHE_MAX_ENTRIES` | 缓存条目上限（LRU 淘汰） | `5000` |
| `CACHE_TTL_SECONDS` | 缓存过期时间（0 = 不过期） | `0` |
| `CACHE_CITATION_MAX_CHARS` | 缓存引文段落最大长度 | `500` |
| `LOG_FILE` | 滚动日志文件路径 | `log/app.log` |

LangSmith 追踪（可选）：

| 变量 | 说明 | 示例 |
|------|------|------|
| `LANGSMITH_API_KEY` | LangSmith API Key | `lsv2_pt_xxx` |
| `LANGSMITH_PROJECT` | LangSmith 项目名 | `agentic-rag` |
| `LANGSMITH_TRACING` | 是否启用追踪 | `true` |

### 2. 启动后端

```bash
# 安装依赖
uv sync

# 启动应用
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# 或直接运行入口文件（效果等同）
uv run python main.py
```

启动后访问 http://localhost:8000/docs 查看 Swagger 文档。

### 3. 启动前端

```bash
cd front
npm install
npm run dev       # 启动开发服务器（端口 3000，API 代理到 8000）
npm run build     # 构建生产包到 front/dist/
```

前端运行在 http://localhost:3000。

### 4. Docker 部署（推荐）

```bash
# 1. 准备环境变量（至少填写 DASHSCOPE_API_KEY）
cp .env.example .env
# 编辑 .env 填入 API Key 与模型配置

# 2. 构建并启动（前端 nginx 反代后端，访问 http://localhost:8080）
docker compose up -d --build
```

- 前端：http://localhost:8080（nginx 托管静态文件并反代 `/api`、`/health` 到后端）
- 后端健康检查：http://localhost:8080/health
- Prometheus 指标：http://localhost:8080/metrics

### 数据持久化与备份

所有数据落在命名卷 `agentic_rag_data`（挂载到容器 `/app/data`）：

| 目录/文件 | 内容 |
|------|------|
| `chroma/` | 向量库（文档块 + 元数据） |
| `kg/kuzu_db/` | 知识图谱 |
| `state/state.db` | 会话历史 / 上传任务 |
| `cache/cache.db` | 多级缓存（精准 + 语义） |
| `temp_uploads/` | 上传中转（自动清理，无需备份） |

备份：

```bash
docker run --rm -v agentic_rag_data:/data -v "$PWD":/backup \
  alpine tar czf /backup/agentic-rag-data-$(date +%F).tar.gz -C / data
```

恢复：

```bash
docker run --rm -v agentic_rag_data:/data -v "$PWD":/backup \
  alpine sh -c "rm -rf /data/* && tar xzf /backup/agentic-rag-data-2026-XX-XX.tar.gz -C /"
docker compose restart backend
```

注意：备份前建议先 `docker compose stop backend` 保证文件一致；向量库、图库、缓存需整体备份（三者内部互相引用 doc_id）。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查（逐组件明细；`?deep=true` 额外做 Embedding 探针） |
| GET | `/metrics` | Prometheus 指标（文本格式，供采集器抓取） |
| POST | `/api/v1/documents/upload` | 上传文档（PDF/MD/TXT，最大 10MB） |
| GET | `/api/v1/documents` | 列出已索引文档 |
| DELETE | `/api/v1/documents/{doc_id}` | 删除文档及其向量块 |
| POST | `/api/v1/chat/stream` | Agent 流式对话（SSE，事件类型：source/path/token/done/error/hallucination等） |
| GET | `/api/v1/chat/history/{session_id}` | 获取会话历史 |
| DELETE | `/api/v1/chat/history/{session_id}` | 删除会话历史（含持久化数据） |

### Agentic 请求体参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `query` | string | 用户问题 | 必填 |
| `session_id` | string | 会话 ID | 自动生成 |
| `stream` | bool | 是否流式输出 | `false` |
| `enable_web_search` | bool | 联网搜索 | `false` |
| `enable_reflection` | bool | 自反思/幻觉检测 | `true` |
| `enable_rerank` | bool | 重排序 | `true` |
| `enable_grade_documents` | bool | 文档相关性评估 | `true` |
| `enable_transform_query` | bool | 查询重写 | `true` |
| `enable_bm25` | bool | BM25 关键词检索 | `false` |
| `enable_hyde` | bool | HyDE 假设文档检索 | `false` |
| `enable_multi_query` | bool | Multi-Query 多角度检索 | `false` |
| `enable_kg` | bool | 知识图谱检索 | `false` |

## LangGraph Agent 状态图

系统包含 15 个节点，通过可配置开关控制执行路径：

```
START
  │
  ▼
analyze_kg_intent ──── (KG 意图分析，决定是否走 KG 路径)
  │
  ▼
retrieve (语义+MMR)
  │
  ├──[bm25_retrieve]───┐   ← 条件 Send，fan-out 并行
  ├──[hyde_retrieve]───┤
  ├──[multi_query_retrieve]┤
  ├──[kg_retrieve]─────┤   ← 仅 kg_intent=True 时触发
  │                    │
  └────────────────────┤
                       ▼
              merge_retrieval (fan-in 收敛，合并去重)
                       │
              ┌────────┼──────────┐
              ▼        ▼          ▼
        rerank ON  rerank OFF  rerank OFF
              │   +grade ON   +grade OFF
              ▼        │          │
       rerank_documents │          │
              │        │          │
              ▼        ▼          │
       grade_documents ←─────────┘
              │
       ┌──────┼──────┐
       ▼      ▼      ▼
   [相关]  [不相关]  [不相关+联网]
       │      │      │
       │      ▼      ▼
       │  transform  web_search
       │  _query      │
       │      │      │
       └──────┴──────┘
              │
              ▼
          generate
              │
              ▼
     check_hallucination ──→ [有幻觉] → generate (重试)
              │
              ▼
            END
```

### 节点说明

| 节点 | 功能 | 模型 |
|------|------|------|
| `analyze_kg_intent` | LLM 分析问题是否需要知识图谱（实体关系/多跳推理 → KG；定义/教程 → 降级） | qwen-turbo |
| `retrieve` | 语义检索 + MMR 多样性检索，合并去重 | Embedding |
| `bm25_retrieve` | BM25 关键词检索，使用 jieba 分词，适合精确术语匹配 | jieba + BM25 |
| `hyde_retrieve` | 生成假设答案 → 向量化 → 用假设答案检索，缩小语义鸿沟 | qwen-turbo + Embedding |
| `multi_query_retrieve` | 多角度改写查询（3-5 个子查询），独立检索后去重合并 | qwen-turbo + Embedding |
| `kg_retrieve` | LLM 抽取实体 → Entity Linking → BFS 子图提取 → 多跳路径推理 → 结构化上下文 | qwen-turbo + Embedding |
| `merge_retrieval` | fan-in 收敛点：合并所有检索策略结果，按内容去重，KG 上下文优先附加 | — |
| `rerank_documents` | LLM Cross-Encoder 对合并文档做二次精排 | qwen-turbo |
| `grade_documents` | 严格评估文档是否包含能回答问题的关键信息 | qwen-turbo |
| `web_search` | 向量库无匹配时，DuckDuckGo 搜索网页作为降级方案 | HTTP |
| `transform_query` | 文档不相关时自动重写查询，优化检索质量 | qwen-turbo |
| `generate` | 基于检索文档 + KG 上下文生成最终答案 | qwen-max |
| `check_hallucination` | 检测答案是否与源文档一致，不一致则重试（最多 3 次） | qwen-turbo |

## 知识图谱模块

### 核心组件

| 组件 | 文件 | 功能 |
|------|------|------|
| GraphStore | `src/knowledge_graph/graph_store.py` | Kuzu 图数据库存储（原生持久化），支持实体/关系 CRUD、子图提取、多跳路径查找 |
| GraphBuilder | `src/knowledge_graph/graph_builder.py` | 文档入库时 LLM 自动抽取实体关系构建图谱 |
| GraphRetriever | `src/knowledge_graph/graph_retriever.py` | 实体抽取 → Entity Linking（精确+别名+numpy 语义搜索）→ BFS 子图 → 路径推理 → 上下文生成 |
| KGIntentAnalyzer | `src/knowledge_graph/kg_intent.py` | LLM 意图分析，判定问题是否适合 KG 查询 |

### 工作流程

```
用户问题
    │
    ▼
analyze_kg_intent (LLM 意图分析)
    │
    ├── SHOULD_NOT_USE_KG → 降级，只走语义检索
    │
    └── SHOULD_USE_KG + enable_kg=True
            │
            ▼
        kg_retrieve
            │
            ├── LLM 从问题抽取实体
            ├── Entity Linking（精确匹配 → 别名 → 语义兜底）
            ├── BFS 子图提取（max_hops=2）
            ├── 多跳路径推理
            └── 生成结构化上下文 → 附加到文档列表首位
```

### 降级策略

三重保护确保系统稳定性：
- `enable_kg=False` → 完全不触发 KG 路径（零开销）
- 图谱为空 → `analyze_kg_intent` 直接标记 `kg_intent=False`
- KG 检索异常/无结果 → 返回空上下文，不影响主流程

## LangSmith 评估

项目内置 8 维度自动化评估流水线，版本化目录管理（v1/v2）。

### 评估指标

| 指标 | 说明 |
|------|------|
| `correctness` | 答案与标准答案的事实一致性 |
| `faithfulness` | ★ 答案是否忠实于检索文档（反幻觉检测） |
| `answer_relevance` | 答案是否直接有效回应用户问题 |
| `completeness` | 答案是否完整覆盖问题要点 |
| `context_precision` | ★ 检索文档中真正有用的比例（去噪音） |
| `retrieval_relevance` | 检索文档与问题的语义相关性 |
| `answer_length` | 答案长度是否合理 |
| `latency` | 端到端响应延迟 |

### 目录结构

```
eval/
├── run_eval.py                  # 评估脚本（--version 参数化）
├── v1/                          # v1：3 篇文档，10 题（基础 RAG）
│   ├── sample_docs/
│   ├── dataset.jsonl
│   └── results/
└── v2/                          # v2：9 篇文档，40 题（含知识图谱）
    ├── sample_docs/
    ├── dataset.jsonl
    └── results/
```

### 运行评估

```bash
# v1 基础评估
uv run python eval/run_eval.py --version v1

# v2 评估 + 知识图谱
uv run python eval/run_eval.py --version v2 --enable-kg

# v2 评估 + 全策略
uv run python eval/run_eval.py --version v2 --enable-kg --enable-multi-query --enable-bm25
```

评估完成后，结果自动保存到 `eval/{version}/results/`（JSON + Markdown 报告），并同步上传至 LangSmith Dashboard。

### 新增评估版本

```bash
mkdir -p eval/v3/sample_docs eval/v3/results
# 1. 放入测试文档到 eval/v3/sample_docs/
# 2. 创建 eval/v3/dataset.jsonl（每行一个 {"question": "...", "answer": "..."}）
uv run python eval/run_eval.py --version v3 --enable-kg
```

## 前端功能

- **三种对话模式**：基础 RAG / Agent 自反思 / Agent 流式输出（SSE 逐字渲染）
- **9 个策略开关**：联网搜索、自反思、重排序、文档评估、查询重写、BM25 检索、HyDE 检索、Multi-Query 检索、知识图谱检索，实时切换
- **来源文档展示**：可展开查看，区分本地文档（绿色）、网页来源（蓝色可点击跳转）、知识图谱（橙色）
- **Agent 路径可视化**：彩色标签展示 Agent 执行的节点流转（含 KG 意图分析、KG 检索等新节点）
- **Sidebar 流程图**：15 节点 SVG 交互式状态图，运行节点实时高亮，显示当前激活节点
- **会话历史**：按会话陈列，点击加载历史消息，支持新建会话
- **知识库管理**：拖拽上传（PDF/MD/TXT），查看/删除已索引文档
- **实时健康监控**：顶部栏显示服务状态，30 秒自动刷新

## 项目结构

```
agentic-rag/
├── main.py                    # 项目入口
├── pyproject.toml             # Python 项目配置（UV 包管理）
├── .env                       # 环境变量（需自行创建）
├── src/                       # 后端源码
│   ├── main.py                # FastAPI 应用入口
│   ├── config/settings.py     # Pydantic Settings 配置
│   ├── api/                   # API 路由与依赖注入
│   │   ├── router.py          # 主路由聚合
│   │   ├── dependencies.py    # 依赖注入（Service/KG 单例）
│   │   ├── chat.py            # 对话 API
│   │   └── documents.py       # 文档管理 API
│   ├── models/                # Pydantic 请求/响应模型
│   │   ├── chat.py            # 对话模型（含 enable_kg 等 12 个开关）
│   │   ├── document.py        # 文档模型
│   │   └── common.py          # 通用模型
│   ├── agent/                 # LangGraph Agent
│   │   ├── state.py           # AgentState 定义（含 kg_intent/kg_context 等 25 字段）
│   │   ├── graph.py           # StateGraph 构建 + 条件路由 + Send fan-out/fan-in
│   │   ├── nodes.py           # 15 个核心节点实现
│   │   ├── prompts.py         # Prompt 模板（含 KG 意图分析、实体抽取等）
│   │   └── tools.py           # Tool Calling（DuckDuckGo 搜索）
│   ├── knowledge_graph/       # 知识图谱模块（Kuzu + numpy）
│   │   ├── __init__.py        # 单例工厂函数
│   │   ├── graph_store.py     # Kuzu 图数据库（原生持久化）
│   │   ├── graph_builder.py   # LLM 实体关系抽取 + 图谱构建
│   │   ├── graph_retriever.py # 实体链接（numpy 语义搜索）+ 子图提取 + 路径推理
│   │   └── kg_intent.py       # LLM 问题意图分析路由
│   ├── backend/               # AI 后端客户端
│   │   ├── llm.py             # ChatOpenAI 工厂（fast/strong/generic 三档）
│   │   ├── embedding.py       # DashScopeEmbeddings 工厂
│   │   └── reranker.py        # 百炼 TextReRank 重排序
│   ├── retrieval/             # 检索策略
│   │   └── bm25.py            # BM25 关键词检索（jieba 分词）
│   ├── pipeline/              # 文档处理管道
│   │   ├── loader.py          # 多格式加载器（PDF/MD/TXT）
│   │   ├── chunker.py         # 文本分块
│   │   └── indexer.py         # 文档索引器（含自动 KG 构建）
│   ├── store/                 # 数据存储
│   │   └── vector_store.py    # ChromaDB 封装（含文档元数据查询）
│   ├── memory/                # 对话记忆
│   │   └── manager.py         # 多会话隔离 + 滑动窗口
│   └── services/              # 业务服务层
│       ├── rag_service.py     # RAG 对话服务（含流式处理、KG 字段初始化）
│       └── document_service.py # 文档管理服务
├── eval/                      # LangSmith 评估
│   ├── run_eval.py            # 评估脚本（支持 --enable-kg 等参数）
│   ├── v1/                    # v1：基础 RAG 评估
│   │   ├── sample_docs/       # 3 篇测试文档
│   │   ├── dataset.jsonl      # 10 题测试集
│   │   └── results/           # 评估结果
│   └── v2/                    # v2：含知识图谱评估
│       ├── sample_docs/       # 9 篇测试文档
│       ├── dataset.jsonl      # 40 题测试集
│       └── results/           # 评估结果
├── front/                     # 前端源码
│   ├── vite.config.ts         # Vite 配置（API 代理到 8000）
│   └── src/
│       ├── main.ts            # Vue 应用入口
│       ├── App.vue            # 根组件
│       ├── router/index.ts    # 路由：/ (对话) + /documents (知识库)
│       ├── types/index.ts     # TypeScript 类型定义
│       ├── api/               # API 调用封装
│       │   ├── client.ts      # fetch 通用封装
│       │   ├── chat.ts        # 对话 API（含 SSE 流式）
│       │   ├── documents.ts   # 文档 API
│       │   └── health.ts      # 健康检查 API
│       ├── composables/       # Vue 组合式函数
│       │   ├── useChat.ts     # 对话状态（含 enable_kg 等 9 个策略开关参数）
│       │   ├── agentFlowState.ts # Agent 流程状态（9 个策略开关 ref）
│       │   ├── useDocuments.ts # 文档管理
│       │   └── useHealth.ts   # 健康状态轮询
│       ├── views/             # 页面视图
│       │   ├── ChatView.vue   # 对话页
│       │   └── DocumentsView.vue # 知识库页
│       └── components/        # 组件
│           ├── layout/        # AppLayout + Sidebar（15 节点 SVG 流程图）
│           ├── common/        # HealthBar
│           ├── chat/          # ChatPanel, ChatInput, MessageBubble, SourcePanel, AgentPathBadge, SessionHistory
│           └── documents/     # DocumentUpload, DocumentList
├── data/                      # 本地数据存储（自动生成）
│   ├── chroma/                # ChromaDB 持久化数据
│   ├── kg/                    # Kuzu 图数据库 + 实体向量索引 .npz
│   └── temp_uploads/          # 大文件临时缓存
└── chroma_data/               # （旧，已迁移到 data/chroma）
```

## 依赖

### 后端 (Python 3.12.13+)

| 包 | 用途 |
|------|------|
| `fastapi` + `uvicorn` | Web 框架与服务器 |
| `langgraph` | Agent 状态图编排 + Send fan-out/fan-in |
| `langchain` + `langchain-openai` + `langchain-chroma` + `langchain-community` | RAG 组件链 |
| `chromadb` | 向量数据库（本地持久化，文档元数据一同存储） |
| `kuzu` | 图数据库（知识图谱存储，原生持久化） |
| `numpy` | 实体向量索引（.npz 二进制文件替代原 FAISS+SQLite） |
| `dashscope` | 百炼 LLM / Embedding / Rerank SDK |
| `langsmith` | LLM 追踪与评估 |
| `ddgs` | DuckDuckGo 网页搜索（联网搜索降级） |
| `pdfminer-six` + `python-docx` + `mistune` | 文档解析（PDF/DOCX/MD/TXT/CSV） |
| `jieba` | 中文分词（BM25 检索） |
| `rank-bm25` | BM25 关键词检索算法 |
| `pydantic-settings` | 配置管理 |
| `sse-starlette` | SSE 流式输出 |
| `python-multipart` | 文件上传支持 |
| `tenacity` | LLM 调用指数退避重试 |
| `tiktoken` | Token 计数与分块 |
| `python-dotenv` | 环境变量加载 |
| `mistune` | Markdown 解析 |

### 前端 (Node.js)

| 包 | 用途 |
|------|------|
| `vue` 3.5 + `vue-router` 4 | 前端框架 |
| `vite` 5 + `@vitejs/plugin-vue` | 构建工具 |
| `tailwindcss` 3.4 | CSS 框架 |
| `lucide-vue-next` | 图标库 |
| `typescript` | 类型检查 |
