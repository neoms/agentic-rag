# Agentic RAG

基于 LangGraph 构建的智能检索增强生成（Agentic RAG）系统。支持多策略检索、重排序精排、自反思评估、联网搜索降级、工具调用、流式输出等高级能力。前后端分离架构，后端 FastAPI + 前端 Vue 3。

## 架构特点

- **多策略检索**：语义检索 + MMR 多样性检索，混合合并去重
- **重排序精排**：百炼 `gte-rerank` 模型对检索结果二次排序，提升 Top-K 文档质量
- **Agent 智能体**：自反思能力（文档评估 → 查询重写 → 幻觉检测）；支持 Tool Calling（联网搜索、计算器）
- **联网搜索降级**：向量库无匹配时自动走 DuckDuckGo 网页搜索，结果带来源 URL
- **LangSmith 评估**：8 维度自动化评估流水线（正确性/忠实度/相关性/完整性/上下文精度/延迟等）
- **百炼平台统一接入**：LLM 使用 OpenAI 兼容协议，Embedding 使用官方 DashScope SDK
- **文档分块**：`RecursiveCharacterTextSplitter`，chunk_size=500、chunk_overlap=100
- **FastAPI + SSE 流式输出**：自动生成 Swagger 文档
- **ChromaDB 本地持久化**：零外部依赖，数据保存在 `chroma_data/` 目录
- **Vue 3 前端**：Vite + TypeScript + TailwindCSS，支持三种对话模式、会话历史、拖拽上传

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
| `CHROMA_PERSIST_DIR` | ChromaDB 数据目录 | `chroma_data` |
| `RETRIEVAL_TOP_K` | 检索候选数 | `20` |
| `RERANK_ENABLED` | 是否启用重排序 | `true` |
| `RERANK_MODEL` | 重排序模型 | `gte-rerank` |
| `RERANK_TOP_K` | 重排序后保留数 | `5` |
| `MEMORY_WINDOW_SIZE` | 对话记忆窗口 | `20` |
| `MAX_UPLOAD_SIZE_MB` | 上传文件大小限制 | `10` |

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

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查（服务状态、LLM/Embedding 型号、文档数） |
| POST | `/api/v1/documents/upload` | 上传文档（PDF/MD/TXT，最大 10MB） |
| GET | `/api/v1/documents` | 列出已索引文档 |
| DELETE | `/api/v1/documents/{doc_id}` | 删除文档及其向量块 |
| POST | `/api/v1/chat/simple` | 基础 RAG 对话 |
| POST | `/api/v1/chat/agentic` | Agent 模式对话（自反思 + 工具调用） |
| POST | `/api/v1/chat/stream` | Agent 流式对话（SSE，事件类型：source/path/token/done/error） |
| GET | `/api/v1/chat/history/{session_id}` | 获取会话历史 |

## LangGraph Agent 状态图

```
START → retrieve → rerank_documents → grade_documents
         ↑                               ├── [RELEVANT] → generate → check_hallucination → END
         │                               └── [IRRELEVANT] →
         │                                   ├── enable_web_search → web_search → generate → ...
         │                                   └── !enable_web_search → transform_query ──┘
         └────────────────────────── (最多循环 max_iterations 次)
```

核心节点：

| 节点 | 功能 | 模型 |
|------|------|------|
| `retrieve` | 语义检索 + MMR 多样性检索，合并去重 | Embedding |
| `rerank_documents` | 百炼 TextReRank 对合并文档做二次精排 | `gte-rerank` |
| `grade_documents` | 严格评估检索文档是否能有效回答用户问题 | qwen-turbo |
| `web_search` | 向量库无匹配时，通过 DuckDuckGo 搜索网页作为降级方案 | 无（HTTP 调用） |
| `transform_query` | 不相关时自动重写查询 | qwen-turbo |
| `generate` | 基于检索文档或网页搜索结果生成回答 | qwen-max |
| `check_hallucination` | 检测答案是否与文档一致，不一致则重试 | qwen-turbo |

### 检索流程详述

1. **语义检索**：`similarity_search_with_relevance_scores`，按阈值 0.5 过滤
2. **MMR 多样性检索**：`max_marginal_relevance_search`，lambda_mult=0.7
3. **合并去重**：按 `page_content` 去重
4. **重排序**：百炼 `gte-rerank` 模型对合并结果做精排，保留 Top K 文档，按 relevance_score 降序

### 联网搜索流程

1. 用户在 Agent/流式模式下开启"联网搜索"
2. `retrieve` → `rerank_documents` → 从向量库检索并精排
3. `grade_documents` 评估：文档是否包含能直接回答问题的关键信息？
4. 如果 **不相关** 且开启了联网搜索 → `web_search` 节点调用 DuckDuckGo 搜索
5. 搜索结果转为 Document（带 URL metadata），直接进入 `generate` 生成回答
6. 前端 SourcePanel 以蓝色卡片展示网页来源，含可点击的 URL 链接

## LangSmith 评估

项目内置 8 维度自动化评估流水线，可对 RAG 系统做全方位质量评测。

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

### 运行评估

```bash
# 运行 v1 版本评估
uv run python eval/run_eval.py --version v1
```

评估完成后，结果自动保存到 `eval/v1/results/`（JSON + Markdown 报告）。

### 新增评估版本

```bash
mkdir -p eval/v2/sample_docs eval/v2/results
# 1. 放入测试文档到 eval/v2/sample_docs/
# 2. 创建 eval/v2/dataset.jsonl（每行一个 {"question": "...", "answer": "..."}）
uv run python eval/run_eval.py --version v2
```

评估器使用 qwen-turbo 作为 LLM 评判者，所有结果同步上传至 LangSmith Dashboard 可在线查看。

## 前端功能

- **三种对话模式**：基础 RAG / Agent 自反思 / Agent 流式输出（SSE 逐字渲染）
- **Agent 选项**：联网搜索开关（开启时弹出提示气泡）、自反思开关
- **来源文档展示**：可展开查看，区分本地文档（绿色）与网页来源（蓝色，可点击跳转）
- **Agent 路径可视化**：彩色标签展示 Agent 执行的节点流转
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
│   │   ├── dependencies.py    # 依赖注入（Service 单例）
│   │   ├── chat.py            # 对话 API
│   │   └── documents.py       # 文档管理 API
│   ├── models/                # Pydantic 请求/响应模型
│   │   ├── chat.py            # 对话模型
│   │   ├── document.py        # 文档模型
│   │   └── common.py          # 通用模型
│   ├── agent/                 # LangGraph Agent
│   │   ├── state.py           # AgentState 定义
│   │   ├── graph.py           # StateGraph 构建 + 路由逻辑
│   │   ├── nodes.py           # 核心节点（retrieve/rerank/grade/generate/check...）
│   │   ├── prompts.py         # Prompt 模板
│   │   └── tools.py           # Tool Calling（计算器 + DuckDuckGo 搜索）
│   ├── backend/               # AI 后端客户端
│   │   ├── llm.py             # ChatOpenAI 工厂（fast/strong/generic 三档）
│   │   ├── embedding.py       # DashScopeEmbeddings 工厂
│   │   └── reranker.py        # 百炼 TextReRank 重排序
│   ├── pipeline/              # 文档处理管道
│   │   ├── loader.py          # 多格式加载器（PDF/MD/TXT）
│   │   ├── chunker.py         # 文本分块
│   │   └── indexer.py         # 文档索引器
│   ├── store/                 # 向量存储
│   │   └── vector_store.py    # ChromaDB 封装
│   ├── memory/                # 对话记忆
│   │   └── manager.py         # 多会话隔离 + 滑动窗口
│   └── services/              # 业务服务层
│       ├── rag_service.py     # RAG 对话服务（含流式处理）
│       └── document_service.py # 文档管理服务
├── eval/                      # LangSmith 评估
│   ├── run_eval.py            # 评估脚本（--version 参数化）
│   └── v1/                    # 版本目录（可扩展 v2, v3...）
│       ├── sample_docs/       # 测试文档
│       ├── dataset.jsonl      # 测试数据集（Q&A）
│       └── results/           # 评估结果（JSON + Markdown）
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
│       │   ├── useChat.ts     # 对话状态管理（含流式 SSE 解析 + 会话管理）
│       │   ├── useDocuments.ts # 文档管理
│       │   └── useHealth.ts   # 健康状态轮询
│       ├── views/             # 页面视图
│       │   ├── ChatView.vue   # 对话页
│       │   └── DocumentsView.vue # 知识库页
│       └── components/        # 组件
│           ├── layout/        # AppLayout + Sidebar
│           ├── common/        # HealthBar
│           ├── chat/          # ChatPanel, ChatInput, MessageBubble, SourcePanel, AgentPathBadge, SessionHistory
│           └── documents/     # DocumentUpload, DocumentList
└── chroma_data/               # ChromaDB 持久化数据（自动生成）
```

## 依赖

### 后端 (Python 3.12.13+)

| 包 | 用途 |
|------|------|
| `fastapi` + `uvicorn` | Web 框架与服务器 |
| `langgraph` | Agent 状态图编排 |
| `langchain` + `langchain-openai` + `langchain-chroma` + `langchain-community` | RAG 组件链 |
| `chromadb` | 向量数据库（本地持久化） |
| `dashscope` | 百炼 LLM / Embedding / Rerank SDK |
| `langsmith` | LLM 追踪与评估 |
| `ddgs` | DuckDuckGo 网页搜索（联网搜索降级） |
| `pypdf2` + `markdown` | 文档解析 |
| `pydantic-settings` | 配置管理 |
| `sse-starlette` | SSE 流式输出 |
| `python-multipart` | 文件上传支持 |

### 前端 (Node.js)

| 包 | 用途 |
|------|------|
| `vue` 3.5 + `vue-router` 4 | 前端框架 |
| `vite` 5 + `@vitejs/plugin-vue` | 构建工具 |
| `tailwindcss` 3.4 | CSS 框架 |
| `lucide-vue-next` | 图标库 |
| `typescript` | 类型检查 |
