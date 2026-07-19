# Agentic RAG

基于 LangGraph 构建的智能检索增强生成（Agentic RAG）系统。支持多策略检索、自反思评估、工具调用、流式输出等高级能力。前后端分离架构，后端 FastAPI + 前端 Vue 3。

## 架构特点

- **多策略检索**：语义检索、关键字检索、混合检索、查询重写、结果重排序，由 LangGraph 状态图编排流转
- **Agent 智能体**：自反思能力，评估检索质量并自动决定是否需要重新检索或优化查询；支持 Tool Calling（联网搜索、计算器等）
- **百炼平台统一接入**：LLM 和 Embedding 均使用阿里云百炼（OpenAI 兼容协议）
- **文档分块**：`RecursiveCharacterTextSplitter`，chunk_size=500、chunk_overlap=100
- **FastAPI + SSE 流式输出**：自动生成 Swagger 文档
- **ChromaDB 本地持久化**：零外部依赖，数据自动保存在 `chroma_data/` 目录
- **Vue 3 前端**：Vite + TypeScript + TailwindCSS，支持三种对话模式、拖拽上传、Agent 路径可视化

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
| `EMBEDDING_MODEL` | 嵌入模型 | `text-embedding-v2` |
| `CHROMA_PERSIST_DIR` | ChromaDB 数据目录 | `chroma_data` |
| `RETRIEVAL_TOP_K` | 检索结果数 | `5` |
| `MEMORY_WINDOW_SIZE` | 对话记忆窗口 | `5` |
| `MAX_UPLOAD_SIZE_MB` | 上传文件大小限制 | `10` |

### 2. 启动后端

```bash
# 安装依赖
uv sync

# 启动应用
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# 或者直接
python main.py
```

启动后访问 http://localhost:8000/docs 查看 Swagger 文档。向量数据自动保存在项目根目录 `chroma_data/` 中。

### 3. 启动前端

```bash
cd front
npm install
npm run dev
```

前端运行在 http://localhost:3000，API 请求自动代理到后端 `localhost:8000`。

构建生产包：

```bash
cd front
npm run build    # 输出到 front/dist/
```

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
START → retrieve → grade_documents
         ↑            ├── [RELEVANT] → generate → check_hallucination
         │            │                               ├── [PASSED] → END
         │            │                               └── [FAILED] → generate (重试)
         │            └── [IRRELEVANT] → transform_query ──┘
         └────────── (最多循环 max_iterations 次)
```

核心节点：

| 节点 | 功能 | 模型 |
|------|------|------|
| `retrieve` | 语义检索 + MMR 多样性检索，合并去重 | Embedding |
| `grade_documents` | 评估检索文档相关性 | qwen-turbo（快速） |
| `transform_query` | 不相关时自动重写查询 | qwen-turbo（快速） |
| `generate` | 基于检索文档生成回答 | qwen-max（强模型） |
| `check_hallucination` | 检测答案是否与文档一致，不一致则重试 | qwen-turbo（快速） |

## 前端功能

- **三种对话模式**：基础 RAG / Agent 自反思 / Agent 流式输出（SSE）
- **Agent 选项**：联网搜索开关、自反思开关（Agent 模式下可配置）
- **来源文档展示**：可展开查看检索到的文档片段及相似度分数
- **Agent 路径可视化**：彩色标签展示 Agent 执行的节点流转和反思轮数
- **知识库管理**：拖拽上传（PDF/MD/TXT），查看/删除已索引文档
- **会话历史**：侧栏展示当前会话的消息历史
- **实时健康监控**：顶部栏显示服务状态，30 秒自动刷新

## 项目结构

```
agentic-rag/
├── main.py                    # 项目入口（等同于 uvicorn src.main:app）
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
│   │   ├── graph.py           # StateGraph 构建
│   │   ├── nodes.py           # 核心节点实现
│   │   ├── prompts.py         # Prompt 模板
│   │   └── tools.py           # Tool Calling 工具
│   ├── backend/               # AI 后端客户端
│   │   ├── llm.py             # ChatOpenAI 工厂
│   │   └── embedding.py       # OpenAIEmbeddings 工厂
│   ├── pipeline/              # 文档处理管道
│   │   ├── loader.py          # 多格式加载器
│   │   ├── chunker.py         # 文本分块
│   │   └── indexer.py         # 文档索引器
│   ├── store/                 # 向量存储
│   │   └── vector_store.py    # ChromaDB 封装
│   ├── memory/                # 对话记忆
│   │   └── manager.py         # 多会话隔离 + 滑动窗口
│   └── services/              # 业务服务层
│       ├── rag_service.py     # RAG 对话服务
│       └── document_service.py # 文档管理服务
├── front/                     # 前端源码
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts         # Vite 配置（API 代理到 8000）
│   ├── tailwind.config.js
│   └── src/
│       ├── main.ts            # Vue 应用入口
│       ├── App.vue            # 根组件
│       ├── router/index.ts    # 路由：/ (对话) + /documents (知识库)
│       ├── types/index.ts     # TypeScript 类型定义
│       ├── api/               # API 调用封装
│       │   ├── client.ts      # fetch 通用封装
│       │   ├── chat.ts        # 对话 API
│       │   ├── documents.ts   # 文档 API
│       │   └── health.ts      # 健康检查 API
│       ├── composables/       # Vue 组合式函数
│       │   ├── useChat.ts     # 对话状态管理
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
| `langchain` + `langchain-openai` + `langchain-chroma` | RAG 组件链 |
| `chromadb` | 向量数据库（本地持久化） |
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
