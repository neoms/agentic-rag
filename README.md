# Agentic RAG

基于 LangGraph 构建的智能检索增强生成（Agentic RAG）系统。支持多策略检索、自反思评估、工具调用、流式输出等高级能力。

## 架构特点

- **多策略检索**：语义检索、关键字检索、混合检索、查询重写、结果重排序，由 LangGraph 状态图编排流转
- **Agent 智能体**：自反思能力，评估检索质量并自动决定是否需要重新检索或优化查询；支持 Tool Calling（联网搜索、计算器等）
- **百炼平台统一接入**：LLM 和 Embedding 均使用阿里云百炼（OpenAI 兼容协议）
- **文档分块**：`RecursiveCharacterTextSplitter`，chunk_size=500、chunk_overlap=100
- **FastAPI + SSE 流式输出**：自动生成 Swagger 文档
- **ChromaDB 本地持久化**：零外部依赖，数据自动保存在 `chroma_data/` 目录

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

### 2. 本地运行

```bash
# 安装依赖
uv sync

# 启动应用（ChromaDB 自动使用本地持久化，无需额外服务）
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问 http://localhost:8000/docs 查看 Swagger 文档。向量数据自动保存在项目根目录 `chroma_data/` 中。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/documents/upload` | 上传文档（PDF/MD/TXT） |
| GET | `/api/v1/documents` | 列出已索引文档 |
| DELETE | `/api/v1/documents/{doc_id}` | 删除文档 |
| POST | `/api/v1/chat/simple` | 基础 RAG 对话 |
| POST | `/api/v1/chat/agentic` | Agent 模式对话（自反思） |
| POST | `/api/v1/chat/stream` | Agent 模式流式对话（SSE） |
| GET | `/api/v1/chat/history/{session_id}` | 获取会话历史 |
| GET | `/health` | 健康检查 |

## LangGraph Agent 状态图

```
START → retrieve → grade_documents
         ↑            ├── [RELEVANT] → generate → check_hallucination → END
         │            └── [IRRELEVANT] → transform_query ──┘
         └────────── (循环, max_iterations 次)
```

## 项目结构

```
src/
├── main.py              # FastAPI 应用入口
├── config/settings.py   # Pydantic Settings 配置
├── api/                 # API 路由与依赖注入
├── models/              # Pydantic 请求/响应模型
├── backend/             # 百炼 LLM & Embedding 客户端
├── pipeline/            # 文档加载、分块、索引管道
├── store/               # ChromaDB 向量存储（本地持久化）
├── agent/               # LangGraph Agent（state/nodes/graph/tools/prompts）
├── memory/              # 对话记忆管理
└── services/            # 业务服务层
```
