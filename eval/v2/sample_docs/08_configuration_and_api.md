# 配置参考与 API 文档

## 1. 系统配置说明

项目通过 pydantic-settings 管理配置，支持 .env 文件和环境变量。

### 1.1 LLM 配置

- **llm_model**: 大语言模型名称，默认 "qwen-plus"
- **llm_base_url**: API 地址，默认 "https://dashscope.aliyuncs.com/compatible-mode/v1"
- **llm_api_key**: API 密钥，必须通过环境变量 DASHSCOPE_API_KEY 设置
- **llm_temperature**: 生成温度参数，默认 0.0（确定性输出）

### 1.2 Embedding 配置

- **embedding_model**: 嵌入模型名称，默认 "text-embedding-v3"
- **embedding_base_url**: 复用 LLM 的 base_url
- **embedding_api_key**: 复用 LLM 的 api_key

### 1.3 分块配置

- **chunk_size**: 文档分块大小，默认 500 字符
- **chunk_overlap**: 块间重叠大小，默认 100 字符

### 1.4 检索配置

- **retrieval_top_k**: 检索返回的最大文档数，默认 8
- **retrieval_similarity_threshold**: 相似度过滤阈值，默认 0.3

### 1.5 知识图谱配置

- **kg_data_dir**: 图谱数据存储目录，默认 "kg_data"
- **kg_max_hops**: 子图提取最大跳数，默认 2
- **kg_max_entities**: 单次查询最多提取实体数，默认 10

### 1.6 服务配置

- **app_host**: 服务绑定地址，默认 "0.0.0.0"
- **app_port**: 服务端口，默认 8000
- **chroma_persist_dir**: ChromaDB 持久化路径，默认 "chroma_data"

## 2. API 接口文档

### 2.1 文档上传

**接口**: `POST /api/documents/upload`

上传文档到知识库。支持 PDF、Markdown 和 TXT 格式。

请求：multipart/form-data，file 字段包含文档文件。

响应示例：
```json
{
  "doc_id": "uuid-string",
  "filename": "document.md",
  "chunk_count": 15
}
```

### 2.2 文档列表

**接口**: `GET /api/documents`

列出所有已入库的文档及分块统计。

### 2.3 文档删除

**接口**: `DELETE /api/documents/{doc_id}`

根据文档ID删除文档及其所有分块。

### 2.4 对话接口（流式 Agent 模式）

**接口**: `POST /api/v1/chat/stream`

使用完整的 Agent 工作流处理对话，SSE 流式输出。

请求体参数：
- **query** (必填): 用户问题
- **session_id** (可选): 会话ID，用于多轮对话
- **enable_web_search** (可选): 启用联网搜索，默认 false
- **enable_reflection** (可选): 启用自反思检测，默认 true
- **enable_rerank** (可选): 启用重排序，默认 true
- **enable_grade_documents** (可选): 启用文档评估，默认 true
- **enable_transform_query** (可选): 启用查询重写，默认 true
- **enable_bm25** (可选): 启用BM25检索，默认 false
- **enable_hyde** (可选): 启用HyDE检索，默认 false
- **enable_multi_query** (可选): 启用多角度检索，默认 false
- **enable_kg** (可选): 启用知识图谱检索，默认 false

### 2.5 知识图谱统计

**接口**: `GET /api/knowledge-graph/stats`

返回知识图谱的统计信息，包括节点数、边数和主要实体类型分布。

## 3. 启动方式

```bash
# 安装依赖
uv sync

# 启动服务
uv run python main.py

# 或直接
python main.py
```

服务默认运行在 `http://0.0.0.0:8000`，前端界面通过浏览器访问。

## 4. Agent 工作流配置

### 4.1 开关组合策略

项目通过多个布尔开关控制 Agent 工作流的行为：

- **关闭所有增强策略**：只执行基础语义检索 + 生成，速度最快
- **开启所有策略**：语义检索 + BM25 + HyDE + Multi-Query + KG，覆盖面最广
- **只开 KG**：语义检索 + 知识图谱，适合结构化知识查询
- **只开关键词**：语义检索 + BM25，适合精确术语搜索

### 4.2 流式输出模式

流式模式下，系统通过 astream_events 实时推送每个 Agent 节点的执行状态，前端在 SVG 流程图中实时高亮当前正在执行的节点，提供可视化的 Agent 执行过程。

每个节点的开始和完成事件通过 SSE 推送给前端，节点名称包含在上述 API 的 agent_path 字段中。
