# Agentic RAG

基于 LangGraph 构建的智能检索增强生成（Agentic RAG）系统。支持多策略并行检索、知识图谱推理、重排序精排、自反思评估、联网搜索降级、工具调用、流式输出等高级能力。前后端分离架构，后端 FastAPI + 前端 Vue 3。

## 架构特点

- **多策略并行检索**：语义检索 + MMR 多样性、BM25 关键词、Multi-Query 多角度查询、知识图谱检索，在 `parallel_retrieve_merge` 节点内用线程池并行执行，可独立开关控制
- **知识图谱模块**：Kuzu 图数据库（原生持久化）+ numpy .npz 二进制向量索引，LLM 自动抽取实体关系构建图谱，支持意图分析路由、实体链接、子图提取、多跳路径推理，不适用时平滑降级到原有 RAG 流程
- **重排序精排**：LLM Cross-Encoder 对检索结果二次排序，提升 Top-K 文档质量
- **Agent 智能体**：自反思能力（文档评估 → 查询重写 → 幻觉检测）；支持 Tool Calling（联网搜索）
- **联网搜索降级**：向量库无匹配时自动走 DuckDuckGo 网页搜索，结果带来源 URL
- **多级缓存（精准 + 语义）**：请求先进缓存，精准缓存按规范化问题 + 策略签名精确匹配，语义缓存按问题向量余弦相似度（默认 ≥0.92）命中；都不命中才调用 LLM，命中时回放存储的答案/来源/路径，零 LLM 调用。语义缓存阶段计算的问题向量复用于检索，不重复调用 Embedding
- **可观测性**：`/metrics` Prometheus 指标（QPS/缓存命中/LLM 调用/耗时直方图）、JSON 结构化滚动日志（`log/app.log`，10MB × 5）、组件化 `/health` 健康检查（`?deep=true` 可做 Embedding 探针）
- **标准评估体系**：RAGAS 标准质量指标（faithfulness/answer_relevancy/factual_correctness/context_precision/context_recall，中英双语显示）+ 性能/成本指标 + Langfuse Cloud 生产追踪与在线采样打分 + Locust 压测；发布门禁（`--gate`）
- **百炼平台统一接入**：LLM 使用 OpenAI 兼容协议，Embedding 使用官方 DashScope SDK
- **文档分块**：`RecursiveCharacterTextSplitter`，chunk_size=500、chunk_overlap=100；每个块带文档标题前缀（`CHUNK_TITLE_CONTEXT`，默认开启），补全文档级上下文，改善重排与幻觉检测的溯源能力
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
| `LLM_MODEL` | 默认 LLM 模型 | `qwen-plus` |
| `LLM_MODEL_FAST` | 快速评估模型（检索评估/复杂度判定/幻觉检测等轻量任务） | `qwen-turbo` |
| `LLM_MODEL_STRONG` | 强生成模型（最终答案生成） | `qwen-max` |
| `EMBEDDING_MODEL` | 嵌入模型 | `text-embedding-v4` |
| `RERANK_MODEL` | 重排序模型（`RERANK_ENABLED=true` 时必填） | `gte-rerank-v2` |

> 模型名只在 `.env` 中配置（无内置默认值，启动校验会拒绝空值）；Docker 部署时 compose 提供示例默认值，可在宿主 `.env` 中覆盖。

可选环境变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `CHROMA_PERSIST_DIR` | ChromaDB 数据目录 | `data/chroma` |
| `RETRIEVAL_TOP_K` | 检索候选数 | `12` |
| `RETRIEVAL_SIMILARITY_THRESHOLD` | 语义检索过滤阈值 | `0.5` |
| `RERANK_ENABLED` | 是否启用重排序 | `true` |
| `RERANK_TOP_K` | 重排序后保留数 | `5` |
| `GRADE_SCORE_IRRELEVANT_MAX` | 文档评估负判定：全体文档最高分 ≤ 此值直接不相关（0 LLM） | `0.25` |
| `GRADE_SCORE_IRRELEVANT_HARD_MIN` | 文档评估负判定收紧：仅当最高分 ≤ 此值才直接不相关（0 LLM），模糊低分区走 LLM 兜底 | `0.10` |
| `GRADE_SCORE_RELEVANT_MIN` | 文档评估正判定：top1 分数下限 | `0.70` |
| `GRADE_SCORE_RELEVANT_GAP` | 文档评估正判定：top1 与 top2 最小分差（断层检测） | `0.10` |
| `TRANSFORM_LOOP_IMPROVE_MIN` | 查询重写循环止损：重写后 top1 重排分提升不足此值则停止重写 | `0.02` |
| `MEMORY_WINDOW_SIZE` | 对话记忆窗口 | `20` |
| `MAX_UPLOAD_SIZE_MB` | 上传文件大小限制 | `10` |
| `KG_DATA_DIR` | 知识图谱数据目录 | `data/kg` |
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
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Langfuse Cloud 追踪与评估（成对配置，不配则降级） | 空 |
| `LANGFUSE_HOST` | Langfuse 服务地址 | `https://cloud.langfuse.com` |
| `EVAL_JUDGE_MODEL` | LLM-as-judge 评判模型（独立强模型，留空回退 `LLM_MODEL_STRONG`） | 空 |
| `EVAL_JUDGE_BASE_URL` / `EVAL_JUDGE_API_KEY` | 评判模型独立接入（可选，默认同 LLM 配置） | 空 |
| `EVAL_JUDGE_EXTRA_BODY` | judge 额外请求体（JSON；思考模型与 RAGAS n=3 冲突时填 `{"enable_thinking": false}`） | 空 |
| `GENERATION_EXTRA_BODY` | 流式生成额外请求体（JSON；思考模型耗尽 token 预算导致空答案时填 `{"enable_thinking": false}`） | 空 |
| `EVAL_GATE_THRESHOLDS` | 发布门禁阈值（JSON，如 `{"faithfulness": 0.85}`） | `{}` |
| `EVAL_SAMPLE_RATE` | 在线评估采样比例 `(0,1]` | `0.1` |
| `EVAL_STUB_LLM` | 压测/评估 stub 模式（不调用真实 LLM） | `false` |
| `LLM_PRICE_INPUT_PER_1M` / `LLM_PRICE_OUTPUT_PER_1M` | LLM 成本估算单价（元/百万 token，0=不计） | `0` |

更多配置项（分块参数、重试、上传阈值、内容校验、模型温度等）见 `src/config/settings.py`；
启动时会自动校验关键配置，错误会逐项给出当前值/原因/修改位置/修复方式（见 `src/config/validation.py`）。

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

### 4. 运行测试

```bash
# 后端测试（47 个用例，数据隔离在临时目录，无需外网）
uv run pytest

# 前端构建检查
cd front && npm run build
```

### 5. Docker 部署（推荐）

```bash
# 1. 准备环境变量（至少填写 DASHSCOPE_API_KEY）
cp .env.example .env
# 编辑 .env 填入 API Key 与模型配置

# 2. 构建并启动（前端 nginx 反代后端，访问 http://localhost:8080）
docker compose up -d --build
```

> 启动时会自动校验关键配置：配置缺失/越界/不自洽会拒绝启动，并在日志中逐项说明
> 当前值、原因、修改位置和正确的写法（见 `src/config/validation.py`）。

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
| POST | `/api/v1/documents/upload` | 上传文档（PDF/MD/TXT/DOCX/CSV，最大 10MB；流式落盘 + 队列上限，超限 413/429） |
| GET | `/api/v1/documents` | 列出已索引文档 |
| DELETE | `/api/v1/documents/{doc_id}` | 删除文档及其向量块 |
| GET | `/api/v1/documents/tasks` | 列出全部上传索引任务 |
| GET | `/api/v1/documents/tasks/{task_id}` | 查询单个任务状态 |
| POST | `/api/v1/chat/stream` | Agent 流式对话（SSE，事件类型：source/path/token/done/error/hallucination等） |
| GET | `/api/v1/chat/sessions` | 获取全部会话摘要（侧边栏列表） |
| GET | `/api/v1/chat/history/{session_id}` | 获取会话历史 |
| DELETE | `/api/v1/chat/history/{session_id}` | 删除会话历史（含持久化数据） |
| POST | `/api/v1/chat/feedback` | 提交用户反馈（👍/👎 写回 Langfuse；未配置返回 503） |

### Agentic 请求体参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `query` | string | 用户问题 | 必填 |
| `session_id` | string | 会话 ID（历史服务端持久化） | 自动生成 |
| `use_cache` | bool | 是否启用多级缓存（评估/调试可绕过） | `true` |
| `enable_web_search` | bool | 联网搜索 | `false` |
| `enable_reflection` | bool | 自反思/幻觉检测 | `true` |
| `enable_rerank` | bool | 重排序 | `true` |
| `enable_transform_query` | bool | 查询重写（默认关闭；手动开启时最多重写 1 次） | `false` |
| `enable_bm25` | bool | BM25 关键词检索 | `true` |
| `enable_multi_query` | bool | Multi-Query 多角度检索 | `false` |
| `enable_kg` | bool | 知识图谱检索 | `true` |

## LangGraph Agent 状态图

图内节点 10 个，配合服务层 3 个缓存虚拟节点与图外幻觉检测。完整执行路径：

```
START
  │
  ▼
cache_lookup（虚拟节点：精准 → 语义，均未命中才进入下图）
  │ 命中 ──────────────► cache_replay（虚拟节点：回放答案/来源/路径，零 LLM 调用）
  ▼ 未命中
analyze_kg_intent（KG 意图分析，决定是否走 KG 路径）
  │
  ▼
parallel_retrieve_merge（语义+MMR；线程池并行 BM25 / Multi-Query / KG 并合并去重）
  │
  ├─ rerank_documents ──► grade_documents（仅当查询重写或联网搜索任一开启时运行；两者都关则跳过 grade 直达 judge）
  │                        ├─ [相关] ────────────► judge_complexity
  │                        ├─ [不相关 + 联网] ────► web_search ──► judge_complexity
  │                        └─ [不相关] ──► transform_query ──► retrieve ──► rerank_documents（默认关闭，最多 1 轮）
  │
  ▼
judge_complexity
  ├─ SIMPLE  ──► generate_simple（LLM_MODEL_FAST 流式）
  └─ COMPLEX ──► generate_complex（LLM_MODEL_STRONG 流式）
  │
  ▼
check_hallucination（图外执行：自反思开启时检测忠实度；失败不重试）
  │
  ▼
cache_store（虚拟节点：反射通过或关闭且答案非空时写回缓存）
  │
  ▼
END
```

### 节点说明

| 节点 | 功能 | 模型 |
|------|------|------|
| `analyze_kg_intent` | LLM 分析问题是否需要知识图谱（实体关系/多跳推理 → KG；定义/教程 → 降级） | LLM_MODEL_FAST |
| `parallel_retrieve_merge` | 语义+MMR 检索 + 线程池并行 BM25/Multi-Query/KG，合并去重；语义缓存阶段的问题向量直接复用 | Embedding + jieba + Kuzu |
| `retrieve` | 查询重写循环内的语义+MMR 检索 | Embedding |
| `rerank_documents` | 百炼 TextReRank 二次精排（接口异常时降级为原始排序并在图中标注） | RERANK_MODEL |
| `grade_documents` | 文档相关性评估（含关键词快速路径，可 0 LLM） | LLM_MODEL_FAST |
| `web_search` | 向量库无匹配时，DuckDuckGo 搜索网页作为降级方案 | HTTP |
| `transform_query` | 文档不相关时自动重写查询（默认关闭，手动开启时最多 1 轮） | LLM_MODEL_FAST |
| `judge_complexity` | 复杂度判定（规则快速路径 + LLM 兜底） | LLM_MODEL_FAST |
| `generate_simple` / `generate_complex` | 流式生成最终答案（简单/复杂分别用快速/强模型） | LLM_MODEL_FAST / LLM_MODEL_STRONG |
| `check_hallucination` | 图外幻觉检测：输出忠实度评分；失败不重试、答案不写缓存 | LLM_MODEL_FAST |
| `cache_lookup` / `cache_replay` / `cache_store` | 缓存虚拟节点（服务层事件驱动，非 LangGraph 节点）：查询/回放/写回 | — |

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

## 标准评估体系

生产级评估体系：**RAGAS 标准质量指标 + 性能/成本指标 + Langfuse Cloud 追踪与在线评估 + Locust 压测**，所有指标**中英双语显示**（如 `faithfulness（忠实度）`、`TTFT（首 token 延迟）`）。

### 质量指标（RAGAS 标准实现，零自研 judge）

| 指标（双语） | 说明 | 前置数据 |
|------|------|------|
| `faithfulness（忠实度）` | 答案陈述能否在检索上下文中找到依据（反幻觉核心） | answer + retrieved_contexts |
| `answer_relevancy（答案相关性）` | 答案是否直接有效回应用户问题 | user_input + answer |
| `factual_correctness（事实正确性）` | 答案与标准答案的事实一致性 | reference |
| `context_precision（上下文精度）` | 检索排序质量 / 噪音比例 | reference_contexts |
| `context_recall（上下文召回）` | 检索漏检检测 | reference_contexts |

所有 judge 调用统一走 `EVAL_JUDGE_MODEL`（独立强评判模型，可配置独立 `EVAL_JUDGE_BASE_URL/API_KEY`）；未配置时回退 `LLM_MODEL_STRONG` 并在报告标注"judge 与被测同源"。

### 性能指标（Prometheus + 压测）

`TTFT（首 token 延迟）`、`e2e_latency（端到端延迟 p50/p95/p99）`、`cost_per_query（单请求成本）`、`cache_hit_rate（缓存命中率）`、`error_rate（错误率）`、`throughput（吞吐 QPS）`。

### 数据集（标准 schema）

```jsonl
{"question": "用户问题", "reference": "标准答案", "reference_contexts": ["golden 上下文", "..."]}
```

`reference_contexts` 可选：缺失时自动跳过 context_precision/context_recall 并在报告标注。内置冒烟数据集 `eval/datasets/smoke.jsonl`（5 条，覆盖全部 5 个质量指标）。

### 运行离线评估

```bash
# stub 冒烟（不调用真实 LLM，验证流水线机制）
uv run python -m src.eval.runner --dataset eval/datasets/smoke.jsonl --fake-scores --name smoke

# 真实评估（需要外网：DashScope + 独立 judge）
uv run python -m src.eval.runner --dataset eval/datasets/smoke.jsonl --name smoke

# 发布门禁（失败退出码非 0，按 EVAL_GATE_THRESHOLDS 判定；未配置阈值时拒绝执行）
uv run python -m src.eval.runner --dataset eval/datasets/v3.jsonl --gate
```

#### 低频不并发模式（采集“完全正常响应”下的数据）

质量评估的样本采集本身是逐条串行的，但 RAGAS judge 打分阶段默认按
`batch_size=4` 并发执行，可能触发上游（如百炼 LLM/重排）的频率限速或偶发
延迟惩罚。加 `--paced` 可开启低频不并发模式：样本之间按 `EVAL_PACED_DELAY`
（默认 5 秒）固定间隔，RAGAS 打分改为 `batch_size=1` 完全串行，确保拿到
正常响应下的指标数据。

```bash
# 低频不并发模式（样本间隔默认 5s，RAGAS 串行打分）
uv run python -m src.eval.runner --dataset eval/datasets/smoke.jsonl --name smoke-paced --paced

# 间隔可通过环境变量调整（0~120 秒，启动校验）
EVAL_PACED_DELAY=8 uv run python -m src.eval.runner --dataset eval/datasets/smoke.jsonl --name smoke-paced --paced
```

报告会标注 `低频不并发模式 / Paced mode: ON`。

评估请求固定 `use_cache=False`（评测真实生成质量，缓存回放不参与评测）；结果输出到 `eval/results/quality/`（JSON + Markdown 双语报告），与压测结果 `eval/results/perf/` 平级，并上传数据集与每 trace 得分到 Langfuse（未配置则跳过）。报告会明确标注三类异常，不静默吞掉：**空答案/兜底样本**（不计分）、**指标计算失败**（judge 调用异常等，含 ragas 过程日志）、**跳过指标**（缺少 golden 上下文）。

### 在线评估与用户反馈

- 生产追踪：配置 `LANGFUSE_PUBLIC_KEY/SECRET_KEY` 后，每次对话自动在 Langfuse 生成 trace（含全链路 span、query/answer/检索上下文/耗时/缓存命中）
- 在线采样打分：`uv run python -m src.eval.online --limit 200 --days 7`，按 `EVAL_SAMPLE_RATE` 采样生产 trace，对无需 reference 的指标（faithfulness/answer_relevancy/context_precision）打分并写回；`--dry-run` 只采样不打分
- 用户反馈：对话气泡 👍/👎 → `POST /api/v1/chat/feedback` → 写回 Langfuse `user_feedback` 评分

### 压测（Locust）

```bash
cd eval/load
locust -f locustfile.py --host http://localhost:8000 --headless -u 5 -r 1 -t 60s
```

**低频不并发模式**（避开上游限速/延迟惩罚，采集“完全正常响应”下的数据）：

```bash
cd eval/load
EVAL_LOAD_PACED=1 EVAL_LOAD_WAIT_MIN=3 EVAL_LOAD_WAIT_MAX=6 \
  locust -f locustfile.py --host http://localhost:8000 --headless -u 1 -r 1 -t 300s
```

- 单用户串行 + 3~6 秒思考间隔，数秒才发一次请求，不测容量吞吐（报告中 QPS 标注“未测容量吞吐 / paced”）
- 启动时若并发用户 >1 或思考间隔 <2 秒会打印警告，提示改为上述配置
- `EVAL_LOAD_QUERIES='["问题1","问题2",...]'`（JSON 数组）可追加未缓存问题，采集未命中缓存的完整流水线指标

SSE 流式消费 `/api/v1/chat/stream`，默认小问题池（首轮写缓存后多为命中，控制成本）；`EVAL_LOAD_UNIQUE=1` 强制未命中。结束时输出双语汇总并写入 `eval/results/perf/`：**吞吐 QPS、端到端延迟 p50/p95/p99、首 token 延迟（TTFT）p50/p95/p99、缓存命中率（精准/语义/未命中分布）、分阶段耗时、单请求成本估算**（best-effort，基于 `/metrics` token 增量 × 单价，不可达时标注 N/A）；按 `EVAL_LOAD_P95_MAX`（默认 10s）与 `EVAL_LOAD_ERROR_RATE_MAX`（默认 0.01）断言，失败退出码非 0。

## 前端功能

- **Agent 流式对话**：SSE 逐字渲染，支持引文标注（[N] 悬停/点击查看来源段落）
- **7 个策略开关**：联网搜索、自反思、重排序、查询重写、BM25 检索、Multi-Query 检索、知识图谱检索，实时切换（文档评估无需手动开关，由查询重写/联网搜索状态自动决定）
- **Sidebar 流程图**：SVG 交互式状态图，含精准缓存/语义缓存/输出回放/缓存写入虚拟节点，命中/未命中箭头与节点耗时实时展示
- **会话历史**：服务端持久化，侧边栏会话列表由 `GET /chat/sessions` 加载，删除会话同步清除后端数据
- **用户反馈**：对话气泡 👍/👎，写回 Langfuse trace 评分
- **知识库管理**：拖拽上传（PDF/MD/TXT/DOCX/CSV），查看/删除已索引文档，上传任务状态实时跟踪
- **实时健康监控**：顶部栏显示服务状态，30 秒自动刷新

## 缓存与可观测性

### 多级缓存

- **精准缓存**：规范化问题（NFKC + 小写 + 空白折叠）+ 策略配置签名精确匹配，秒级命中
- **语义缓存**：问题向量余弦相似度 ≥ `CACHE_SEMANTIC_THRESHOLD`（默认 0.92）命中
- **向量复用**：语义缓存阶段计算的问题向量经 `AgentState.query_embedding` 传入图内，检索/MMR 直接复用，不重复调用 Embedding
- **写回条件**：反射关闭或幻觉检测通过且答案非空；命中时回放存储的答案/来源/路径/幻觉结果，零 LLM 调用
- **生命周期**：LRU 容量上限（`CACHE_MAX_ENTRIES`）+ 可选 TTL；删除文档、同名文档更新内容时按 doc_id 精确失效相关条目

### 可观测性

- `/metrics`：Prometheus 指标（请求数、缓存命中 `{type=exact|semantic}`、LLM/Embedding 调用数、token 用量 `{model,type}`、成本估算、TTFT 与分阶段耗时直方图、缓存节省 LLM 调用、上传统计）
- `/health`：逐组件健康明细（chroma/state_db/cache/kg/config），`?deep=true` 额外做 Embedding 探针
- 日志：JSON 结构化行格式，`log/app.log` 滚动（10MB × 5），异常堆栈完整保留在 `exc_info` 字段
- Langfuse：生产对话全链路 trace + 在线采样打分 + 用户反馈（未配置时全链路优雅降级）

## 项目结构

```
agentic-rag/
├── main.py                    # 项目入口
├── pyproject.toml             # Python 项目配置（UV 包管理）
├── .env                       # 环境变量（需自行创建）
├── src/                       # 后端源码
│   ├── main.py                # FastAPI 应用入口（日志/健康/指标/生命周期）
│   ├── config/settings.py     # Pydantic Settings 配置
│   ├── config/validation.py   # 启动配置校验（错误含当前值/原因/位置/修复）
│   ├── api/                   # API 路由与依赖注入
│   │   ├── router.py          # 主路由聚合
│   │   ├── dependencies.py    # 依赖注入（Service/KG 单例）
│   │   ├── chat.py            # 对话 API
│   │   └── documents.py       # 文档管理 API
│   ├── models/                # Pydantic 请求/响应模型
│   │   ├── chat.py            # 对话模型（策略开关 + use_cache）
│   │   ├── document.py        # 文档模型
│   │   └── common.py          # 通用模型
│   ├── agent/                 # LangGraph Agent
│   │   ├── state.py           # AgentState 定义（含 query_embedding/kg_intent 等）
│   │   ├── graph.py           # StateGraph 构建 + 条件路由
│   │   ├── nodes.py           # 图内节点实现（并行检索合并、评估、生成等）
│   │   ├── prompts.py         # Prompt 模板（含 KG 意图分析、实体抽取等）
│   │   └── tools.py           # Tool Calling（DuckDuckGo 搜索）
│   ├── cache/                 # 多级缓存（精准 + 语义）
│   │   ├── service.py         # 缓存编排（lookup/store/replay/invalidate）
│   │   └── storage.py         # SQLite 持久化 + numpy 内存向量索引
│   ├── eval/                  # 标准评估体系
│   │   ├── metrics.py         # 双语指标注册表（RAGAS 标准质量指标 + 性能/体验指标）
│   │   ├── judge.py           # LLM-as-judge 评判模型（EVAL_JUDGE_MODEL）
│   │   ├── langfuse.py        # Langfuse 接入（懒加载 + 优雅降级 + 打分）
│   │   ├── dataset.py         # 数据集 schema 校验与加载
│   │   ├── report.py          # 双语报告与门禁判定
│   │   ├── runner.py          # 离线评估 CLI（含 --fake-scores / --gate）
│   │   └── online.py          # 在线采样打分 CLI
│   ├── knowledge_graph/       # 知识图谱模块（Kuzu + numpy）
│   │   ├── __init__.py        # 单例工厂函数
│   │   ├── graph_store.py     # Kuzu 图数据库（原生持久化）
│   │   ├── graph_builder.py   # LLM 实体关系抽取 + 图谱构建
│   │   ├── graph_retriever.py # 实体链接（numpy 语义搜索）+ 子图提取 + 路径推理
│   │   └── kg_intent.py       # LLM 问题意图分析路由
│   ├── backend/               # AI 后端客户端
│   │   ├── llm.py             # ChatOpenAI 工厂（重试/计数/客户端复用）
│   │   ├── embedding.py       # DashScopeEmbeddings 工厂
│   │   └── reranker.py        # 百炼 TextReRank 重排序
│   ├── metrics.py             # Prometheus 指标定义
│   ├── retrieval/             # 检索策略
│   │   └── bm25.py            # BM25 关键词检索（jieba 分词）
│   ├── pipeline/              # 文档处理管道
│   │   ├── loader.py          # 多格式加载器（PDF/MD/TXT/DOCX/CSV）
│   │   ├── chunker.py         # 文本分块
│   │   └── indexer.py         # 文档索引器（含自动 KG 构建）
│   ├── store/                 # 数据存储
│   │   ├── vector_store.py    # ChromaDB 封装（含文档元数据查询）
│   │   └── state_store.py     # 会话历史/上传任务 SQLite 持久化
│   ├── memory/                # 对话记忆
│   │   └── manager.py         # 多会话隔离 + 滑动窗口 + 持久化恢复
│   └── services/              # 业务服务层
│       ├── rag_service.py     # RAG 对话服务（流式 + 缓存虚拟节点 + 图编排）
│       ├── document_service.py # 文档管理服务（有界索引队列）
│       ├── generator.py       # 生成节点（引文标注 + 流式输出）
│       └── hallucination_checker.py # 幻觉检测
├── tests/                     # pytest 测试（数据隔离，47 个用例）
├── Dockerfile                 # 后端容器镜像
├── docker-compose.yml         # 后端 + 前端（nginx 反代）编排
├── front/Dockerfile           # 前端容器镜像（node 构建 → nginx）
├── front/nginx.conf           # nginx 静态托管 + /api 反代配置
├── eval/                      # 评估数据与压测
│   ├── datasets/smoke.jsonl   # 标准 schema 冒烟数据集（5 条，含 golden 上下文）
│   ├── load/locustfile.py     # Locust SSE 压测（双语汇总 + 阈值断言）
│   ├── v1/                    # 历史数据集（不参与新体系，保留备查）
│   └── v2/                    # 历史数据集（不参与新体系，保留备查）
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
│       │   ├── useChat.ts     # 对话状态（服务端会话 + SSE 解析）
│       │   ├── agentFlowState.ts # Agent 流程状态（8 个策略开关 ref）
│       │   ├── useDocuments.ts # 文档管理
│       │   └── useHealth.ts   # 健康状态轮询
│       ├── views/             # 页面视图
│       │   ├── ChatView.vue   # 对话页
│       │   └── DocumentsView.vue # 知识库页
│       └── components/        # 组件
│           ├── layout/        # AppLayout + Sidebar（SVG 流程图含缓存节点）
│           ├── common/        # HealthBar
│           ├── chat/          # ChatPanel, ChatInput, MessageBubble, SourcePanel, SessionHistory
│           └── documents/     # DocumentUpload, DocumentList
├── data/                      # 本地数据存储（自动生成，已被 .gitignore 排除）
│   ├── chroma/                # ChromaDB 持久化数据
│   ├── kg/                    # Kuzu 图数据库
│   ├── cache/cache.db         # 多级缓存
│   ├── state/state.db         # 会话历史 / 上传任务
│   └── temp_uploads/          # 大文件上传中转（自动清理）
└── log/                       # 滚动日志（log/app.log，10MB × 5）
```

## 依赖

### 后端 (Python 3.12.13+)

| 包 | 用途 |
|------|------|
| `fastapi` + `uvicorn` | Web 框架与服务器 |
| `langgraph` | Agent 状态图编排 + 条件路由 |
| `langchain` + `langchain-openai` + `langchain-chroma` + `langchain-community` | RAG 组件链 |
| `chromadb` | 向量数据库（本地持久化，文档元数据一同存储） |
| `kuzu` | 图数据库（知识图谱存储，原生持久化） |
| `numpy` | 实体向量索引（.npz 二进制文件替代原 FAISS+SQLite） |
| `dashscope` | 百炼 LLM / Embedding / Rerank SDK |
| `langfuse` | LLM 生产追踪与评估平台（Cloud/自托管） |
| `ragas` | RAG 标准评估指标库（faithfulness/context_precision 等） |
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
| `prometheus-client` | Prometheus 指标（`/metrics` 导出，纯 Python） |
| `sqlite3`（标准库） | 多级缓存 / 会话历史 / 上传任务持久化，无需独立部署数据库 |

### 后端开发依赖

| 包 | 用途 |
|------|------|
| `pytest` | 自动化测试（`tests/`，47 个用例） |
| `locust` | SSE 压测（`eval/load/locustfile.py`） |

### 前端 (Node.js)

| 包 | 用途 |
|------|------|
| `vue` 3.5 + `vue-router` 4 | 前端框架 |
| `vite` 5 + `@vitejs/plugin-vue` | 构建工具 |
| `tailwindcss` 3.4 | CSS 框架 |
| `lucide-vue-next` | 图标库 |
| `typescript` | 类型检查 |
