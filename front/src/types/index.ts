// ===== 通用模型 =====
export interface HealthResponse {
  status: 'ok' | 'degraded'
  version: string
}

// ===== 对话模型 =====
export interface AgenticChatRequest {
  query: string
  session_id?: string
  enable_web_search?: boolean
  enable_reflection?: boolean
  enable_rerank?: boolean
  enable_grade_documents?: boolean
  enable_transform_query?: boolean
  enable_bm25?: boolean
  enable_multi_query?: boolean
  enable_kg?: boolean
}

export interface SourceDocument {
  content: string
  metadata: Record<string, string | number>
  score: number | null
}

export interface ChatHistoryMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface ChatHistoryResponse {
  session_id: string
  messages: ChatHistoryMessage[]
  total: number
}

// ===== SSE 流式事件 =====
export type SSEEventType = 'source' | 'citations' | 'path' | 'token' | 'done' | 'error' | 'hallucination' | 'node_start' | 'node_step' | 'node_data'

export interface SSEEvent {
  type: SSEEventType
  data: string
}

// ===== 文档模型 =====
export type TaskStatus = 'pending' | 'processing' | 'completed' | 'failed'

export interface TaskInfo {
  task_id: string
  doc_id: string
  filename: string
  status: TaskStatus
  message: string
  created_at: string
  completed_at: string | null
  chunk_count: number
}

export interface TaskSubmitResponse {
  success: boolean
  task_id: string
  doc_id: string
  filename: string
  status: TaskStatus
  message: string
}

export interface DocumentUploadResponse {
  success: boolean
  doc_id: string
  filename: string
  chunk_count: number
  message: string
}

export interface DocumentInfo {
  doc_id: string
  filename: string
  file_type: string
  chunk_count: number
  size_bytes: number
  created_at: string
}

export interface DocumentListResponse {
  documents: DocumentInfo[]
  total: number
}

export interface DocumentDeleteResponse {
  success: boolean
  doc_id: string
  message: string
}

// ===== 引文标注模型 =====
export interface CitationInfo {
  filename: string
  source_type: string
  url: string
  paragraph_text: string
  doc_index: number
  para_index: number
}

export type CitationMetadata = Record<string, CitationInfo>

// ===== 对话 UI 模型 =====
export interface UIMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: number
  sources?: SourceDocument[]
  agent_path?: string[]
  reflection_count?: number
  tool_calls?: Record<string, string | number | boolean>[]
  isStreaming?: boolean
  hallucination_passed?: boolean
  citations?: CitationMetadata
}
