// ===== 通用模型 =====
export interface HealthResponse {
  status: 'ok' | 'degraded'
  version: string
  chroma_count: number
  llm_model: string
  embedding_model: string
}

// ===== 对话模型 =====
export interface ChatRequest {
  query: string
  session_id?: string
  top_k?: number | null
}

export interface AgenticChatRequest extends ChatRequest {
  enable_web_search?: boolean
  enable_reflection?: boolean
  stream?: boolean
}

export interface SourceDocument {
  content: string
  metadata: Record<string, string | number>
  score: number | null
}

export interface ChatResponse {
  answer: string
  session_id: string
  sources: SourceDocument[]
  reflection_count: number
}

export interface AgenticChatResponse extends ChatResponse {
  tool_calls: Record<string, string | number | boolean>[]
  agent_path: string[]
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
export type SSEEventType = 'source' | 'path' | 'token' | 'done' | 'error'

export interface SSEEvent {
  type: SSEEventType
  data: string
}

// ===== 文档模型 =====
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
}
