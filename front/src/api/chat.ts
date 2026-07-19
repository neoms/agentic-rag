import { request, fetchRaw } from './client'
import type {
  ChatRequest,
  AgenticChatRequest,
  ChatResponse,
  AgenticChatResponse,
  ChatHistoryResponse,
} from '../types'

export function simpleChat(data: ChatRequest): Promise<ChatResponse> {
  return request<ChatResponse>('/chat/simple', {
    method: 'POST',
    body: JSON.stringify(data),
  })
}

export function agenticChat(data: AgenticChatRequest): Promise<AgenticChatResponse> {
  return request<AgenticChatResponse>('/chat/agentic', {
    method: 'POST',
    body: JSON.stringify(data),
  })
}

export function streamChat(data: AgenticChatRequest): Promise<Response> {
  return fetchRaw('/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
}

export function getChatHistory(sessionId: string): Promise<ChatHistoryResponse> {
  return request<ChatHistoryResponse>(`/chat/history/${encodeURIComponent(sessionId)}`)
}
