import { request, fetchRaw } from './client'
import type {
  AgenticChatRequest,
  ChatHistoryResponse,
} from '../types'

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
