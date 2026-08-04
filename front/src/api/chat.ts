import { request, fetchRaw } from './client'
import type {
  AgenticChatRequest,
  ChatHistoryResponse,
  ChatSessionsResponse,
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

export function listChatSessions(): Promise<ChatSessionsResponse> {
  return request<ChatSessionsResponse>('/chat/sessions')
}

export function deleteChatHistory(sessionId: string): Promise<{ success: boolean; message: string }> {
  return request(`/chat/history/${encodeURIComponent(sessionId)}`, { method: 'DELETE' })
}

export function submitFeedback(
  traceId: string,
  rating: number,
  comment?: string,
): Promise<{ success: boolean; message: string }> {
  return request('/chat/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trace_id: traceId, rating, comment }),
  })
}
