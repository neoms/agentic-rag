import { request } from './client'
import type {
  TaskSubmitResponse,
  TaskInfo,
  DocumentListResponse,
  DocumentDeleteResponse,
} from '../types'

/**
 * 上传文档（异步后台处理）
 * 返回 202 + task_id，实际索引在后台进行。
 */
export function uploadDocument(file: File): Promise<TaskSubmitResponse> {
  const formData = new FormData()
  formData.append('file', file)

  return fetch('/api/v1/documents/upload', {
    method: 'POST',
    body: formData,
  }).then(async (response) => {
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: '上传失败' }))
      throw new Error(error.detail || '上传失败')
    }
    return response.json()
  })
}

/**
 * 查询后台任务状态
 * GET /api/v1/documents/tasks/{task_id}
 */
export function getTaskStatus(taskId: string): Promise<TaskInfo> {
  return request<TaskInfo>(`/documents/tasks/${encodeURIComponent(taskId)}`)
}

/**
 * 获取所有后台任务状态
 * GET /api/v1/documents/tasks
 */
export function listTasks(): Promise<TaskInfo[]> {
  return request<TaskInfo[]>('/documents/tasks')
}

/**
 * 获取已索引文档列表
 * GET /api/v1/documents
 */
export function getDocumentList(): Promise<DocumentListResponse> {
  return request<DocumentListResponse>('/documents')
}

/**
 * 删除文档
 * DELETE /api/v1/documents/{doc_id}
 */
export function deleteDocument(docId: string): Promise<DocumentDeleteResponse> {
  return request<DocumentDeleteResponse>(`/documents/${encodeURIComponent(docId)}`, {
    method: 'DELETE',
  })
}
