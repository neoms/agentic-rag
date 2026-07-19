import { request } from './client'
import type {
  DocumentUploadResponse,
  DocumentListResponse,
  DocumentDeleteResponse,
} from '../types'

export function uploadDocument(file: File): Promise<DocumentUploadResponse> {
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

export function getDocumentList(): Promise<DocumentListResponse> {
  return request<DocumentListResponse>('/documents')
}

export function deleteDocument(docId: string): Promise<DocumentDeleteResponse> {
  return request<DocumentDeleteResponse>(`/documents/${encodeURIComponent(docId)}`, {
    method: 'DELETE',
  })
}
