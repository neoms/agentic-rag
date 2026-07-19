import { ref } from 'vue'
import type { DocumentInfo } from '../types'
import { getDocumentList, uploadDocument, deleteDocument } from '../api/documents'

export function useDocuments() {
  const documents = ref<DocumentInfo[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)
  const uploading = ref(false)
  const uploadError = ref<string | null>(null)

  async function fetchDocuments() {
    loading.value = true
    error.value = null
    const result = await getDocumentList()
    if (result) {
      documents.value = result.documents
    } else {
      error.value = '获取文档列表失败'
    }
    loading.value = false
  }

  async function upload(file: File): Promise<boolean> {
    uploading.value = true
    uploadError.value = null

    const ext = file.name.split('.').pop()?.toLowerCase()
    const allowed = ['pdf', 'md', 'txt']
    if (!ext || !allowed.includes(ext)) {
      uploadError.value = `不支持的文件格式: ${ext || '未知'}，仅支持 ${allowed.join(', ')}`
      uploading.value = false
      return false
    }

    const maxSize = 10 * 1024 * 1024
    if (file.size > maxSize) {
      uploadError.value = '文件大小超过 10MB 限制'
      uploading.value = false
      return false
    }

    if (file.size === 0) {
      uploadError.value = '文件内容为空'
      uploading.value = false
      return false
    }

    const result = await uploadDocument(file)
    if (result && result.success) {
      uploading.value = false
      await fetchDocuments()
      return true
    }

    uploadError.value = '上传失败'
    uploading.value = false
    return false
  }

  async function remove(docId: string): Promise<boolean> {
    const result = await deleteDocument(docId)
    if (result && result.success) {
      await fetchDocuments()
      return true
    }
    return false
  }

  return {
    documents,
    loading,
    error,
    uploading,
    uploadError,
    fetchDocuments,
    upload,
    remove,
  }
}
