import { ref, onUnmounted } from 'vue'
import type { DocumentInfo, TaskStatus } from '../types'
import {
  getDocumentList,
  uploadDocument,
  deleteDocument,
  getTaskStatus,
} from '../api/documents'

const ALLOWED_EXTENSIONS = ['pdf', 'md', 'txt', 'docx', 'csv']
const MAX_UPLOAD_SIZE_MB = 10
const POLL_INTERVAL_MS = 2000
const MAX_POLL_COUNT = 60 // 最多轮询 2 分钟

export function useDocuments() {
  const documents = ref<DocumentInfo[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)
  const uploading = ref(false)
  const uploadError = ref<string | null>(null)

  // 异步任务状态跟踪
  const uploadTaskStatus = ref<TaskStatus | null>(null)
  const uploadTaskMessage = ref<string>('')
  const uploadTaskId = ref<string | null>(null)

  let pollTimer: ReturnType<typeof setInterval> | null = null

  onUnmounted(() => {
    stopPolling()
  })

  function stopPolling() {
    if (pollTimer !== null) {
      clearInterval(pollTimer)
      pollTimer = null
    }
  }

  /**
   * 轮询任务状态直到完成或失败
   */
  function startPolling(taskId: string) {
    stopPolling()
    let pollCount = 0

    pollTimer = setInterval(async () => {
      pollCount++
      try {
        const task = await getTaskStatus(taskId)
        uploadTaskStatus.value = task.status
        uploadTaskMessage.value = task.message

        if (task.status === 'completed') {
          stopPolling()
          uploading.value = false
          await fetchDocuments()
          // 3 秒后清除完成状态
          setTimeout(() => {
            if (uploadTaskId.value === taskId) {
              uploadTaskStatus.value = null
              uploadTaskMessage.value = ''
              uploadTaskId.value = null
            }
          }, 3000)
        } else if (task.status === 'failed') {
          stopPolling()
          uploading.value = false
          uploadError.value = task.message || '文档处理失败'
        } else if (pollCount >= MAX_POLL_COUNT) {
          stopPolling()
          uploading.value = false
          uploadError.value = '文档处理超时，请稍后刷新查看结果'
        }
      } catch (err) {
        stopPolling()
        uploading.value = false
        uploadError.value = err instanceof Error ? err.message : '查询任务状态失败'
      }
    }, POLL_INTERVAL_MS)
  }

  /**
   * 获取文档列表
   */
  async function fetchDocuments() {
    loading.value = true
    error.value = null
    try {
      const result = await getDocumentList()
      documents.value = result.documents
    } catch (err) {
      error.value = err instanceof Error ? err.message : '获取文档列表失败'
    } finally {
      loading.value = false
    }
  }

  /**
   * 上传文档（异步后台处理）
   * 1. 前端本地校验
   * 2. 提交文件 → 获取 task_id
   * 3. 轮询任务状态 → 完成后刷新列表
   */
  async function upload(file: File): Promise<boolean> {
    uploading.value = true
    uploadError.value = null
    uploadTaskStatus.value = null
    uploadTaskMessage.value = ''
    uploadTaskId.value = null

    // 前端校验：扩展名
    const ext = file.name.split('.').pop()?.toLowerCase()
    if (!ext || !ALLOWED_EXTENSIONS.includes(ext)) {
      uploadError.value = `不支持的文件格式: ${ext || '未知'}，仅支持 ${ALLOWED_EXTENSIONS.join(', ').toUpperCase()}`
      uploading.value = false
      return false
    }

    // 前端校验：文件大小
    const maxSize = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if (file.size > maxSize) {
      uploadError.value = `文件大小超过 ${MAX_UPLOAD_SIZE_MB}MB 限制`
      uploading.value = false
      return false
    }

    // 前端校验：空文件
    if (file.size === 0) {
      uploadError.value = '文件内容为空'
      uploading.value = false
      return false
    }

    try {
      const result = await uploadDocument(file)
      if (result.success) {
        uploadTaskId.value = result.task_id
        uploadTaskStatus.value = result.status
        uploadTaskMessage.value = result.message
        // 启动轮询
        startPolling(result.task_id)
        return true
      }
      uploadError.value = '上传失败'
      uploading.value = false
      return false
    } catch (err) {
      uploadError.value = err instanceof Error ? err.message : '上传失败'
      uploading.value = false
      return false
    }
  }

  /**
   * 删除文档
   */
  async function remove(docId: string): Promise<boolean> {
    try {
      const result = await deleteDocument(docId)
      if (result.success) {
        await fetchDocuments()
        return true
      }
      return false
    } catch (err) {
      error.value = err instanceof Error ? err.message : '删除失败'
      return false
    }
  }

  /**
   * 清除上传错误
   */
  function clearUploadError() {
    uploadError.value = null
  }

  return {
    documents,
    loading,
    error,
    uploading,
    uploadError,
    uploadTaskStatus,
    uploadTaskMessage,
    uploadTaskId,
    fetchDocuments,
    upload,
    remove,
    clearUploadError,
  }
}
