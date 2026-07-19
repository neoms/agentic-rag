<script setup lang="ts">
import { onMounted } from 'vue'
import { useDocuments } from '../composables/useDocuments'
import DocumentUpload from '../components/documents/DocumentUpload.vue'
import DocumentList from '../components/documents/DocumentList.vue'

const { documents, loading, uploading, uploadError, fetchDocuments, upload, remove } = useDocuments()

onMounted(() => {
  fetchDocuments()
})

function handleUpload(file: File) {
  upload(file)
}

function handleDelete(docId: string) {
  remove(docId)
}
</script>

<template>
  <div class="h-full overflow-y-auto">
    <div class="max-w-3xl mx-auto px-6 py-8">
      <!-- 页面标题 -->
      <div class="mb-8">
        <h1 class="text-xl font-bold text-slate-100">知识库管理</h1>
        <p class="text-sm text-slate-500 mt-1">上传文档构建知识库，支持 PDF、Markdown、TXT 格式</p>
      </div>

      <!-- 上传区 -->
      <div class="mb-8">
        <DocumentUpload
          :uploading="uploading"
          :error="uploadError"
          @upload="handleUpload"
        />
      </div>

      <!-- 文档列表 -->
      <DocumentList
        :documents="documents"
        :loading="loading"
        @delete="handleDelete"
      />
    </div>
  </div>
</template>
