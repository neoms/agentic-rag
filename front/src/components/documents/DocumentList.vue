<script setup lang="ts">
import { onMounted } from 'vue'
import { File, Trash2, Hash, Calendar, Loader2, Database } from 'lucide-vue-next'
import type { DocumentInfo } from '../../types'

defineProps<{
  documents: DocumentInfo[]
  loading?: boolean
}>()

const emit = defineEmits<{
  delete: [docId: string]
}>()

const fileTypeColors: Record<string, string> = {
  pdf: 'bg-red-400/10 text-red-400 border-red-400/20',
  md: 'bg-blue-400/10 text-blue-400 border-blue-400/20',
  txt: 'bg-slate-400/10 text-slate-400 border-slate-400/20',
}

const fileTypeIcons: Record<string, string> = {
  pdf: 'PDF',
  md: 'MD',
  txt: 'TXT',
}

function formatDate(dateStr: string): string {
  const date = new Date(dateStr)
  return date.toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function formatSize(bytes: number): string {
  if (bytes === 0) return '-'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
</script>

<template>
  <div class="space-y-3">
    <!-- 表头 -->
    <div class="flex items-center justify-between">
      <h2 class="text-sm font-semibold text-slate-300">
        已索引文档 ({{ documents.length }})
      </h2>
      <button
        v-if="loading"
        class="flex items-center gap-1.5 text-xs text-slate-500"
      >
        <Loader2 class="w-3 h-3 animate-spin" />
        刷新中
      </button>
    </div>

    <!-- 空状态 -->
    <div
      v-if="!loading && documents.length === 0"
      class="flex flex-col items-center justify-center py-16 text-slate-500"
    >
      <Database class="w-12 h-12 mb-3 text-slate-600" />
      <p class="text-sm">暂无已索引文档</p>
      <p class="text-xs mt-1">上传 PDF、Markdown 或 TXT 文件开始构建知识库</p>
    </div>

    <!-- 加载中 -->
    <div
      v-if="loading && documents.length === 0"
      class="flex flex-col items-center justify-center py-16 text-slate-500"
    >
      <Loader2 class="w-8 h-8 mb-3 animate-spin text-primary-400" />
      <p class="text-sm">加载文档列表...</p>
    </div>

    <!-- 文档列表 -->
    <TransitionGroup name="list" tag="div" class="grid gap-3">
      <div
        v-for="doc in documents"
        :key="doc.doc_id"
        class="glass-hover rounded-lg p-4 flex items-center justify-between group animate-fade-in"
      >
        <div class="flex items-center gap-3 min-w-0">
          <div class="w-9 h-9 rounded-lg bg-slate-700/50 flex items-center justify-center flex-shrink-0">
            <File class="w-4 h-4 text-slate-400" />
          </div>
          <div class="min-w-0">
            <p class="text-sm font-medium text-slate-200 truncate">{{ doc.filename }}</p>
            <div class="flex items-center gap-3 mt-1">
              <span
                :class="[
                  'text-[10px] font-medium px-1.5 py-0.5 rounded border',
                  fileTypeColors[doc.file_type] || fileTypeColors.txt
                ]"
              >
                {{ fileTypeIcons[doc.file_type] || doc.file_type.toUpperCase() }}
              </span>
              <span class="text-[11px] text-slate-500 flex items-center gap-1">
                <Hash class="w-3 h-3" />
                {{ doc.chunk_count }} 块
              </span>
              <span class="text-[11px] text-slate-500 flex items-center gap-1">
                <Calendar class="w-3 h-3" />
                {{ formatDate(doc.created_at) }}
              </span>
              <span class="text-[11px] text-slate-600">{{ formatSize(doc.size_bytes) }}</span>
            </div>
          </div>
        </div>

        <button
          @click="emit('delete', doc.doc_id)"
          class="p-2 rounded-lg text-slate-600 hover:text-red-400 hover:bg-red-400/10 transition-all duration-200 opacity-0 group-hover:opacity-100 flex-shrink-0"
          title="删除文档"
        >
          <Trash2 class="w-4 h-4" />
        </button>
      </div>
    </TransitionGroup>
  </div>
</template>

<style scoped>
.list-enter-active,
.list-leave-active {
  transition: all 0.3s ease;
}
.list-enter-from {
  opacity: 0;
  transform: translateY(10px);
}
.list-leave-to {
  opacity: 0;
  transform: translateX(20px);
}
</style>
