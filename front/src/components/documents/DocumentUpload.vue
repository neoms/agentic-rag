<script setup lang="ts">
import { ref, computed } from 'vue'
import { Upload, FileText, AlertCircle, Loader2, CheckCircle2, XCircle, Clock } from 'lucide-vue-next'
import type { TaskStatus } from '../../types'

const emit = defineEmits<{
  upload: [file: File]
  'clear-error': []
}>()

const props = defineProps<{
  uploading?: boolean
  error?: string | null
  taskStatus?: TaskStatus | null
  taskMessage?: string
}>()

const dragOver = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)

const allowedFormats = computed(() => 'PDF, Markdown, TXT, DOCX, CSV · 最大 10MB')

const statusIcon = computed(() => {
  switch (props.taskStatus) {
    case 'pending': return Clock
    case 'processing': return Loader2
    case 'completed': return CheckCircle2
    case 'failed': return XCircle
    default: return null
  }
})

const statusColor = computed(() => {
  switch (props.taskStatus) {
    case 'pending': return 'text-amber-400'
    case 'processing': return 'text-primary-400'
    case 'completed': return 'text-emerald-400'
    case 'failed': return 'text-red-400'
    default: return ''
  }
})

const statusBgColor = computed(() => {
  switch (props.taskStatus) {
    case 'pending': return 'bg-amber-400/10'
    case 'processing': return 'bg-primary-400/10'
    case 'completed': return 'bg-emerald-400/10'
    case 'failed': return 'bg-red-400/10'
    default: return ''
  }
})

function onDragOver(e: DragEvent) {
  e.preventDefault()
  dragOver.value = true
}

function onDragLeave() {
  dragOver.value = false
}

function onDrop(e: DragEvent) {
  e.preventDefault()
  dragOver.value = false
  const file = e.dataTransfer?.files?.[0]
  if (file) {
    emit('upload', file)
  }
}

function onFileSelect(e: Event) {
  const input = e.target as HTMLInputElement
  const file = input.files?.[0]
  if (file) {
    emit('upload', file)
    input.value = ''
  }
}

function triggerFileInput() {
  fileInput.value?.click()
}
</script>

<template>
  <div
    @dragover="onDragOver"
    @dragleave="onDragLeave"
    @drop="onDrop"
    :class="[
      'relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300',
      uploading ? 'cursor-default border-slate-600' : 'cursor-pointer',
      dragOver
        ? 'border-primary-400 bg-primary-400/10 scale-[1.02]'
        : 'border-slate-600 hover:border-slate-500 hover:bg-slate-800/40'
    ]"
    @click="!uploading && triggerFileInput()"
  >
    <input
      ref="fileInput"
      type="file"
      accept=".pdf,.md,.txt,.docx,.csv"
      class="hidden"
      @change="onFileSelect"
    />

    <!-- 上传中 / 任务处理中 -->
    <div v-if="uploading && taskStatus" class="flex flex-col items-center gap-3">
      <div
        :class="[
          'w-12 h-12 rounded-2xl flex items-center justify-center transition-colors',
          statusBgColor
        ]"
      >
        <component
          :is="statusIcon"
          :class="['w-6 h-6', statusColor, taskStatus === 'processing' ? 'animate-spin' : '']"
        />
      </div>
      <p class="text-sm font-medium text-slate-200">
        {{ taskStatus === 'pending' ? '任务已提交' : taskStatus === 'processing' ? '正在处理文档' : taskStatus === 'completed' ? '处理完成' : '处理失败' }}
      </p>
      <p class="text-xs text-slate-500 max-w-xs">{{ taskMessage }}</p>
    </div>

    <!-- 上传中（初始状态，尚未获取 task_id） -->
    <div v-else-if="uploading" class="flex flex-col items-center gap-3">
      <Loader2 class="w-10 h-10 text-primary-400 animate-spin" />
      <p class="text-sm text-slate-300">正在上传文件...</p>
    </div>

    <!-- 正常状态 -->
    <template v-else>
      <div
        :class="[
          'w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-4 transition-colors',
          dragOver ? 'bg-primary-500/20' : 'bg-slate-700/50'
        ]"
      >
        <Upload
          :class="[
            'w-6 h-6',
            dragOver ? 'text-primary-400' : 'text-slate-400'
          ]"
        />
      </div>

      <p class="text-sm font-medium text-slate-200 mb-1">
        拖拽文件到此处或点击选择
      </p>
      <p class="text-xs text-slate-500">{{ allowedFormats }}</p>

      <!-- 错误提示（可关闭） -->
      <div
        v-if="error"
        class="mt-4 flex items-center justify-between gap-2 text-xs text-red-400 bg-red-400/10 rounded-lg px-3 py-2 mx-auto max-w-md animate-fade-in"
      >
        <div class="flex items-center gap-2">
          <AlertCircle class="w-3.5 h-3.5 flex-shrink-0" />
          <span class="text-left">{{ error }}</span>
        </div>
        <button
          @click.stop="emit('clear-error')"
          class="text-red-400/60 hover:text-red-400 flex-shrink-0 transition-colors"
          title="关闭"
        >
          <XCircle class="w-3.5 h-3.5" />
        </button>
      </div>
    </template>
  </div>
</template>
