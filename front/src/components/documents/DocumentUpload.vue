<script setup lang="ts">
import { ref, computed } from 'vue'
import { Upload, FileText, AlertCircle, Loader2 } from 'lucide-vue-next'

const emit = defineEmits<{
  upload: [file: File]
}>()

const props = defineProps<{
  uploading?: boolean
  error?: string | null
}>()

const dragOver = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)

const allowedFormats = computed(() => 'PDF, Markdown, TXT · 最大 10MB')

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
      'relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 cursor-pointer',
      dragOver
        ? 'border-primary-400 bg-primary-400/10 scale-[1.02]'
        : 'border-slate-600 hover:border-slate-500 hover:bg-slate-800/40'
    ]"
    @click="triggerFileInput"
  >
    <input
      ref="fileInput"
      type="file"
      accept=".pdf,.md,.txt"
      class="hidden"
      @change="onFileSelect"
    />

    <div v-if="uploading" class="flex flex-col items-center gap-3">
      <Loader2 class="w-10 h-10 text-primary-400 animate-spin" />
      <p class="text-sm text-slate-300">正在上传并索引文档...</p>
    </div>

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

      <!-- 错误提示 -->
      <div
        v-if="error"
        class="mt-4 flex items-center justify-center gap-2 text-xs text-red-400 bg-red-400/10 rounded-lg px-3 py-2 mx-auto max-w-md animate-fade-in"
      >
        <AlertCircle class="w-3.5 h-3.5 flex-shrink-0" />
        <span>{{ error }}</span>
      </div>
    </template>
  </div>
</template>
