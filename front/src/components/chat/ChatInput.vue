<script setup lang="ts">
import { ref, computed } from 'vue'
import { Send, Loader2, Globe, Brain, Zap } from 'lucide-vue-next'
import type { ChatMode } from '../../types'

const props = defineProps<{
  mode: ChatMode
  sending: boolean
  enableWebSearch?: boolean
  enableReflection?: boolean
}>()

const emit = defineEmits<{
  send: [query: string]
  'update:mode': [mode: ChatMode]
  'update:enableWebSearch': [val: boolean]
  'update:enableReflection': [val: boolean]
}>()

const query = ref('')

const isAgentMode = computed(() => props.mode === 'agentic' || props.mode === 'stream')

function handleSend() {
  if (!query.value.trim() || props.sending) return
  emit('send', query.value)
  query.value = ''
}

function handleKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    handleSend()
  }
}

const modeOptions: { value: ChatMode; label: string; desc: string }[] = [
  { value: 'simple', label: '基础', desc: '检索 + 生成' },
  { value: 'agentic', label: 'Agent', desc: '自反思 + 工具调用' },
  { value: 'stream', label: '流式', desc: 'Agent + 实时输出' },
]
</script>

<template>
  <div class="border-t border-slate-700/50 bg-slate-900/60 backdrop-blur-lg">
    <!-- Agent 选项 -->
    <div
      v-if="isAgentMode"
      class="flex items-center gap-3 px-4 pt-3 pb-1"
    >
      <button
        @click="emit('update:enableWebSearch', !enableWebSearch)"
        :class="[
          'flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs transition-all duration-200 border',
          enableWebSearch
            ? 'bg-emerald-400/10 text-emerald-400 border-emerald-400/20'
            : 'text-slate-500 border-slate-700/50 hover:text-slate-400'
        ]"
      >
        <Globe class="w-3 h-3" />
        联网搜索
      </button>
      <button
        @click="emit('update:enableReflection', !enableReflection)"
        :class="[
          'flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs transition-all duration-200 border',
          enableReflection
            ? 'bg-blue-400/10 text-blue-400 border-blue-400/20'
            : 'text-slate-500 border-slate-700/50 hover:text-slate-400'
        ]"
      >
        <Brain class="w-3 h-3" />
        自反思
      </button>
      <span class="text-[10px] text-slate-600 ml-auto">
        Shift+Enter 换行
      </span>
    </div>

    <!-- 输入区 -->
    <div class="flex items-end gap-3 px-4 py-3">
      <!-- 模式切换 -->
      <div class="flex items-center bg-slate-800 rounded-lg p-0.5 gap-0.5">
        <button
          v-for="opt in modeOptions"
          :key="opt.value"
          @click="emit('update:mode', opt.value)"
          :class="[
            'px-3 py-1.5 rounded-md text-xs font-medium transition-all duration-200 flex items-center gap-1.5',
            mode === opt.value
              ? 'bg-primary-500/20 text-primary-400 shadow-sm'
              : 'text-slate-500 hover:text-slate-400'
          ]"
          :title="opt.desc"
        >
          <Zap v-if="opt.value === 'stream'" class="w-3 h-3" />
          <span>{{ opt.label }}</span>
        </button>
      </div>

      <!-- 输入框 -->
      <div class="flex-1 relative">
        <textarea
          v-model="query"
          @keydown="handleKeydown"
          :disabled="sending"
          rows="1"
          placeholder="输入您的问题..."
          class="w-full bg-slate-800/60 text-sm text-slate-200 placeholder-slate-600 rounded-xl px-4 py-2.5 resize-none outline-none border border-slate-700/50 focus:border-primary-500/50 focus:bg-slate-800 transition-all duration-200 disabled:opacity-50"
        />
      </div>

      <!-- 发送按钮 -->
      <button
        @click="handleSend"
        :disabled="!query.trim() || sending"
        class="p-2.5 rounded-xl bg-gradient-to-r from-primary-500 to-primary-600 text-white hover:shadow-lg hover:shadow-primary-500/20 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed active:scale-95"
      >
        <Loader2 v-if="sending" class="w-4 h-4 animate-spin" />
        <Send v-else class="w-4 h-4" />
      </button>
    </div>
  </div>
</template>
