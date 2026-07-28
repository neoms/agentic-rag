<script setup lang="ts">
import { ref } from 'vue'
import { Send, Loader2, Globe, Brain, ListOrdered, Sparkles, Shuffle, SearchCode, Lightbulb, GitBranch, Network } from 'lucide-vue-next'
import {
  enableWebSearch,
  enableReflection,
  enableRerank,
  enableGradeDocuments,
  enableTransformQuery,
  enableBm25,
  enableHyde,
  enableMultiQuery,
  enableKg,
} from '../../composables/agentFlowState'

const props = defineProps<{
  sending: boolean
}>()

const emit = defineEmits<{
  send: [query: string]
}>()

const query = ref('')
const webSearchBtn = ref<HTMLElement | null>(null)
const showWebSearchTip = ref(false)
const tipStyle = ref<Record<string, string>>({})
let webSearchTipTimer: ReturnType<typeof setTimeout> | null = null

function updateTipPosition() {
  if (!webSearchBtn.value) return
  const rect = webSearchBtn.value.getBoundingClientRect()
  tipStyle.value = {
    left: `${rect.left + rect.width / 2}px`,
    top: `${rect.top - 8}px`,
  }
}

function handleWebSearchToggle() {
  enableWebSearch.value = !enableWebSearch.value
  if (enableWebSearch.value) {
    updateTipPosition()
    showWebSearchTip.value = true
    if (webSearchTipTimer) clearTimeout(webSearchTipTimer)
    webSearchTipTimer = setTimeout(() => {
      showWebSearchTip.value = false
    }, 3000)
  }
}

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
</script>

<template>
  <div class="border-t border-slate-700/50 bg-slate-900/60 backdrop-blur-lg">
    <!-- Agent 选项第一行 -->
    <div class="flex items-center gap-2 px-4 pt-3 pb-0.5 flex-wrap">
      <button
        ref="webSearchBtn"
        @click="handleWebSearchToggle"
        :class="[
          'flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] transition-all duration-200 border',
          enableWebSearch
            ? 'bg-emerald-400/10 text-emerald-400 border-emerald-400/20'
            : 'text-slate-500 border-slate-700/50 hover:text-slate-400'
        ]"
      >
        <Globe class="w-3 h-3" />
        联网搜索
      </button>
      <button
        @click="enableRerank = !enableRerank"
        :class="[
          'flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] transition-all duration-200 border',
          enableRerank
            ? 'bg-amber-400/10 text-amber-400 border-amber-400/20'
            : 'text-slate-500 border-slate-700/50 hover:text-slate-400'
        ]"
      >
        <ListOrdered class="w-3 h-3" />
        重排序
      </button>
      <button
        @click="enableGradeDocuments = !enableGradeDocuments"
        :class="[
          'flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] transition-all duration-200 border',
          enableGradeDocuments
            ? 'bg-violet-400/10 text-violet-400 border-violet-400/20'
            : 'text-slate-500 border-slate-700/50 hover:text-slate-400'
        ]"
      >
        <Sparkles class="w-3 h-3" />
        文档评估
      </button>
      <button
        @click="enableTransformQuery = !enableTransformQuery"
        :class="[
          'flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] transition-all duration-200 border',
          enableTransformQuery
            ? 'bg-rose-400/10 text-rose-400 border-rose-400/20'
            : 'text-slate-500 border-slate-700/50 hover:text-slate-400'
        ]"
      >
        <Shuffle class="w-3 h-3" />
        查询重写
      </button>
      <button
        @click="enableBm25 = !enableBm25"
        :class="[
          'flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] transition-all duration-200 border',
          enableBm25
            ? 'bg-cyan-400/10 text-cyan-400 border-cyan-400/20'
            : 'text-slate-500 border-slate-700/50 hover:text-slate-400'
        ]"
      >
        <SearchCode class="w-3 h-3" />
        BM25
      </button>
      <button
        @click="enableHyde = !enableHyde"
        :class="[
          'flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] transition-all duration-200 border',
          enableHyde
            ? 'bg-pink-400/10 text-pink-400 border-pink-400/20'
            : 'text-slate-500 border-slate-700/50 hover:text-slate-400'
        ]"
      >
        <Lightbulb class="w-3 h-3" />
        HyDE
      </button>
      <button
        @click="enableMultiQuery = !enableMultiQuery"
        :class="[
          'flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] transition-all duration-200 border',
          enableMultiQuery
            ? 'bg-indigo-400/10 text-indigo-400 border-indigo-400/20'
            : 'text-slate-500 border-slate-700/50 hover:text-slate-400'
        ]"
      >
        <GitBranch class="w-3 h-3" />
        多角度查询
      </button>
      <button
        @click="enableKg = !enableKg"
        :class="[
          'flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] transition-all duration-200 border',
          enableKg
            ? 'bg-orange-400/10 text-orange-400 border-orange-400/20'
            : 'text-slate-500 border-slate-700/50 hover:text-slate-400'
        ]"
      >
        <Network class="w-3 h-3" />
        知识图谱
      </button>
      <span class="text-[10px] text-slate-600 ml-auto">
        Shift+Enter 换行
      </span>
    </div>

    <!-- 输入区 -->
    <div class="flex items-end gap-3 px-4 py-3">
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

  <!-- Teleport 到 body 避免被 overflow-hidden 裁切 -->
  <Teleport to="body">
    <Transition name="tip-fade">
      <div
        v-if="showWebSearchTip"
        :style="tipStyle"
        class="fixed -translate-x-1/2 -translate-y-full bg-slate-700 text-slate-200 text-[11px] px-2.5 py-1.5 rounded-lg whitespace-nowrap shadow-lg border border-slate-600/50 z-[9999] pointer-events-none"
      >
        该功能需要连接国际互联网
        <div class="absolute -bottom-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-slate-700 rotate-45 border-r border-b border-slate-600/50" />
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.tip-fade-enter-active {
  transition: opacity 0.25s ease, transform 0.25s ease;
}
.tip-fade-leave-active {
  transition: opacity 0.3s ease, transform 0.3s ease;
}
.tip-fade-enter-from {
  opacity: 0;
  transform: translateY(4px);
}
.tip-fade-leave-to {
  opacity: 0;
  transform: translateY(4px);
}
</style>
