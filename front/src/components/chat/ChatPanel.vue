<script setup lang="ts">
import { ref, watch, nextTick } from 'vue'
import { Bot, ShieldCheck, ShieldAlert } from 'lucide-vue-next'
import type { UIMessage } from '../../types'
import MessageBubble from './MessageBubble.vue'
import ChatInput from './ChatInput.vue'
import SourcePanel from './SourcePanel.vue'
import AgentPathBadge from './AgentPathBadge.vue'

const props = defineProps<{
  messages: UIMessage[]
  sending: boolean
  error: string | null
  enableWebSearch: boolean
  enableReflection: boolean
  enableRerank: boolean
  enableGradeDocuments: boolean
  enableTransformQuery: boolean
  isStreaming: boolean
  hallucinationResults: Record<string, { passed: boolean; faithfulness: number }>
}>()

const emit = defineEmits<{
  send: [query: string]
  'update:enableWebSearch': [val: boolean]
  'update:enableReflection': [val: boolean]
  'update:enableRerank': [val: boolean]
  'update:enableGradeDocuments': [val: boolean]
  'update:enableTransformQuery': [val: boolean]
}>()

const messagesContainer = ref<HTMLElement | null>(null)
const expandedSources = ref<string | null>(null)

function scrollToBottom() {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

watch(
  () => props.messages.length,
  () => scrollToBottom()
)

watch(
  () => props.messages[props.messages.length - 1]?.content,
  () => scrollToBottom()
)

function toggleSources(msgId: string) {
  expandedSources.value = expandedSources.value === msgId ? null : msgId
}

function handleSend(query: string) {
  emit('send', query)
  scrollToBottom()
}
</script>

<template>
  <div class="flex flex-col h-full">
    <!-- 消息列表 -->
    <div
      ref="messagesContainer"
      class="flex-1 overflow-y-auto px-4 py-6 space-y-4"
    >
      <!-- 欢迎界面 -->
      <div
        v-if="messages.length === 0"
        class="flex flex-col items-center justify-center h-full text-center px-4"
      >
        <div class="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center mb-5 shadow-xl shadow-primary-500/20">
          <Bot class="w-8 h-8 text-white" />
        </div>
        <h2 class="text-lg font-bold gradient-text mb-2">Agentic RAG</h2>
        <p class="text-sm text-slate-500 max-w-sm">
          基于 LangGraph 的多策略检索增强生成系统，支持自反思 Agent 和工具调用
        </p>
        <div class="mt-6 grid grid-cols-3 gap-2 max-w-sm">
          <div
            v-for="item in ['检索增强', 'Agent 反思', '流式输出']"
            :key="item"
            class="px-3 py-1.5 rounded-lg border border-slate-700%0 text-xs text-slate-400"
          >
            {{ item }}
          </div>
        </div>
      </div>

      <!-- 错误提示 -->
      <div
        v-if="error"
        class="mx-4 bg-red-400/10 border border-red-400/20 rounded-lg px-4 py-2.5 text-sm text-red-400 animate-fade-in"
      >
        {{ error }}
      </div>

      <!-- 消息列表 -->
      <template v-for="msg in messages" :key="msg.id">
        <MessageBubble :message="msg" />

        <!-- Agent 路径可视化 -->
        <AgentPathBadge
          v-if="msg.agent_path && msg.agent_path.length > 0 && msg.role === 'assistant'"
          :path="msg.agent_path"
          :reflectionCount="msg.reflection_count ?? 0"
        />

        <!-- 幻觉检测结果（含忠实度分数） -->
        <div
          v-if="msg.role === 'assistant' && msg.isStreaming === false && msg.content.length > 0"
          class="px-3 pb-1"
        >
          <span
            v-if="hallucinationResults[msg.id]?.passed === true"
            class="inline-flex items-center gap-1 text-[10px] font-medium px-1.5 py-0.5 rounded border bg-emerald-400/10 text-emerald-400 border-emerald-400/20"
          >
            <ShieldCheck class="w-3 h-3" />
            知识忠实度 {{ hallucinationResults[msg.id].faithfulness }}%
          </span>
          <span
            v-else-if="hallucinationResults[msg.id]?.passed === false"
            class="inline-flex items-center gap-1 text-[10px] font-medium px-1.5 py-0.5 rounded border bg-red-400/10 text-red-400 border-red-400/20"
          >
            <ShieldAlert class="w-3 h-3" />
            知识忠实度 {{ hallucinationResults[msg.id].faithfulness }}%
          </span>
        </div>

        <!-- 来源文档面板 -->
        <SourcePanel
          v-if="msg.sources && msg.sources.length > 0 && msg.role === 'assistant'"
          :sources="msg.sources"
          :expanded="expandedSources === msg.id"
          @toggle="toggleSources(msg.id)"
        />
      </template>
    </div>

    <!-- 输入区 -->
    <ChatInput
      :sending="sending"
      :enableWebSearch="enableWebSearch"
      :enableReflection="enableReflection"
      :enableRerank="enableRerank"
      :enableGradeDocuments="enableGradeDocuments"
      :enableTransformQuery="enableTransformQuery"
      @send="handleSend"
      @update:enableWebSearch="(v) => emit('update:enableWebSearch', v)"
      @update:enableReflection="(v) => emit('update:enableReflection', v)"
      @update:enableRerank="(v) => emit('update:enableRerank', v)"
      @update:enableGradeDocuments="(v) => emit('update:enableGradeDocuments', v)"
      @update:enableTransformQuery="(v) => emit('update:enableTransformQuery', v)"
    />
  </div>
</template>
