<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { History, MessageSquare, Loader2, Clock } from 'lucide-vue-next'
import type { ChatHistoryMessage } from '../../types'
import { getChatHistory } from '../../api/chat'

const sessions = ref<{ session_id: string; preview: string; updated_at: Date }[]>([])
const activeSession = ref<string | null>(null)
const messages = ref<ChatHistoryMessage[]>([])
const loading = ref(false)
const expanded = ref(false)

onMounted(() => {
  loadHistory('default')
})

async function loadHistory(sessionId: string) {
  loading.value = true
  activeSession.value = sessionId
  const result = await getChatHistory(sessionId)
  if (result) {
    messages.value = result.messages
    if (sessions.value.length === 0) {
      sessions.value = [{
        session_id: sessionId,
        preview: result.messages.length > 0
          ? result.messages[result.messages.length - 1].content.slice(0, 30) + '...'
          : '空会话',
        updated_at: new Date(),
      }]
    }
  }
  loading.value = false
}

function formatDate(date: Date): string {
  const now = new Date()
  const diff = now.getTime() - date.getTime()
  if (diff < 60000) return '刚刚'
  if (diff < 3600000) return `${Math.floor(diff / 60000)} 分钟前`
  return date.toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
}
</script>

<template>
  <div class="border-l border-slate-700/50 bg-slate-900/40 w-64 flex-shrink-0 flex flex-col h-full">
    <div class="px-4 py-3 border-b border-slate-700/50 flex items-center justify-between">
      <div class="flex items-center gap-2">
        <History class="w-4 h-4 text-slate-500" />
        <span class="text-xs font-medium text-slate-400">会话历史</span>
      </div>
    </div>

    <div class="flex-1 overflow-y-auto p-3 space-y-2">
      <template v-if="loading">
        <div class="flex items-center justify-center py-8">
          <Loader2 class="w-4 h-4 text-slate-600 animate-spin" />
        </div>
      </template>

      <template v-else>
        <div
          v-for="msg in messages"
          :key="msg.content.slice(0, 20)"
          class="p-2 rounded-lg hover:bg-slate-800/40 transition-colors cursor-pointer"
        >
          <div class="flex items-start gap-2">
            <MessageSquare
              :class="[
                'w-3 h-3 mt-0.5 flex-shrink-0',
                msg.role === 'user' ? 'text-primary-400' : 'text-slate-500'
              ]"
            />
            <div class="min-w-0">
              <p class="text-[11px] text-slate-500 font-medium mb-0.5">
                {{ msg.role === 'user' ? '用户' : '助手' }}
              </p>
              <p class="text-xs text-slate-400 truncate">
                {{ msg.content.slice(0, 50) }}
              </p>
            </div>
          </div>
        </div>

        <div
          v-if="messages.length === 0"
          class="text-center py-8"
        >
          <Clock class="w-5 h-5 text-slate-600 mx-auto mb-2" />
          <p class="text-xs text-slate-600">暂无历史消息</p>
        </div>
      </template>
    </div>
  </div>
</template>
