<script setup lang="ts">
import { ref } from 'vue'
import { History, MessageSquare, Plus, Clock, Loader2, Trash2 } from 'lucide-vue-next'
import type { SessionSummary } from '../../composables/useChat'

defineProps<{
  sessions: SessionSummary[]
  activeSessionId: string
  loading: boolean
}>()

const emit = defineEmits<{
  'select-session': [sessionId: string]
  'new-session': []
  'delete-session': [sessionId: string]
}>()

const confirmDeleteId = ref<string | null>(null)

function requestDelete(id: string, e: Event) {
  e.stopPropagation()
  confirmDeleteId.value = id
}

function confirmDelete(id: string) {
  emit('delete-session', id)
  confirmDeleteId.value = null
}

function cancelDelete(e: Event) {
  e.stopPropagation()
  confirmDeleteId.value = null
}

function formatTime(ts: number): string {
  const now = Date.now()
  const diff = now - ts
  if (diff < 60000) return '刚刚'
  if (diff < 3600000) return `${Math.floor(diff / 60000)} 分钟前`
  if (diff < 86400000) return `${Math.floor(diff / 3600000)} 小时前`
  const d = new Date(ts)
  return `${d.getMonth() + 1}/${d.getDate()} ${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}`
}
</script>

<template>
  <div class="border-l border-slate-700/50 bg-slate-900/40 w-64 flex-shrink-0 flex flex-col h-full">
    <!-- 标题栏 + 新建按钮 -->
    <div class="px-4 py-3 border-b border-slate-700/50 flex items-center justify-between">
      <div class="flex items-center gap-2">
        <History class="w-4 h-4 text-slate-500" />
        <span class="text-xs font-medium text-slate-400">会话历史</span>
      </div>
      <button
        @click="emit('new-session')"
        class="p-1 rounded-md hover:bg-slate-700/50 text-slate-500 hover:text-slate-300 transition-colors"
        title="新建会话"
      >
        <Plus class="w-3.5 h-3.5" />
      </button>
    </div>

    <!-- 会话列表 -->
    <div class="flex-1 overflow-y-auto p-2 space-y-1">
      <template v-if="loading">
        <div class="flex items-center justify-center py-8">
          <Loader2 class="w-4 h-4 text-slate-600 animate-spin" />
        </div>
      </template>

      <template v-else-if="sessions.length > 0">
        <div
          v-for="session in sessions"
          :key="session.session_id"
          @click="emit('select-session', session.session_id)"
          class="p-2.5 rounded-lg cursor-pointer transition-colors group"
          :class="[
            activeSessionId === session.session_id
              ? 'bg-primary-500/10 border border-primary-500/20'
              : 'hover:bg-slate-800/40 border border-transparent'
          ]"
        >
          <div class="flex items-start gap-2">
            <MessageSquare
              class="w-3 h-3 mt-0.5 flex-shrink-0"
              :class="activeSessionId === session.session_id ? 'text-primary-400' : 'text-slate-500'"
            />
            <div class="min-w-0 flex-1">
              <p
                class="text-xs truncate"
                :class="activeSessionId === session.session_id ? 'text-slate-200' : 'text-slate-400'"
              >
                {{ session.preview }}
              </p>
              <div class="flex items-center gap-2 mt-1">
                <span class="text-[10px] text-slate-600">
                  {{ session.message_count }} 条消息
                </span>
                <span class="text-[10px] text-slate-600 flex items-center gap-0.5">
                  <Clock class="w-2.5 h-2.5" />
                  {{ formatTime(session.updated_at) }}
                </span>
              </div>
            </div>

            <!-- 删除按钮 -->
            <template v-if="confirmDeleteId === session.session_id">
              <div class="flex items-center gap-1 flex-shrink-0" @click.stop>
                <button
                  @click="confirmDelete(session.session_id)"
                  class="p-0.5 rounded text-[10px] text-red-400 hover:bg-red-400/10 transition-colors"
                  title="确认删除"
                >
                  确认
                </button>
                <button
                  @click="cancelDelete"
                  class="p-0.5 rounded text-[10px] text-slate-500 hover:text-slate-300 transition-colors"
                  title="取消"
                >
                  取消
                </button>
              </div>
            </template>
            <button
              v-else
              @click="requestDelete(session.session_id, $event)"
              class="p-0.5 rounded text-slate-600 hover:text-red-400 hover:bg-red-400/10 transition-colors opacity-0 group-hover:opacity-100 flex-shrink-0"
              title="删除会话"
            >
              <Trash2 class="w-3 h-3" />
            </button>
          </div>
        </div>
      </template>

      <template v-else>
        <div class="text-center py-8">
          <Clock class="w-5 h-5 text-slate-600 mx-auto mb-2" />
          <p class="text-xs text-slate-600">暂无会话</p>
          <p class="text-[10px] text-slate-600/60 mt-1">发送消息后自动保存</p>
        </div>
      </template>
    </div>
  </div>
</template>
