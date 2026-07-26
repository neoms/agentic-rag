<script setup lang="ts">
import { ref } from 'vue'
import { useChat } from '../composables/useChat'
import ChatPanel from '../components/chat/ChatPanel.vue'
import SessionHistory from '../components/chat/SessionHistory.vue'

const {
  messages,
  sending,
  sessionId,
  error,
  sessions,
  enableWebSearch,
  enableReflection,
  isStreaming,
  send,
  loadSession,
  newSession,
} = useChat()

const historyLoading = ref(false)

async function onSelectSession(id: string) {
  historyLoading.value = true
  await loadSession(id)
  historyLoading.value = false
}

function onNewSession() {
  newSession()
}
</script>

<template>
  <div class="flex h-full">
    <!-- 对话区 -->
    <div class="flex-1 flex flex-col min-w-0">
      <ChatPanel
        :messages="messages"
        :sending="sending"
        :error="error"
        :enableWebSearch="enableWebSearch"
        :enableReflection="enableReflection"
        :isStreaming="isStreaming"
        @send="send"
        @update:enableWebSearch="enableWebSearch = $event"
        @update:enableReflection="enableReflection = $event"
      />
    </div>

    <!-- 会话历史侧栏 -->
    <SessionHistory
      :sessions="sessions"
      :activeSessionId="sessionId"
      :loading="historyLoading"
      @select-session="onSelectSession"
      @new-session="onNewSession"
    />
  </div>
</template>
