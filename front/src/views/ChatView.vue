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
  enableRerank,
  enableGradeDocuments,
  enableTransformQuery,
  isStreaming,
  hallucinationResults,
  send,
  loadSession,
  newSession,
  deleteSession,
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
        :enableRerank="enableRerank"
        :enableGradeDocuments="enableGradeDocuments"
        :enableTransformQuery="enableTransformQuery"
        :isStreaming="isStreaming"
        :hallucinationResults="hallucinationResults"
        @send="send"
        @update:enableWebSearch="enableWebSearch.value = $event"
        @update:enableReflection="enableReflection.value = $event"
        @update:enableRerank="enableRerank.value = $event"
        @update:enableGradeDocuments="enableGradeDocuments.value = $event"
        @update:enableTransformQuery="enableTransformQuery.value = $event"
      />
    </div>

    <!-- 右侧栏：会话历史 -->
    <div class="border-l border-slate-700/50 bg-slate-900/40 w-64 flex-shrink-0 flex flex-col h-full">
      <SessionHistory
        :sessions="sessions"
        :activeSessionId="sessionId"
        :loading="historyLoading"
        @select-session="onSelectSession"
        @new-session="onNewSession"
        @delete-session="deleteSession"
      />
    </div>
  </div>
</template>
