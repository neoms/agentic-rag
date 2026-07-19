<script setup lang="ts">
import { useChat } from '../composables/useChat'
import ChatPanel from '../components/chat/ChatPanel.vue'
import SessionHistory from '../components/chat/SessionHistory.vue'

const {
  messages,
  sending,
  mode,
  error,
  enableWebSearch,
  enableReflection,
  isStreaming,
  send,
} = useChat()
</script>

<template>
  <div class="flex h-full">
    <!-- 对话区 -->
    <div class="flex-1 flex flex-col min-w-0">
      <ChatPanel
        :messages="messages"
        :sending="sending"
        :mode="mode"
        :error="error"
        :enableWebSearch="enableWebSearch"
        :enableReflection="enableReflection"
        :isStreaming="isStreaming"
        @send="send"
        @update:mode="mode = $event"
        @update:enableWebSearch="enableWebSearch = $event"
        @update:enableReflection="enableReflection = $event"
      />
    </div>

    <!-- 会话历史侧栏 -->
    <SessionHistory />
  </div>
</template>
