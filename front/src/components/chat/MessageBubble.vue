<script setup lang="ts">
import type { UIMessage } from '../../types'
import { User, Bot } from 'lucide-vue-next'

defineProps<{
  message: UIMessage
}>()
</script>

<template>
  <div
    :class="[
      'flex gap-3 animate-fade-in',
      message.role === 'user' ? 'justify-end' : 'justify-start'
    ]"
  >
    <!-- 用户头像 -->
    <div
      v-if="message.role === 'assistant'"
      class="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center flex-shrink-0 shadow-lg shadow-primary-500/20"
    >
      <Bot class="w-4 h-4 text-white" />
    </div>

    <!-- 消息气泡 -->
    <div
      :class="[
        'max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed',
        message.role === 'user'
          ? 'bg-gradient-to-r from-primary-500 to-primary-600 text-white rounded-br-md shadow-lg shadow-primary-500/15'
          : 'bg-slate-800/80 text-slate-200 rounded-bl-md border border-slate-700/50'
      ]"
    >
      <!-- 内容（支持换行） -->
      <p class="whitespace-pre-wrap break-words">{{ message.content }}</p>

      <!-- 流式输出光标 -->
      <span
        v-if="message.isStreaming"
        class="inline-block w-1.5 h-4 bg-primary-400 ml-0.5 align-middle animate-pulse rounded-sm"
      />

      <!-- 来源文档数 -->
      <div
        v-if="message.sources && message.sources.length > 0"
        class="mt-2 pt-2 border-t border-slate-700/50"
      >
        <p class="text-xs text-slate-500">参考文档: {{ message.sources.length }} 个</p>
      </div>
    </div>

    <!-- 用户头像 -->
    <div
      v-if="message.role === 'user'"
      class="w-8 h-8 rounded-lg bg-slate-600 flex items-center justify-center flex-shrink-0"
    >
      <User class="w-4 h-4 text-slate-300" />
    </div>
  </div>
</template>
