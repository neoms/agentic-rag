<script setup lang="ts">
import { computed, ref } from 'vue'
import type { UIMessage, CitationInfo } from '../../types'
import { User, Bot, FileText, ExternalLink } from 'lucide-vue-next'

const props = defineProps<{
  message: UIMessage
}>()

// 活跃的引用弹出信息
const activeCitationKey = ref<string | null>(null)

// 解析消息内容，将 [DocX-ParaY] 替换为 HTML 标记
const renderedContent = computed(() => {
  const content = props.message.content
  if (!content) return ''

  // 替换 [DocX-ParaY] 或 [DocX-ParaY, DocX-ParaY] 格式的引用标记
  // 匹配单个或多个引用： [Doc1-Para2] 或 [Doc1-Para2, Doc3-Para1]
  return content.replace(
    /\[((?:Doc\d+-Para\d+)(?:,\s*Doc\d+-Para\d+)*)\]/g,
    (match, keyGroup: string) => {
      const keys = keyGroup.split(',').map((k: string) => k.trim())
      const badges = keys.map((key: string) => {
        const info = props.message.citations?.[key]
        if (!info) {
          // 无元数据时只显示简短编号
          return `<sup class="citation-marker" data-citation="${key}" title="${key}">[${key.replace('Doc', '').replace('-Para', '-')}]</sup>`
        }
        return `<sup class="citation-marker has-info" data-citation="${key}" title="来源: ${info.filename} · 段落 ${info.para_index}">[${info.doc_index}-${info.para_index}]</sup>`
      })
      return badges.join('')
    }
  )
})

// 获取当前悬停的引文信息
const activeCitationInfo = computed<CitationInfo | null>(() => {
  if (!activeCitationKey.value || !props.message.citations) return null
  return props.message.citations[activeCitationKey.value] || null
})

function showCitation(key: string) {
  activeCitationKey.value = activeCitationKey.value === key ? null : key
}

function hideCitation() {
  activeCitationKey.value = null
}

function handleContentClick(event: MouseEvent) {
  const target = event.target as HTMLElement
  const marker = target.closest('.citation-marker') as HTMLElement
  if (marker) {
    const key = marker.dataset.citation
    if (key) {
      showCitation(key)
    }
  }
}
</script>

<template>
  <div
    :class="[
      'flex gap-3 animate-fade-in',
      message.role === 'user' ? 'justify-end' : 'justify-start'
    ]"
  >
    <!-- AI 头像 -->
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
      <!-- AI 消息：渲染带引文标注的内容 -->
      <div v-if="message.role === 'assistant'" class="message-content">
        <!-- 使用 v-html 渲染带 <sup> 标记的内容 -->
        <span v-html="renderedContent" class="whitespace-pre-wrap break-words" @click.stop="handleContentClick" />
        <!-- 流式输出光标 -->
        <span
          v-if="message.isStreaming"
          class="inline-block w-1.5 h-4 bg-primary-400 ml-0.5 align-middle animate-pulse rounded-sm"
        />
      </div>

      <!-- 用户消息：纯文本 -->
      <p v-else class="whitespace-pre-wrap break-words">{{ message.content }}</p>

      <!-- 流式输出光标（用户消息不会有光标，但保留兜底） -->
      <span
        v-if="message.isStreaming && message.role === 'user'"
        class="inline-block w-1.5 h-4 bg-primary-400 ml-0.5 align-middle animate-pulse rounded-sm"
      />

      <!-- 引文弹出面板 -->
      <div
        v-if="activeCitationInfo && activeCitationKey"
        class="mt-2 pt-2 border-t border-slate-700/50 citation-popover animate-fade-in"
      >
        <div class="flex items-center gap-1.5 mb-1.5">
          <FileText class="w-3 h-3 text-primary-400" />
          <span class="text-[11px] font-semibold text-primary-300">
            {{ activeCitationKey }}
          </span>
          <span class="text-[10px] text-slate-500">·</span>
          <span class="text-[10px] text-slate-400 truncate max-w-[200px]">
            {{ activeCitationInfo.filename }}
          </span>
          <a
            v-if="activeCitationInfo.url"
            :href="activeCitationInfo.url"
            target="_blank"
            rel="noopener"
            class="ml-auto text-[10px] text-blue-400 hover:text-blue-300"
            @click.stop
          >
            <ExternalLink class="w-3 h-3" />
          </a>
        </div>
        <p class="text-[11px] text-slate-400 leading-relaxed whitespace-pre-wrap max-h-[300px] overflow-y-auto">
          {{ activeCitationInfo.paragraph_text }}
        </p>
        <div class="flex items-center gap-2 mt-1">
          <span class="text-[9px] text-slate-600">
            来源类型: {{ {local: '本地文档', web: '网页', knowledge_graph: '知识图谱'}[activeCitationInfo.source_type] || activeCitationInfo.source_type }}
          </span>
          <span class="text-[9px] text-slate-600">
            文档 #{{ activeCitationInfo.doc_index }} · 段落 #{{ activeCitationInfo.para_index }}
          </span>
        </div>
      </div>

      <!-- 来源文档数 -->
      <div
        v-if="message.sources && message.sources.length > 0"
        class="mt-2 pt-2 border-t border-slate-700/50"
      >
        <p class="text-xs text-slate-500">参考 {{ message.sources.length }} 个来源文档</p>
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

<style scoped>
/* 引文标注样式 */
.message-content :deep(.citation-marker) {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 1.6em;
  height: 1.2em;
  padding: 0 0.3em;
  font-size: 0.65em;
  font-weight: 600;
  line-height: 1;
  color: #60a5fa;
  background: rgba(96, 165, 250, 0.1);
  border: 1px solid rgba(96, 165, 250, 0.25);
  border-radius: 3px;
  cursor: pointer;
  vertical-align: super;
  transition: all 0.15s ease;
  margin: 0 0.05em;
  position: relative;
}
.message-content :deep(.citation-marker:hover) {
  background: rgba(96, 165, 250, 0.2);
  border-color: rgba(96, 165, 250, 0.5);
  color: #93c5fd;
}
.message-content :deep(.citation-marker.has-info) {
  color: #34d399;
  background: rgba(52, 211, 153, 0.1);
  border-color: rgba(52, 211, 153, 0.25);
}
.message-content :deep(.citation-marker.has-info:hover) {
  background: rgba(52, 211, 153, 0.2);
  border-color: rgba(52, 211, 153, 0.5);
  color: #6ee7b7;
}

/* 引文弹出面板 */
.citation-popover {
  background: rgba(30, 41, 59, 0.95);
  border: 1px solid rgba(96, 165, 250, 0.2);
  border-radius: 8px;
  padding: 8px 10px;
}

/* 动画 */
:deep(.animate-fade-in) {
  animation: fadeIn 0.2s ease-out;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(-4px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
