<script setup lang="ts">
import { computed } from 'vue'
import type { SourceDocument } from '../../types'
import { ChevronDown, FileText, ExternalLink, Globe, Database } from 'lucide-vue-next'

const props = defineProps<{
  sources: SourceDocument[]
  expanded: boolean
}>()

const emit = defineEmits<{
  toggle: []
}>()

const preview = computed(() => {
  if (props.sources.length === 0) return ''
  return props.sources[0].content.slice(0, 150)
})

function isWebSource(source: SourceDocument): boolean {
  return source.metadata?.source === 'web'
}

function getSourceUrl(source: SourceDocument): string {
  return (source.metadata?.url as string) || ''
}

function getSourceLabel(source: SourceDocument): string {
  if (isWebSource(source)) {
    return (source.metadata?.title as string) || '网页搜索结果'
  }
  return (source.metadata?.filename as string) || '本地文档'
}

function getScoreColor(score: number | null): string {
  if (score === null) return 'text-slate-500'
  if (score >= 0.8) return 'text-emerald-400'
  if (score >= 0.5) return 'text-amber-400'
  return 'text-red-400'
}

function openUrl(url: string) {
  if (url) window.open(url, '_blank', 'noopener')
}
</script>

<template>
  <div class="px-3">
    <!-- 折叠触发器 -->
    <button
      @click="emit('toggle')"
      class="flex items-center gap-2 text-xs text-slate-500 hover:text-slate-300 transition-colors group"
    >
      <FileText class="w-3.5 h-3.5" />
      <span>参考 {{ sources.length }} 个来源</span>
      <ChevronDown
        :class="[
          'w-3.5 h-3.5 transition-transform duration-200',
          expanded && 'rotate-180'
        ]"
      />
      <span v-if="!expanded" class="text-slate-600 truncate max-w-[200px] ml-1">
        {{ preview }}...
      </span>
    </button>

    <!-- 展开面板 -->
    <div
      v-if="expanded"
      class="mt-2 space-y-2 animate-fade-in"
    >
      <div
        v-for="(source, idx) in sources"
        :key="idx"
        :class="[
          'border rounded-lg p-3 transition-colors',
          isWebSource(source)
            ? 'bg-blue-500/5 border-blue-500/20 hover:border-blue-500/40 cursor-pointer'
            : 'bg-slate-800/60 border-slate-700/50'
        ]"
        @click="isWebSource(source) && getSourceUrl(source) && openUrl(getSourceUrl(source))"
      >
        <div class="flex items-center justify-between mb-1.5">
          <div class="flex items-center gap-2 min-w-0">
            <span class="text-[10px] font-medium text-slate-500 bg-slate-700/50 px-1.5 py-0.5 rounded flex-shrink-0">
              #{{ idx + 1 }}
            </span>
            <Globe
              v-if="isWebSource(source)"
              class="w-3 h-3 text-blue-400 flex-shrink-0"
            />
            <Database
              v-else
              class="w-3 h-3 text-emerald-400 flex-shrink-0"
            />
            <span class="text-[10px] font-medium truncate" :class="isWebSource(source) ? 'text-blue-300' : 'text-slate-400'">
              {{ getSourceLabel(source) }}
            </span>
            <span
              v-if="source.score !== null && !isWebSource(source)"
              :class="['text-[10px] font-medium flex-shrink-0', getScoreColor(source.score)]"
            >
              相似度: {{ (source.score * 100).toFixed(1) }}%
            </span>
          </div>
          <ExternalLink
            v-if="isWebSource(source) && getSourceUrl(source)"
            class="w-3 h-3 text-blue-400 flex-shrink-0"
          />
        </div>
        <p class="text-xs text-slate-400 leading-relaxed whitespace-pre-wrap line-clamp-6">
          {{ source.content }}
        </p>
        <!-- 网页来源链接 -->
        <div
          v-if="isWebSource(source) && getSourceUrl(source)"
          class="mt-2 pt-2 border-t border-blue-500/10"
        >
          <a
            :href="getSourceUrl(source)"
            target="_blank"
            rel="noopener"
            class="text-[10px] text-blue-400 hover:text-blue-300 hover:underline truncate block"
            @click.stop
          >
            {{ getSourceUrl(source) }}
          </a>
        </div>
        <!-- 本地文档元数据 -->
        <div
          v-if="!isWebSource(source) && Object.keys(source.metadata).length > 0"
          class="mt-2 pt-2 border-t border-slate-700/30 flex flex-wrap gap-1.5"
        >
          <span
            v-for="(val, key) in source.metadata"
            :key="key"
            class="text-[10px] text-slate-600 bg-slate-700/30 px-1.5 py-0.5 rounded"
          >
            {{ key }}: {{ val }}
          </span>
        </div>
      </div>
    </div>
  </div>
</template>
