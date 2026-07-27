<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import {
  Search, ListOrdered, Sparkles, Shuffle, Globe,
  Brain, CheckCircle, FileText, Network,
} from 'lucide-vue-next'

const props = defineProps<{
  completedNodes: string[]
  currentNode: string | null
  enableRerank: boolean
  enableGradeDocuments: boolean
  enableTransformQuery: boolean
  enableWebSearch: boolean
  enableReflection: boolean
}>()

// 使用内部 ref 跟踪，确保每次 props 变化都强制重新计算
const _completedNodes = ref<string[]>([])
const _currentNode = ref<string | null>(null)
watch(() => props.completedNodes, v => { _completedNodes.value = [...v] }, { immediate: true })
watch(() => _currentNode.value, v => { _currentNode.value = v }, { immediate: true })

const completedSet = computed(() => new Set(_completedNodes.value))

function nodeStatus(nodeId: string): 'active' | 'done' | 'disabled' | 'pending' {
  const enabledMap: Record<string, boolean> = {
    retrieve: true,
    rerank_documents: props.enableRerank,
    grade_documents: props.enableGradeDocuments,
    web_search: props.enableWebSearch,
    transform_query: props.enableTransformQuery,
    generate: true,
    check_hallucination: props.enableReflection,
  }
  if (!enabledMap[nodeId]) return 'disabled'
  // 已完成优先于活跃（节点 end 后在 start 之前会保持 done 状态）
  if (completedSet.value.has(nodeId)) return 'done'
  if (_currentNode.value === nodeId) return 'active'
  return 'pending'
}

const statusColors: Record<string, string> = {
  active:   'border-primary-400 text-primary-400 bg-primary-400/10 shadow-[0_0_12px_rgba(96,165,250,0.3)]',
  done:     'border-emerald-400/40 text-emerald-400 bg-emerald-400/10',
  disabled: 'border-slate-700/30 text-slate-600 bg-transparent',
  pending:  'border-slate-700/30 text-slate-600 bg-transparent',
}

function lineColor(nodeId: string): string {
  const s = nodeStatus(nodeId)
  if (s === 'active') return 'bg-primary-400'
  if (s === 'done') return 'bg-emerald-400/60'
  return 'bg-slate-700/20'
}
</script>

<template>
  <div class="border-t border-slate-700/50 px-3 py-3 space-y-1 flex-shrink-0">
    <div class="flex items-center gap-1.5 mb-2">
      <Network class="w-3 h-3 text-slate-500" />
      <span class="text-[10px] font-medium text-slate-500">Agent 流程</span>
    </div>

    <div class="relative flex flex-col items-stretch gap-0">

      <!-- retrieve -->
      <div class="flex items-center gap-2">
        <div
          :class="[
            'flex items-center gap-1.5 px-2 py-1 rounded-lg border text-[11px] transition-all duration-300',
            statusColors[nodeStatus('retrieve')],
            nodeStatus('retrieve') === 'active' ? 'animate-pulse' : '',
          ]"
        >
          <Search class="w-3 h-3 flex-shrink-0" />
          检索
          <CheckCircle v-if="nodeStatus('retrieve') === 'done'" class="w-2.5 h-2.5 flex-shrink-0" />
        </div>
      </div>

      <!-- rerank -->
      <div class="flex items-center gap-2">
        <div class="w-3 flex flex-col items-center">
          <div :class="['w-0.5 h-3', lineColor('rerank_documents')]" />
          <div :class="['w-1.5 h-1.5 rounded-full flex-shrink-0', completedSet.has('rerank_documents') ? 'bg-emerald-400' : 'bg-slate-700']" />
          <div :class="['w-0.5 h-3', lineColor('rerank_documents')]" />
        </div>
        <div
          :class="[
            'flex items-center gap-1.5 px-2 py-1 rounded-lg border text-[11px] transition-all duration-300',
            statusColors[nodeStatus('rerank_documents')],
            nodeStatus('rerank_documents') === 'active' ? 'animate-pulse' : '',
            !enableRerank ? 'line-through' : '',
          ]"
        >
          <ListOrdered class="w-3 h-3 flex-shrink-0" />
          重排序
          <CheckCircle v-if="nodeStatus('rerank_documents') === 'done'" class="w-2.5 h-2.5 flex-shrink-0" />
        </div>
      </div>

      <!-- grade_documents -->
      <div class="flex items-center gap-2">
        <div class="w-3 flex flex-col items-center">
          <div :class="['w-0.5 h-3', lineColor('grade_documents')]" />
          <div :class="['w-1.5 h-1.5 rounded-full flex-shrink-0', completedSet.has('grade_documents') ? 'bg-emerald-400' : 'bg-slate-700']" />
          <div :class="['w-0.5 h-3', lineColor('grade_documents')]" />
        </div>
        <div
          :class="[
            'flex items-center gap-1.5 px-2 py-1 rounded-lg border text-[11px] transition-all duration-300',
            statusColors[nodeStatus('grade_documents')],
            nodeStatus('grade_documents') === 'active' ? 'animate-pulse' : '',
            !enableGradeDocuments ? 'line-through' : '',
          ]"
        >
          <Sparkles class="w-3 h-3 flex-shrink-0" />
          文档评估
          <CheckCircle v-if="nodeStatus('grade_documents') === 'done'" class="w-2.5 h-2.5 flex-shrink-0" />
        </div>
      </div>

      <!-- 不相关时分支标记 -->
      <div class="flex items-start gap-2">
        <div class="w-3 flex flex-col items-center min-h-0">
          <div :class="['w-0.5 flex-1 min-h-[6px]', lineColor('grade_documents')]" />
        </div>
        <span
          :class="[
            'text-[9px] px-1.5 py-0.5 rounded border',
            completedSet.has('web_search') || completedSet.has('transform_query')
              ? 'border-orange-400/30 text-orange-400 bg-orange-400/10'
              : 'border-slate-700/30 text-slate-600',
          ]"
        >不相关时</span>
      </div>

      <!-- web_search -->
      <div class="flex items-start gap-2">
        <div class="w-3 flex flex-col items-center">
          <div class="w-0.5 h-2 flex-shrink-0 bg-slate-700/20" />
          <div class="flex items-center gap-0.5">
            <div :class="['w-0.5 h-3', lineColor('web_search')]" />
            <div :class="['w-1.5 h-1.5 rounded-full flex-shrink-0', completedSet.has('web_search') ? 'bg-emerald-400' : 'bg-slate-700']" />
          </div>
        </div>
        <div
          :class="[
            'flex items-center gap-1.5 px-2 py-1 rounded-lg border text-[11px] transition-all duration-300',
            nodeStatus('web_search') === 'active' ? 'animate-pulse' : '',
            enableWebSearch
              ? statusColors[nodeStatus('web_search')]
              : 'border-slate-700/20 text-slate-600 bg-transparent line-through',
          ]"
        >
          <Globe class="w-3 h-3 flex-shrink-0" />
          联网搜索
          <CheckCircle v-if="nodeStatus('web_search') === 'done'" class="w-2.5 h-2.5 flex-shrink-0" />
        </div>
      </div>

      <!-- transform_query -->
      <div class="flex items-start gap-2">
        <div class="w-3 flex flex-col items-center">
          <div class="w-0.5 h-2 flex-shrink-0 bg-slate-700/20" />
          <div class="flex items-center gap-0.5">
            <div :class="['w-0.5 h-3', lineColor('transform_query')]" />
            <div :class="['w-1.5 h-1.5 rounded-full flex-shrink-0', completedSet.has('transform_query') ? 'bg-emerald-400' : 'bg-slate-700']" />
          </div>
        </div>
        <div
          :class="[
            'flex items-center gap-1.5 px-2 py-1 rounded-lg border text-[11px] transition-all duration-300',
            nodeStatus('transform_query') === 'active' ? 'animate-pulse' : '',
            enableTransformQuery
              ? statusColors[nodeStatus('transform_query')]
              : 'border-slate-700/20 text-slate-600 bg-transparent line-through',
          ]"
        >
          <Shuffle class="w-3 h-3 flex-shrink-0" />
          查询重写
          <CheckCircle v-if="nodeStatus('transform_query') === 'done'" class="w-2.5 h-2.5 flex-shrink-0" />
        </div>
      </div>

      <!-- generate -->
      <div class="flex items-center gap-2">
        <div class="w-3 flex flex-col items-center">
          <div :class="['w-0.5 h-3', lineColor('generate')]" />
          <div :class="['w-1.5 h-1.5 rounded-full flex-shrink-0', completedSet.has('generate') ? 'bg-emerald-400' : 'bg-slate-700']" />
          <div :class="['w-0.5 h-3', lineColor('generate')]" />
        </div>
        <div
          :class="[
            'flex items-center gap-1.5 px-2 py-1 rounded-lg border text-[11px] transition-all duration-300',
            statusColors[nodeStatus('generate')],
            nodeStatus('generate') === 'active' ? 'animate-pulse' : '',
          ]"
        >
          <FileText class="w-3 h-3 flex-shrink-0" />
          生成回答
          <CheckCircle v-if="nodeStatus('generate') === 'done'" class="w-2.5 h-2.5 flex-shrink-0" />
        </div>
      </div>

      <!-- check_hallucination -->
      <div class="flex items-center gap-2">
        <div class="w-3 flex flex-col items-center">
          <div :class="['w-0.5 h-3', lineColor('check_hallucination')]" />
          <div :class="['w-1.5 h-1.5 rounded-full flex-shrink-0', completedSet.has('check_hallucination') ? 'bg-emerald-400' : 'bg-slate-700']" />
          <div :class="['w-0.5 h-3', lineColor('check_hallucination')]" />
        </div>
        <div
          :class="[
            'flex items-center gap-1.5 px-2 py-1 rounded-lg border text-[11px] transition-all duration-300',
            nodeStatus('check_hallucination') === 'active' ? 'animate-pulse' : '',
            enableReflection
              ? statusColors[nodeStatus('check_hallucination')]
              : 'border-slate-700/20 text-slate-600 bg-transparent line-through',
          ]"
        >
          <Brain class="w-3 h-3 flex-shrink-0" />
          幻觉检测
          <CheckCircle v-if="nodeStatus('check_hallucination') === 'done'" class="w-2.5 h-2.5 flex-shrink-0" />
        </div>
      </div>

      <!-- END -->
      <div class="flex items-center gap-2">
        <div class="w-3 flex flex-col items-center">
          <div :class="['w-0.5 h-3', completedSet.has('__end__') ? 'bg-emerald-400' : 'bg-slate-700/20']" />
          <div :class="['w-2 h-2 rounded-full flex-shrink-0', completedSet.has('__end__') ? 'bg-emerald-400' : 'bg-slate-700/50']" />
        </div>
        <span :class="['text-[10px] font-medium', completedSet.has('__end__') ? 'text-emerald-400' : 'text-slate-600']">
          完成
        </span>
      </div>
    </div>
  </div>
</template>
