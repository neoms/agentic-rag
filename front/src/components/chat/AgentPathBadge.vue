<script setup lang="ts">
import { ArrowRight, RotateCcw } from 'lucide-vue-next'

const props = defineProps<{
  path: string[]
  reflectionCount: number
}>()

const nodeColors: Record<string, string> = {
  retrieve: 'bg-blue-400/10 text-blue-400 border-blue-400/20',
  rerank_documents: 'bg-violet-400/10 text-violet-400 border-violet-400/20',
  rerank: 'bg-violet-400/10 text-violet-400 border-violet-400/20',
  grade_documents: 'bg-amber-400/10 text-amber-400 border-amber-400/20',
  transform_query: 'bg-purple-400/10 text-purple-400 border-purple-400/20',
  generate: 'bg-emerald-400/10 text-emerald-400 border-emerald-400/20',
  check_hallucination: 'bg-pink-400/10 text-pink-400 border-pink-400/20',
  web_search: 'bg-cyan-400/10 text-cyan-400 border-cyan-400/20',
  bm25_retrieve: 'bg-teal-400/10 text-teal-400 border-teal-400/20',
  hyde_retrieve: 'bg-rose-400/10 text-rose-400 border-rose-400/20',
  multi_query_retrieve: 'bg-indigo-400/10 text-indigo-400 border-indigo-400/20',
  merge_retrieval: 'bg-sky-400/10 text-sky-400 border-sky-400/20',
  decide_strategy: 'bg-slate-400/10 text-slate-400 border-slate-400/20',
  analyze_kg_intent: 'bg-orange-400/10 text-orange-400 border-orange-400/20',
  kg_retrieve: 'bg-amber-500/10 text-amber-500 border-amber-500/20',
}

const nodeLabels: Record<string, string> = {
  retrieve: '检索',
  rerank_documents: '重排序',
  rerank: '重排序',
  grade_documents: '评估相关性',
  transform_query: '查询重写',
  generate: '生成回答',
  check_hallucination: '幻觉检测',
  web_search: '联网搜索',
  bm25_retrieve: 'BM25 检索',
  hyde_retrieve: 'HyDE 检索',
  multi_query_retrieve: '多查询检索',
  merge_retrieval: '合并检索',
  decide_strategy: '策略决策',
  analyze_kg_intent: '图谱意图分析',
  kg_retrieve: '图谱检索',
}

const defaultColor = 'bg-slate-400/10 text-slate-400 border-slate-400/20'

function getNodeBaseName(node: string): string {
  return node.replace(/\s*\(.*\)\s*/, '')
}

function getNodeColor(node: string): string {
  if (nodeColors[node]) return nodeColors[node]
  const base = getNodeBaseName(node)
  if (nodeColors[base]) return nodeColors[base]
  return defaultColor
}

function getNodeLabel(node: string): string {
  if (nodeLabels[node]) return nodeLabels[node]
  const base = getNodeBaseName(node)
  if (nodeLabels[base]) return nodeLabels[base]
  return node
}
</script>

<template>
  <div class="px-3 py-2">
    <div class="flex items-center gap-2 flex-wrap">
      <span class="text-[10px] text-slate-600 font-medium">Agent 路径</span>
      <template v-for="(node, idx) in path" :key="idx">
        <span
          :class="[
            'text-[10px] font-medium px-1.5 py-0.5 rounded border',
            getNodeColor(node)
          ]"
        >
          {{ getNodeLabel(node) }}
        </span>
        <ArrowRight
          v-if="idx < path.length - 1"
          class="w-3 h-3 text-slate-600"
        />
      </template>

      <span
        v-if="reflectionCount > 0"
        class="flex items-center gap-1 text-[10px] text-slate-500"
      >
        <RotateCcw class="w-3 h-3" />
        {{ reflectionCount }} 轮反思
      </span>
    </div>
  </div>
</template>
