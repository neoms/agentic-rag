<script setup lang="ts">
import { computed } from 'vue'
import { ArrowRight, RotateCcw } from 'lucide-vue-next'

const props = defineProps<{
  path: string[]
  reflectionCount: number
}>()

const nodeColors: Record<string, string> = {
  retrieve: 'bg-blue-400/10 text-blue-400 border-blue-400/20',
  grade_documents: 'bg-amber-400/10 text-amber-400 border-amber-400/20',
  transform_query: 'bg-purple-400/10 text-purple-400 border-purple-400/20',
  generate: 'bg-emerald-400/10 text-emerald-400 border-emerald-400/20',
  check_hallucination: 'bg-pink-400/10 text-pink-400 border-pink-400/20',
  web_search: 'bg-cyan-400/10 text-cyan-400 border-cyan-400/20',
}

const nodeLabels: Record<string, string> = {
  retrieve: '检索',
  grade_documents: '评估相关性',
  transform_query: '查询重写',
  generate: '生成回答',
  check_hallucination: '幻觉检测',
  web_search: '联网搜索',
}

const uniquePath = computed(() => {
  const seen = new Set<string>()
  return props.path.filter(n => {
    if (seen.has(n)) return false
    seen.add(n)
    return true
  })
})
</script>

<template>
  <div class="px-3 py-2">
    <div class="flex items-center gap-2 flex-wrap">
      <span class="text-[10px] text-slate-600 font-medium">Agent 路径</span>
      <template v-for="(node, idx) in uniquePath" :key="idx">
        <span
          :class="[
            'text-[10px] font-medium px-1.5 py-0.5 rounded border',
            nodeColors[node] || 'bg-slate-400/10 text-slate-400 border-slate-400/20'
          ]"
        >
          {{ nodeLabels[node] || node }}
        </span>
        <ArrowRight
          v-if="idx < uniquePath.length - 1"
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
