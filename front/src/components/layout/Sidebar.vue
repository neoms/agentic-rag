<script setup lang="ts">
import { useRoute } from 'vue-router'
import { MessageCircle, Database, Search, ListOrdered, Sparkles, Shuffle, Globe, Brain, FileText, Network } from 'lucide-vue-next'
import * as flow from '../../composables/agentFlowState'

const route = useRoute()

const navItems = [
  { path: '/', label: '对话', icon: MessageCircle },
  { path: '/documents', label: '知识库', icon: Database },
]

// ── 流程图节点定义 ──
const NODE_DEFS = [
  { id: 'retrieve',             label: '检索',     icon: Search },
  { id: 'rerank_documents',     label: '重排序',   icon: ListOrdered, enableKey: 'enableRerank' },
  { id: 'grade_documents',      label: '文档评估', icon: Sparkles,   enableKey: 'enableGradeDocuments' },
] as const

const BRANCH_DEFS = [
  { id: 'web_search',      label: '联网搜索', icon: Globe,   enableKey: 'enableWebSearch' },
  { id: 'transform_query', label: '查询重写', icon: Shuffle, enableKey: 'enableTransformQuery' },
] as const

const TAIL_DEFS = [
  { id: 'generate',             label: '生成回答', icon: FileText },
  { id: 'check_hallucination',  label: '幻觉检测', icon: Brain,   enableKey: 'enableReflection' },
] as const

// ── 状态函数（直接读 flow ref，零 props） ──
function isEnabled(key?: string): boolean {
  if (!key) return true
  return (flow as any)[key].value as boolean
}

function isCompleted(id: string): boolean {
  return flow.completedNodes.value.includes(id)
}

function isCurrent(id: string): boolean {
  return flow.currentNode.value === id
}

function nodeClass(id: string, extraKey?: string): string {
  const base = 'flex items-center gap-1.5 px-2 py-1 rounded-lg border text-[11px] transition-all duration-300'
  if (extraKey && !isEnabled(extraKey))
    return `${base} border-slate-700/20 text-slate-600 bg-transparent line-through`
  if (isCompleted(id))
    return `${base} border-emerald-400/40 text-emerald-400 bg-emerald-400/10`
  if (isCurrent(id))
    return `${base} border-primary-400 text-primary-400 bg-primary-400/10 shadow-[0_0_12px_rgba(96,165,250,0.3)] animate-pulse`
  return `${base} border-slate-700/30 text-slate-600 bg-transparent`
}

function dotClass(id: string): string {
  if (isCompleted(id)) return 'w-1.5 h-1.5 rounded-full flex-shrink-0 bg-emerald-400'
  if (isCurrent(id)) return 'w-1.5 h-1.5 rounded-full flex-shrink-0 bg-primary-400 animate-pulse'
  return 'w-1.5 h-1.5 rounded-full flex-shrink-0 bg-slate-700'
}

function isBranchActive(): boolean {
  return isCompleted('web_search') || isCompleted('transform_query')
}
</script>

<template>
  <aside class="w-56 flex-shrink-0 glass border-r border-slate-700/50 flex flex-col h-full">
    <!-- Logo -->
    <div class="px-5 py-6 border-b border-slate-700/50">
      <div class="flex items-center gap-3">
        <div class="w-9 h-9 rounded-lg bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center shadow-lg shadow-primary-500/25">
          <span class="text-white font-bold text-sm">AR</span>
        </div>
        <div>
          <h1 class="text-sm font-bold gradient-text">Agentic RAG</h1>
          <p class="text-[11px] text-slate-500">智能知识检索</p>
        </div>
      </div>
    </div>

    <!-- Navigation -->
    <nav class="px-3 py-4 space-y-1">
      <router-link
        v-for="item in navItems"
        :key="item.path"
        :to="item.path"
        :class="[
          'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-200',
          route.path === item.path
            ? 'bg-primary-500/15 text-primary-400 border border-primary-500/20'
            : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/40 border border-transparent'
        ]"
      >
        <component :is="item.icon" class="w-4 h-4" />
        <span>{{ item.label }}</span>
      </router-link>
    </nav>

    <!-- Agent 流程图（内联，零组件传递） -->
    <div class="border-t border-slate-700/50 px-3 py-3 space-y-1">
      <div class="flex items-center gap-1.5 mb-2">
        <Network class="w-3 h-3 text-slate-500" />
        <span class="text-[10px] font-medium text-slate-500">Agent 流程</span>
      </div>

      <div class="relative flex flex-col items-stretch gap-0">
        <!-- 主流程：retrieve → rerank → grade -->
        <template v-for="(node, i) in NODE_DEFS" :key="node.id">
          <div class="flex items-center gap-2">
            <div class="w-3 flex flex-col items-center" v-if="i > 0">
              <div class="w-0.5 h-3" :class="isCurrent(node.id) ? 'bg-primary-400' : isCompleted(node.id) ? 'bg-emerald-400/60' : 'bg-slate-700/20'" />
              <div :class="dotClass(node.id)" />
              <div class="w-0.5 h-3" :class="isCurrent(node.id) ? 'bg-primary-400' : isCompleted(node.id) ? 'bg-emerald-400/60' : 'bg-slate-700/20'" />
            </div>
            <div :class="nodeClass(node.id, node.enableKey)">
              <component :is="node.icon" class="w-3 h-3 flex-shrink-0" />
              {{ node.label }}
            </div>
          </div>
        </template>

        <!-- 不相关时分支 -->
        <div class="flex items-start gap-2">
          <div class="w-3 flex flex-col items-center min-h-0">
            <div class="w-0.5 flex-1 min-h-[6px]" :class="isBranchActive() ? 'bg-emerald-400/60' : 'bg-slate-700/20'" />
          </div>
          <span :class="['text-[9px] px-1.5 py-0.5 rounded border', isBranchActive() ? 'border-orange-400/30 text-orange-400 bg-orange-400/10' : 'border-slate-700/30 text-slate-600']">
            不相关时
          </span>
        </div>

        <!-- 分支：web_search / transform_query -->
        <template v-for="node in BRANCH_DEFS" :key="node.id">
          <div class="flex items-start gap-2">
            <div class="w-3 flex flex-col items-center">
              <div class="w-0.5 h-2 flex-shrink-0 bg-slate-700/20" />
              <div class="flex items-center gap-0.5">
                <div class="w-0.5 h-3" :class="isCurrent(node.id) ? 'bg-primary-400' : isCompleted(node.id) ? 'bg-emerald-400/60' : 'bg-slate-700/20'" />
                <div :class="dotClass(node.id)" />
              </div>
            </div>
            <div :class="nodeClass(node.id, node.enableKey)">
              <component :is="node.icon" class="w-3 h-3 flex-shrink-0" />
              {{ node.label }}
            </div>
          </div>
        </template>

        <!-- 汇合：generate → check_hallucination -->
        <template v-for="node in TAIL_DEFS" :key="node.id">
          <div class="flex items-center gap-2">
            <div class="w-3 flex flex-col items-center">
              <div class="w-0.5 h-3" :class="isCurrent(node.id) ? 'bg-primary-400' : isCompleted(node.id) ? 'bg-emerald-400/60' : 'bg-slate-700/20'" />
              <div :class="dotClass(node.id)" />
              <div class="w-0.5 h-3" :class="isCurrent(node.id) ? 'bg-primary-400' : isCompleted(node.id) ? 'bg-emerald-400/60' : 'bg-slate-700/20'" />
            </div>
            <div :class="nodeClass(node.id, node.enableKey)">
              <component :is="node.icon" class="w-3 h-3 flex-shrink-0" />
              {{ node.label }}
              <span v-if="node.id === 'check_hallucination' && !isEnabled(node.enableKey)" class="text-[9px] text-slate-600">(流式后置)</span>
            </div>
          </div>
        </template>
      </div>
    </div>

    <!-- Footer -->
    <div class="mt-auto px-3 py-4 border-t border-slate-700/50">
      <div class="text-[11px] text-slate-600 text-center">
        v0.1.0
      </div>
    </div>
  </aside>
</template>
