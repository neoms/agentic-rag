<script setup lang="ts">
import { useRoute } from 'vue-router'
import { MessageCircle, Database, Network } from 'lucide-vue-next'
import * as flow from '../../composables/agentFlowState'

const route = useRoute()
const navItems = [
  { path: '/', label: '对话', icon: MessageCircle },
  { path: '/documents', label: '知识库', icon: Database },
]

// ── SVG 节点布局 ──
// 主流程: retrieve → rerank → grade → generate → check_hallucination (垂直居中)
// 左分支: transform_query (不相关时, 循环回 retrieve)
// 右分支: web_search      (不相关+联网时, 进入 generate)
// viewBox 0 0 220 340
interface N { id: string; label: string; x: number; y: number; w: number; h: number }
const NODES: N[] = [
  { id: 'retrieve',             label: '检索',       x: 55, y: 5,   w: 110, h: 22 },
  { id: 'rerank_documents',     label: '重排序',     x: 55, y: 40,  w: 110, h: 22 },
  { id: 'grade_documents',      label: '文档评估',   x: 55, y: 75,  w: 110, h: 22 },
  { id: 'transform_query',      label: '查询重写',   x: 5,  y: 130, w: 92,  h: 22 },
  { id: 'web_search',           label: '联网搜索',   x: 123,y: 130, w: 92,  h: 22 },
  { id: 'generate',             label: '生成回答',   x: 55, y: 200, w: 110, h: 22 },
  { id: 'check_hallucination',  label: '幻觉检测',   x: 55, y: 270, w: 110, h: 22 },
]
const byId = (id: string): N => NODES.find(n => n.id === id)!

// ── 状态 ──
const ENABLED: Record<string, keyof typeof flow> = {
  rerank_documents: 'enableRerank',
  grade_documents: 'enableGradeDocuments',
  web_search: 'enableWebSearch',
  transform_query: 'enableTransformQuery',
  check_hallucination: 'enableReflection',
}
function enabled(id: string): boolean {
  const k = ENABLED[id]
  return k ? (flow[k].value as boolean) : true
}
function state(id: string): 'active' | 'done' | 'disabled' | 'pending' {
  if (!enabled(id)) return 'disabled'
  if (flow.completedNodes.value.includes(id)) return 'done'
  if (flow.currentNode.value === id) return 'active'
  return 'pending'
}
const FILL: Record<string, string> = {
  active: '#1d4ed8', done: '#065f46', disabled: '#1e293b', pending: '#1e293b',
}
const STROKE: Record<string, string> = {
  active: '#3b82f6', done: '#34d399', disabled: '#334155', pending: '#475569',
}
const TEXT: Record<string, string> = {
  active: '#93c5fd', done: '#6ee7b7', disabled: '#64748b', pending: '#94a3b8',
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
      <router-link v-for="item in navItems" :key="item.path" :to="item.path"
        :class="['flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-200',
          route.path === item.path
            ? 'bg-primary-500/15 text-primary-400 border border-primary-500/20'
            : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/40 border border-transparent']">
        <component :is="item.icon" class="w-4 h-4" />
        <span>{{ item.label }}</span>
      </router-link>
    </nav>

    <!-- SVG 流程图 -->
    <div class="border-t border-slate-700/50 px-2 py-2">
      <div class="flex items-center gap-1.5 mb-1.5 pl-1">
        <Network class="w-3 h-3 text-slate-500" />
        <span class="text-[10px] font-medium text-slate-500">Agent 流程</span>
      </div>

      <svg viewBox="0 0 220 310" class="w-full" style="max-height: 310px">
        <defs>
          <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/>
          </marker>
          <filter id="gl">
            <feGaussianBlur stdDeviation="2.5" result="b"/>
            <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>

        <!-- ═══ 直线上箭头 (g=灰色) ═══ -->
        <g stroke="#475569" stroke-width="1.5" fill="none">
          <!-- 主流程: retrieve → rerank → grade -->
          <line x1="110" :y1="byId('retrieve').y+22"         x2="110" :y2="byId('rerank_documents').y"     marker-end="url(#arr)"/>
          <line x1="110" :y1="byId('rerank_documents').y+22" x2="110" :y2="byId('grade_documents').y"      marker-end="url(#arr)"/>

          <!-- grade → generate (相关) -->
          <line x1="110" :y1="byId('grade_documents').y+22"  x2="110" :y2="byId('generate').y"            marker-end="url(#arr)"/>
          <!-- grade → web_search (不相关+联网)  斜线向右下 -->
          <line x1="140" :y1="byId('grade_documents').y+22"  x2="170" :y2="byId('web_search').y+11"       marker-end="url(#arr)"/>
          <!-- grade → transform_query (不相关) 斜线向左下 -->
          <line x1="80"  :y1="byId('grade_documents').y+22"  x2="50"  :y2="byId('transform_query').y+11"  marker-end="url(#arr)"/>

          <!-- web_search → generate -->
          <line x1="170" :y1="byId('web_search').y+22"       x2="170" :y2="byId('generate').y+3"          marker-end="url(#arr)"/>

          <!-- generate → check_hallucination (开自反思) -->
          <line x1="110" :y1="byId('generate').y+22"         x2="110" :y2="byId('check_hallucination').y" marker-end="url(#arr)"/>
        </g>

        <!-- ═══ 回环曲线 ═══ -->
        <!-- transform_query → retrieve (循环回检索) -->
        <path d="M 50 152 C 50 170, -5 160, -5 100 C -5 30, 40 20, 55 20"
              fill="none" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="4 2" marker-end="url(#arr)"/>
        <!-- check_hallucination → generate (检测失败重试) -->
        <path d="M 170 281 C 195 281, 195 210, 170 210"
              fill="none" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="4 2" marker-end="url(#arr)"/>

        <!-- ═══ 分支标签 ═══ -->
        <text x="118" y="107" class="text-[7px] fill-green-500/70">相关</text>
        <text x="145" y="110" class="text-[7px] fill-orange-400/70">不相关+联网</text>
        <text x="52"  y="110" class="text-[7px] fill-orange-400/70">不相关</text>
        <text x="130" y="246" class="text-[7px] fill-slate-500">开自反思</text>
        <text x="178" y="246" class="text-[7px] fill-slate-500">关</text>

        <!-- ═══ 回环标签 ═══ -->
        <text x="25" y="104" class="text-[7px] fill-amber-500/80" transform="rotate(-90 25 104)">回检索</text>
        <text x="183" y="257" class="text-[7px] fill-red-400/80">失败重试</text>

        <!-- ═══ 节点 ═══ -->
        <g v-for="n in NODES" :key="n.id">
          <rect
            :x="n.x" :y="n.y" :width="n.w" :height="n.h" rx="6" ry="6"
            :fill="FILL[state(n.id)]"
            :stroke="STROKE[state(n.id)]"
            :stroke-width="state(n.id) === 'active' ? 2 : 1"
            :filter="state(n.id) === 'active' ? 'url(#gl)' : ''"
            :style="state(n.id) === 'active' ? 'animation: pulse 1.4s ease-in-out infinite' : ''"
          />
          <text
            :x="n.x + n.w/2" :y="n.y + n.h/2"
            text-anchor="middle" dominant-baseline="central"
            :fill="TEXT[state(n.id)]"
            :style="state(n.id) === 'disabled' ? 'text-decoration: line-through' : ''"
            class="text-[9px]"
          >{{ n.label }}</text>
          <text v-if="state(n.id) === 'done'"
            :x="n.x + n.w - 7" :y="n.y + n.h/2"
            text-anchor="middle" dominant-baseline="central"
            class="text-[10px] fill-emerald-300"
          >✓</text>
        </g>
      </svg>
    </div>

    <!-- Footer -->
    <div class="mt-auto px-3 py-4 border-t border-slate-700/50">
      <div class="text-[11px] text-slate-600 text-center">v0.1.0</div>
    </div>
  </aside>
</template>

<style>
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
</style>
