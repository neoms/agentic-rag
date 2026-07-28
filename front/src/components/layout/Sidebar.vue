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
// START → analyze_kg_intent → retrieve → 5 路并行检索 → merge → [可选节点带 bypass] → END
// viewBox 0 0 380 500
interface N { id: string; label: string; x: number; y: number; w: number; h: number }
const NODES: N[] = [
  // START
  // analyze_kg_intent
  // 并行检索行（同级同行）
  { id: 'retrieve',             label: '语义检索',   x: 8,  y: 82, w: 60, h: 25 },
  { id: 'bm25_retrieve',        label: 'BM25',       x: 80, y: 82, w: 52, h: 25 },
  { id: 'hyde_retrieve',        label: 'HyDE',       x: 144,y: 82, w: 52, h: 25 },
  { id: 'multi_query_retrieve', label: '多角度查询',  x: 208,y: 82, w: 68, h: 25 },
  { id: 'kg_retrieve',          label: '图谱检索',   x: 288,y: 82, w: 68, h: 25 },
  // 合并
  { id: 'merge_retrieval',      label: '合并去重',   x: 155,y: 127,w: 60, h: 25 },
  // 可选节点（主链，可被 bypass）
  { id: 'rerank_documents',     label: '重排序',     x: 155,y: 172,w: 60, h: 25 },
  { id: 'grade_documents',      label: '文档评估',   x: 155,y: 217,w: 60, h: 25 },
  // 分支节点
  { id: 'transform_query',      label: '查询重写',   x: 8,  y: 255,w: 60, h: 25 },
  { id: 'web_search',           label: '联网搜索',   x: 252,y: 255,w: 60, h: 25 },
  // 固定节点
  { id: 'generate',             label: '生成回答',   x: 155,y: 292,w: 60, h: 25 },
  { id: 'check_hallucination',  label: '幻觉检测',   x: 155,y: 342,w: 60, h: 25 },
]
const byId = (id: string): N => NODES.find(n => n.id === id)!

// ── 状态 ──
const ENABLED: Record<string, keyof typeof flow> = {
  bm25_retrieve: 'enableBm25',
  hyde_retrieve: 'enableHyde',
  multi_query_retrieve: 'enableMultiQuery',
  kg_retrieve: 'enableKg',
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

// ── Bypass 线条样式（节点禁用时路径线高亮，启用时几乎不可见） ──
function bypassOpacity(id: string): number { return enabled(id) ? 0.10 : 0.55 }
function bypassColor(id: string): string { return enabled(id) ? '#475569' : '#f59e0b' }
function mainEdgeOpacity(id: string): number { return enabled(id) ? 1 : 0.15 }
</script>

<template>
  <aside class="w-72 flex-shrink-0 glass border-r border-slate-700/50 flex flex-col h-full">
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

      <svg viewBox="0 0 380 500" class="w-full" style="max-height: 500px">
        <defs>
          <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/>
          </marker>
          <marker id="arrCyan" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#06b6d4"/>
          </marker>
          <marker id="arrPink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#f472b6"/>
          </marker>
          <marker id="arrIndigo" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#818cf8"/>
          </marker>
          <marker id="arrOrange" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#f97316"/>
          </marker>
          <filter id="gl">
            <feGaussianBlur stdDeviation="2.5" result="b"/>
            <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>

        <!-- ═══ START 节点 ═══ -->
        <rect x="150" y="5" width="60" height="24" rx="12" ry="12" fill="#1e293b" stroke="#475569" stroke-width="1.5"/>
        <text x="180" y="17" text-anchor="middle" dominant-baseline="central" class="text-[12px] fill-slate-400">START</text>

        <!-- ═══ analyze_kg_intent 节点 ═══ -->
        <rect x="145" y="36" width="70" height="26" rx="4" ry="4"
          :fill="FILL[state('analyze_kg_intent')]"
          :stroke="STROKE[state('analyze_kg_intent')]"
          :stroke-width="state('analyze_kg_intent') === 'active' ? 2 : 1"
          :filter="state('analyze_kg_intent') === 'active' ? 'url(#gl)' : ''"
          :style="state('analyze_kg_intent') === 'active' ? 'animation: pulse 1.4s ease-in-out infinite' : ''"/>
        <text x="180" y="49" text-anchor="middle" dominant-baseline="central"
          :fill="TEXT[state('analyze_kg_intent')]"
          class="text-[10px]">意图分析</text>

        <!-- ═══ START → analyze_kg_intent → 五路检索（扇出） ═══ -->
        <g stroke="#475569" stroke-width="1.5" fill="none">
          <line x1="180" y1="27" x2="180" y2="36" marker-end="url(#arr)"/>
          <line x1="180" y1="62" x2="180" y2="68"/>
          <line x1="44" y1="68" x2="322" y2="68"/>
          <line x1="44" y1="68" x2="44" :y2="byId('retrieve').y" marker-end="url(#arr)"/>
          <line x1="112"y1="68" x2="112":y2="byId('bm25_retrieve').y" marker-end="url(#arrCyan)"/>
          <line x1="176"y1="68" x2="176":y2="byId('hyde_retrieve').y" marker-end="url(#arrPink)"/>
          <line x1="248"y1="68" x2="248":y2="byId('multi_query_retrieve').y" marker-end="url(#arrIndigo)"/>
          <line x1="322"y1="68" x2="322":y2="byId('kg_retrieve').y" marker-end="url(#arrOrange)"/>
        </g>

        <!-- ═══ 五路检索 → merge 收敛 ═══ -->
        <g stroke="#475569" stroke-width="1.5" fill="none">
          <line :x1="byId('retrieve').x+30" :y1="byId('retrieve').y+25"
                :x2="byId('merge_retrieval').x+10" :y2="byId('merge_retrieval').y" marker-end="url(#arr)"/>
          <line :x1="byId('bm25_retrieve').x+26" :y1="byId('bm25_retrieve').y+25"
                :x2="byId('merge_retrieval').x+20" :y2="byId('merge_retrieval').y" marker-end="url(#arr)"/>
          <line :x1="byId('hyde_retrieve').x+26" :y1="byId('hyde_retrieve').y+25"
                :x2="byId('merge_retrieval').x+30" :y2="byId('merge_retrieval').y" marker-end="url(#arr)"/>
          <line :x1="byId('multi_query_retrieve').x+34" :y1="byId('multi_query_retrieve').y+25"
                :x2="byId('merge_retrieval').x+40" :y2="byId('merge_retrieval').y" marker-end="url(#arr)"/>
          <line :x1="byId('kg_retrieve').x+34" :y1="byId('kg_retrieve').y+25"
                :x2="byId('merge_retrieval').x+50" :y2="byId('merge_retrieval').y" marker-end="url(#arr)"/>
        </g>

        <!-- ═══ 主链箭头 merge → rerank → grade → generate → check → END ═══ -->
        <g stroke="#475569" stroke-width="1.5" fill="none">
          <!-- merge → rerank -->
          <line :x1="byId('merge_retrieval').x+byId('merge_retrieval').w/2" :y1="byId('merge_retrieval').y+25"
                :x2="byId('rerank_documents').x+byId('rerank_documents').w/2" :y2="byId('rerank_documents').y"
                marker-end="url(#arr)" :opacity="mainEdgeOpacity('rerank_documents')"/>
          <!-- rerank → grade -->
          <line :x1="byId('rerank_documents').x+byId('rerank_documents').w/2" :y1="byId('rerank_documents').y+25"
                :x2="byId('grade_documents').x+byId('grade_documents').w/2" :y2="byId('grade_documents').y"
                marker-end="url(#arr)" :opacity="mainEdgeOpacity('rerank_documents')"/>
          <!-- grade → generate（相关） -->
          <line :x1="byId('grade_documents').x+byId('grade_documents').w/2" :y1="byId('grade_documents').y+25"
                :x2="byId('generate').x+byId('generate').w/2" :y2="byId('generate').y"
                marker-end="url(#arr)" :opacity="mainEdgeOpacity('grade_documents')"/>
          <!-- generate → check_hallucination -->
          <line :x1="byId('generate').x+byId('generate').w/2" :y1="byId('generate').y+25"
                :x2="byId('check_hallucination').x+byId('check_hallucination').w/2" :y2="byId('check_hallucination').y"
                marker-end="url(#arr)"/>
        </g>

        <!-- ═══ Bypass 绕过线（可选节点右侧虚线弧线） ═══ -->
        <path :d="`M ${byId('merge_retrieval').x+byId('merge_retrieval').w} ${byId('merge_retrieval').y+12} C ${byId('merge_retrieval').x+byId('merge_retrieval').w+20} ${byId('merge_retrieval').y+12}, ${byId('grade_documents').x+byId('grade_documents').w+20} ${byId('grade_documents').y+12}, ${byId('grade_documents').x+byId('grade_documents').w} ${byId('grade_documents').y+12}`"
              fill="none" :stroke="bypassColor('rerank_documents')" stroke-width="1.2" stroke-dasharray="4 3"
              :opacity="bypassOpacity('rerank_documents')" marker-end="url(#arr)"/>
        <path :d="`M ${byId('rerank_documents').x+byId('rerank_documents').w} ${byId('rerank_documents').y+12} C ${byId('rerank_documents').x+byId('rerank_documents').w+20} ${byId('rerank_documents').y+12}, ${byId('generate').x+byId('generate').w+20} ${byId('generate').y+12}, ${byId('generate').x+byId('generate').w} ${byId('generate').y+12}`"
              fill="none" :stroke="bypassColor('grade_documents')" stroke-width="1.2" stroke-dasharray="4 3"
              :opacity="bypassOpacity('grade_documents')" marker-end="url(#arr)"/>
        <path :d="`M ${byId('generate').x+byId('generate').w} ${byId('generate').y+12} C ${byId('generate').x+byId('generate').w+25} ${byId('generate').y+12}, ${byId('generate').x+byId('generate').w+25} 395, 215 420`"
              fill="none" :stroke="bypassColor('check_hallucination')" stroke-width="1.2" stroke-dasharray="4 3"
              :opacity="bypassOpacity('check_hallucination')" marker-end="url(#arr)"/>

        <!-- ═══ Bypass 标签 ═══ -->
        <text :x="byId('merge_retrieval').x+byId('merge_retrieval').w+4" :y="byId('merge_retrieval').y+25"
              :opacity="bypassOpacity('rerank_documents')" :fill="bypassColor('rerank_documents')"
              class="text-[9px]">绕过</text>
        <text :x="byId('rerank_documents').x+byId('rerank_documents').w+4" :y="byId('rerank_documents').y+25"
              :opacity="bypassOpacity('grade_documents')" :fill="bypassColor('grade_documents')"
              class="text-[9px]">绕过</text>
        <text :x="byId('generate').x+byId('generate').w+4" :y="byId('generate').y+25"
              :opacity="bypassOpacity('check_hallucination')" :fill="bypassColor('check_hallucination')"
              class="text-[9px]">绕过</text>

        <!-- ═══ 分支：grade → web_search / transform_query ═══ -->
        <g stroke="#475569" stroke-width="1.5" fill="none">
          <line :x1="byId('grade_documents').x" :y1="byId('grade_documents').y+25"
                :x2="byId('transform_query').x+byId('transform_query').w/2" :y2="byId('transform_query').y"
                marker-end="url(#arr)"/>
          <line :x1="byId('grade_documents').x+byId('grade_documents').w" :y1="byId('grade_documents').y+25"
                :x2="byId('web_search').x+byId('web_search').w/2" :y2="byId('web_search').y"
                marker-end="url(#arr)"/>
        </g>

        <!-- ═══ web_search → generate ═══ -->
        <line :x1="byId('web_search').x+byId('web_search').w/2" :y1="byId('web_search').y+25"
              :x2="byId('web_search').x+byId('web_search').w/2" :y2="byId('generate').y+3"
              stroke="#475569" stroke-width="1.5" fill="none" marker-end="url(#arr)"/>

        <!-- ═══ 回环曲线 ═══ -->
        <!-- transform_query → retrieve（循环） -->
        <path :d="`M ${byId('transform_query').x} ${byId('transform_query').y+25} C ${byId('transform_query').x} ${byId('transform_query').y+32}, -5 ${byId('transform_query').y+10}, -5 155 C -5 98, 8 94, 8 94`"
              fill="none" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="4 2" marker-end="url(#arr)"/>
        <!-- check_hallucination → generate（幻觉重试） -->
        <path :d="`M ${byId('check_hallucination').x+byId('check_hallucination').w} ${byId('check_hallucination').y+12} C ${byId('check_hallucination').x+byId('check_hallucination').w+12} ${byId('check_hallucination').y+12}, ${byId('check_hallucination').x+byId('check_hallucination').w+12} ${byId('generate').y+12}, ${byId('generate').x+byId('generate').w} ${byId('generate').y+12}`"
              fill="none" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="4 2" marker-end="url(#arr)"/>

        <!-- ═══ 分支标签 ═══ -->
        <text :x="byId('grade_documents').x+byId('grade_documents').w/2+2" :y="byId('grade_documents').y+25+14"
              class="text-[9px] fill-green-500/70">相关</text>
        <text :x="byId('web_search').x-30" :y="byId('web_search').y+12"
              class="text-[9px] fill-orange-400/70">不相关+联网</text>
        <text :x="byId('transform_query').x+byId('transform_query').w+4" :y="byId('transform_query').y+12"
              class="text-[9px] fill-orange-400/70">不相关</text>

        <!-- ═══ 回环标签 ═══ -->
        <text x="22" y="172" class="text-[9px] fill-amber-500/80" transform="rotate(-90 22 172)">回检索</text>
        <text :x="byId('check_hallucination').x+byId('check_hallucination').w+4" :y="byId('check_hallucination').y"
              class="text-[9px] fill-red-400/80">重试</text>

        <!-- ═══ END 节点 ═══ -->
        <rect x="150" y="415" width="60" height="24" rx="12" ry="12" fill="#1e293b" stroke="#475569" stroke-width="1.5"/>
        <text x="180" y="427" text-anchor="middle" dominant-baseline="central" class="text-[12px] fill-slate-400">END</text>

        <!-- ═══ check → END ═══ -->
        <g stroke="#475569" stroke-width="1.5" fill="none">
          <line :x1="byId('check_hallucination').x+byId('check_hallucination').w/2" :y1="byId('check_hallucination').y+25"
                x2="180" y2="415" marker-end="url(#arr)"/>
          <text :x="byId('generate').x+byId('generate').w+4" :y="byId('generate').y+16"
                class="text-[9px] fill-slate-500">关</text>
        </g>

        <!-- ═══ 节点 ═══ -->
        <g v-for="n in NODES" :key="n.id">
          <rect
            :x="n.x" :y="n.y" :width="n.w" :height="n.h" rx="4" ry="4"
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
            class="text-[12px]"
          >{{ n.label }}</text>
          <text v-if="state(n.id) === 'done'"
            :x="n.x + n.w - 6" :y="n.y + n.h/2"
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
