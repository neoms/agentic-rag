<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { MessageCircle, Database, Network, X } from 'lucide-vue-next'
import * as flow from '../../composables/agentFlowState'

const route = useRoute()
const navItems = [
  { path: '/', label: '对话', icon: MessageCircle },
  { path: '/documents', label: '知识库', icon: Database },
]

// ── SVG 节点布局 ──
// Flow:
//   START → analyze_kg_intent
//              ↓ (总线)
//       → retrieve | bm25 | 多角度查询 | 图谱(意图分析自动)
//              ↓ (全部收敛到合并)
//       → merge → rerank → grade → [相关] → generate → [反思] → check → END
//                                  → [不相关] → transform_query ─ retry → retrieve
//                                            → web_search ──────────── → generate
//   check_hallucination → [FAILED + retries<max] → generate (retry loop)
//   viewBox 0 0 380 480
type NodeState = 'active' | 'done' | 'disabled' | 'skipped' | 'pending'
interface N { id: string; label: string; x: number; y: number; w: number; h: number }
const NODES: N[] = [
  // 入口
  { id: 'analyze_kg_intent',   label: '意图分析',   x: 145, y: 34, w: 70, h: 26 },
  // 检索策略行（平级，语义检索必选，其余可选）
  { id: 'retrieve',             label: '语义检索',   x: 8,   y: 78, w: 60, h: 22 },
  { id: 'bm25_retrieve',        label: 'BM25',       x: 80,  y: 78, w: 52, h: 22 },
  { id: 'multi_query_retrieve', label: '多角度查询',  x: 144, y: 78, w: 68, h: 22 },
  { id: 'kg_retrieve',          label: '图谱检索',   x: 224, y: 78, w: 68, h: 22 },
  // 合并
  { id: 'parallel_retrieve_merge',label:'检索合并',  x: 150, y: 126,w: 60, h: 22 },
  // 可选节点（主链，可被 bypass）
  { id: 'rerank_documents',     label: '重排序',     x: 150, y: 166,w: 60, h: 22 },
  { id: 'grade_documents',      label: '文档评估',   x: 150, y: 206,w: 60, h: 22 },
  // 分支节点
  { id: 'transform_query',      label: '查询重写',   x: 8,   y: 248,w: 60, h: 22 },
  { id: 'web_search',           label: '联网搜索',   x: 252, y: 248,w: 60, h: 22 },
  // 固定节点
  { id: 'generate',             label: '生成回答',   x: 150, y: 282,w: 60, h: 22 },
  { id: 'check_hallucination',  label: '幻觉检测',   x: 150, y: 334,w: 60, h: 22 },
]
const byId = (id: string): N => NODES.find(n => n.id === id)!

// ── 状态 ──
const ENABLED: Record<string, keyof typeof flow> = {
  bm25_retrieve: 'enableBm25',
  multi_query_retrieve: 'enableMultiQuery',
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
function state(id: string): NodeState {
  if (!enabled(id)) return 'disabled'
  if (flow.skippedNodes.value.includes(id)) return 'skipped'
  if (flow.completedNodes.value.includes(id)) return 'done'
  if (flow.currentNode.value === id) return 'active'
  return 'pending'
}

// ── 节点点击交互 ──
const svgContainerRef = ref<HTMLElement | null>(null)
const popoverLeft = ref(0)
const popoverTop = ref(0)
const popoverShowInput = ref(false)
const popoverShowOutput = ref(false)

const selectedNodeInfo = computed(() =>
  flow.selectedNodeId.value ? getNodeInfo(flow.selectedNodeId.value) : undefined
)

function isNodeExecuted(id: string): boolean {
  const s = state(id)
  return s === 'done' || s === 'active'
}

function getNotExecutedReason(id: string): string {
  const s = state(id)
  if (s === 'disabled') return '该节点被策略开关关闭，未执行'
  if (s === 'skipped')  return '该节点在运行时被条件路由跳过'
  if (s === 'pending')  return '该节点在本次流程中未被调度执行'
  return ''
}

function nodeStatusColor(id: string): string {
  const s = state(id)
  if (s === 'done')     return 'text-emerald-400'
  if (s === 'active')   return 'text-blue-400'
  if (s === 'disabled') return 'text-slate-500'
  if (s === 'skipped')  return 'text-orange-400'
  return 'text-slate-500'
}

function nodeStatusText(id: string): string {
  const s = state(id)
  if (s === 'done')     return '✅ 已执行'
  if (s === 'active')   return '⏳ 执行中'
  if (s === 'disabled') return '⛔ 已禁用'
  if (s === 'skipped')  return '⏭ 已跳过'
  return '⏸ 待执行'
}

function getNodeInfo(id: string): flow.NodeDataInfo | undefined {
  return flow.nodeDataMap.value[id]
}

function isArray(val: unknown): val is unknown[] {
  return Array.isArray(val)
}

function onNodeClick(n: N, event: MouseEvent) {
  // 点击已选中节点 → 关闭
  if (flow.selectedNodeId.value === n.id) {
    flow.selectedNodeId.value = null
    return
  }

  // 重设展开状态
  popoverShowInput.value = false
  popoverShowOutput.value = false

  // 计算弹窗位置（相对视口 fixed 定位，避免被父容器 overflow 裁剪）
  flow.selectedNodeId.value = n.id
  const target = event.currentTarget as HTMLElement
  const rect = target.getBoundingClientRect()

  popoverLeft.value = rect.right + 4
  popoverTop.value = rect.top - 4
}

function closePopover() {
  flow.selectedNodeId.value = null
  popoverShowInput.value = false
  popoverShowOutput.value = false
}

// 点击弹窗外部关闭
function onDocumentClick(e: MouseEvent) {
  if (!flow.selectedNodeId.value) return
  const target = e.target as HTMLElement
  if (!target.closest('.flow-node') && !target.closest('.node-popover') && !target.closest('.popover-close-btn')) {
    closePopover()
  }
}

onMounted(() => document.addEventListener('click', onDocumentClick))
onUnmounted(() => document.removeEventListener('click', onDocumentClick))
const FILL: Record<string, string> = {
  active: '#1d4ed8', done: '#065f46', disabled: '#1e293b', skipped: '#1e293b', pending: '#1e293b',
}
const STROKE: Record<string, string> = {
  active: '#3b82f6', done: '#34d399', disabled: '#334155', skipped: '#d97706', pending: '#475569',
}
const TEXT: Record<string, string> = {
  active: '#93c5fd', done: '#6ee7b7', disabled: '#64748b', skipped: '#d97706', pending: '#94a3b8',
}

// ── Bypass 线条样式 ──
// Config-disabled node: bypass line is amber/visible, main edge dimmed
// Enabled node: bypass line nearly invisible, main edge full opacity
// 三条 bypass 弧线互斥，分别对应 rerank/grade 四种组合
//   Case 1 (rerank=ON, grade=ON):  均不 bypass，主链全亮
//   Case 2 (rerank=OFF, grade=OFF): 合并→直达生成（均关弧）
//   Case 3 (rerank=ON, grade=OFF):  重排序→直达生成（绕过 grade 弧）
//   Case 4 (rerank=OFF, grade=ON):  合并→直达文档评估（绕过 rerank 弧）
function rerankOnlyOff(): boolean { return !enabled('rerank_documents') && enabled('grade_documents') }
function gradeOnlyOff(): boolean  { return enabled('rerank_documents') && !enabled('grade_documents') }
function bothOff(): boolean       { return !enabled('rerank_documents') && !enabled('grade_documents') }
function bypassOpacity(pat: string): number {
  if (pat === 'rerank') return rerankOnlyOff() ? 0.55 : 0.04
  if (pat === 'grade')  return gradeOnlyOff()  ? 0.55 : 0.04
  return 0.04
}
function bypassColor(pat: string): string {
  if (pat === 'rerank') return rerankOnlyOff() ? '#f59e0b' : '#475569'
  if (pat === 'grade')  return gradeOnlyOff()  ? '#f59e0b' : '#475569'
  return '#475569'
}
function mainEdgeOpacity(seg: string): number {
  if (seg === 'merge_to_rerank')  return enabled('rerank_documents') ? 1 : 0.12
  if (seg === 'rerank_to_grade') return enabled('rerank_documents') && enabled('grade_documents') ? 1 : 0.12
  if (seg === 'grade_to_gen')    return enabled('grade_documents') ? 1 : 0.12
  return 1
}
// Check_hallucination bypass (独立，不影响 rerank/grade)
function reflectionBypassOpacity(): number { return enabled('check_hallucination') ? 0.08 : 0.55 }
function reflectionBypassColor(): string   { return enabled('check_hallucination') ? '#475569' : '#f59e0b' }

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
    <div ref="svgContainerRef" class="border-t border-slate-700/50 px-2 py-2 relative" style="z-index: 1">
      <div class="flex items-center gap-1.5 mb-1.5 pl-1">
        <Network class="w-3 h-3 text-slate-500" />
        <span class="text-[10px] font-medium text-slate-500">Agent 流程</span>
        <span v-if="flow.selectedNodeId.value" class="text-[10px] text-slate-600 ml-auto cursor-pointer popover-close-btn"
          @click.stop="closePopover">✕ 关闭</span>
      </div>

      <svg viewBox="0 0 380 480" class="w-full" style="max-height: 480px">
        <defs>
          <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/>
          </marker>
          <marker id="arrCyan" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#06b6d4"/>
          </marker>
          <marker id="arrAmber" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#f59e0b"/>
          </marker>
          <filter id="gl">
            <feGaussianBlur stdDeviation="2.5" result="b"/>
            <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>

        <!-- ═══ START ═══ -->
        <rect x="150" y="2" width="60" height="24" rx="12" ry="12" fill="#1e293b" stroke="#475569" stroke-width="1.5"/>
        <text x="180" y="14" text-anchor="middle" dominant-baseline="central" class="text-[12px] fill-slate-400">START</text>

        <!-- ═══ START → analyze ═══ -->
        <line x1="180" y1="26" x2="180" y2="34" stroke="#475569" stroke-width="1.5" fill="none" marker-end="url(#arr)"/>

        <!-- ═══ analyze_kg_intent 节点 ═══ -->
        <rect x="145" y="34" width="70" height="26" rx="4" ry="4"
          :fill="FILL[state('analyze_kg_intent')]"
          :stroke="STROKE[state('analyze_kg_intent')]"
          :stroke-width="state('analyze_kg_intent') === 'active' ? 2 : 1"
          :filter="state('analyze_kg_intent') === 'active' ? 'url(#gl)' : ''"
          :style="state('analyze_kg_intent') === 'active' ? 'animation: pulse 1.4s ease-in-out infinite' : ''"/>
        <text x="180" y="47" text-anchor="middle" dominant-baseline="central"
          :fill="TEXT[state('analyze_kg_intent')]"
          class="text-[10px]">意图分析</text>

        <!-- ═══ analyze → retrieve（左下弯折线） ═══ -->
        <path d="M 180 60 L 180 67 L 38 67 L 38 74"
          stroke="#475569" stroke-width="1.5" fill="none" marker-end="url(#arr)"/>

        <!-- ═══ 意图分析 → 检索总线（语义必选，其余可选扇出） ═══ -->
        <g stroke="#475569" stroke-width="1.5" fill="none">
          <!-- 意图分析 底部连到总线 -->
          <line x1="180" y1="60" x2="180" y2="70"/>
          <!-- 总线水平 -->
          <line x1="38" y1="70" x2="322" y2="70"/>
          <!-- 总线 → 语义检索（必选，始终可见） -->
          <line x1="38" y1="70" x2="38" y2="78" marker-end="url(#arrCyan)"/>
          <!-- 总线 → bm25（可选） -->
          <line x1="106" y1="70" x2="106" y2="78" marker-end="url(#arrCyan)"/>
          <!-- 总线 → 多角度查询（可选） -->
          <line x1="178" y1="70" x2="178" y2="78" marker-end="url(#arrCyan)"/>
          <!-- 总线 → 图谱检索（意图分析自动） -->
          <line x1="258" y1="70" x2="258" y2="78" marker-end="url(#arrCyan)"/>
        </g>

        <!-- ═══ 4 路检索 → merge 收敛 ═══ -->
        <g stroke="#475569" stroke-width="1.5" fill="none">
          <!-- 语义检索 → merge -->
          <line x1="38" y1="100" x2="155" y2="126" marker-end="url(#arr)"/>
          <!-- bm25 → merge -->
          <line x1="106" y1="100" x2="165" y2="126" marker-end="url(#arr)"/>
          <!-- 多角度查询 → merge -->
          <line x1="178" y1="100" x2="180" y2="126" marker-end="url(#arr)"/>
          <!-- 图谱检索 → merge -->
          <line x1="258" y1="100" x2="195" y2="126" marker-end="url(#arr)"/>
        </g>

        <!-- ═══ 主链箭头 merge → rerank → grade → generate → check → END ═══ -->
        <g stroke="#475569" stroke-width="1.5" fill="none">
          <!-- merge → rerank -->
          <line x1="180" :y1="byId('parallel_retrieve_merge').y + byId('parallel_retrieve_merge').h"
                x2="180" :y2="byId('rerank_documents').y"
            marker-end="url(#arr)" :opacity="mainEdgeOpacity('merge_to_rerank')"/>
          <!-- rerank → grade -->
          <line x1="180" :y1="byId('rerank_documents').y + byId('rerank_documents').h"
                x2="180" :y2="byId('grade_documents').y"
            marker-end="url(#arr)" :opacity="mainEdgeOpacity('rerank_to_grade')"/>
          <!-- grade → generate（相关） -->
          <line x1="180" :y1="byId('grade_documents').y + byId('grade_documents').h"
                x2="180" :y2="byId('generate').y"
            marker-end="url(#arr)" :opacity="mainEdgeOpacity('grade_to_gen')"/>
          <!-- generate → check -->
          <line x1="180" :y1="byId('generate').y + byId('generate').h"
                x2="180" :y2="byId('check_hallucination').y" marker-end="url(#arr)"/>
        </g>

        <!-- ═══ Bypass 绕过线（右侧虚线弧线） ═══ -->
        <!-- case 4: merge → 绕过 rerank → grade -->
        <path :d="`M 210 ${byId('parallel_retrieve_merge').y + byId('parallel_retrieve_merge').h/2} C 230 ${byId('parallel_retrieve_merge').y + byId('parallel_retrieve_merge').h/2}, 230 ${byId('grade_documents').y + byId('grade_documents').h/2}, 210 ${byId('grade_documents').y + byId('grade_documents').h/2}`"
          fill="none" :stroke="bypassColor('rerank')" stroke-width="1.2" stroke-dasharray="4 3"
          :opacity="bypassOpacity('rerank')" marker-end="url(#arrAmber)"/>
        <!-- case 3: rerank → 绕过 grade → generate -->
        <path :d="`M 210 ${byId('rerank_documents').y + byId('rerank_documents').h/2} C 230 ${byId('rerank_documents').y + byId('rerank_documents').h/2}, 230 ${byId('generate').y + byId('generate').h/2}, 210 ${byId('generate').y + byId('generate').h/2}`"
          fill="none" :stroke="bypassColor('grade')" stroke-width="1.2" stroke-dasharray="4 3"
          :opacity="bypassOpacity('grade')" marker-end="url(#arrAmber)"/>
        <!-- case 2: merge → 绕过 rerank 和 grade → generate -->
        <path :d="`M 210 ${byId('parallel_retrieve_merge').y + byId('parallel_retrieve_merge').h/2} C 240 ${byId('parallel_retrieve_merge').y + byId('parallel_retrieve_merge').h/2}, 240 ${byId('generate').y + byId('generate').h/2}, 210 ${byId('generate').y + byId('generate').h/2}`"
          fill="none" stroke="#f59e0b" stroke-width="1.2" stroke-dasharray="4 3"
          :opacity="bothOff() ? 0.55 : 0.04" marker-end="url(#arrAmber)"/>
        <!-- generate → 绕过 check → END -->
        <path :d="`M 210 ${byId('generate').y + byId('generate').h/2} C 230 ${byId('generate').y + byId('generate').h/2}, 230 ${byId('check_hallucination').y + byId('check_hallucination').h + 60}, 210 ${byId('check_hallucination').y + byId('check_hallucination').h + 60}`"
          fill="none" :stroke="reflectionBypassColor()" stroke-width="1.2" stroke-dasharray="4 3"
          :opacity="reflectionBypassOpacity()" marker-end="url(#arrAmber)"/>

        <!-- ═══ Bypass 标签 ═══ -->
        <text x="215" :y="byId('rerank_documents').y + byId('rerank_documents').h/2 - 5" :opacity="bypassOpacity('rerank')" :fill="bypassColor('rerank')" class="text-[9px]">绕过</text>
        <text x="215" :y="byId('grade_documents').y + byId('grade_documents').h + 14" :opacity="bypassOpacity('grade')" :fill="bypassColor('grade')" class="text-[9px]">绕过</text>
        <text x="225" :y="(byId('parallel_retrieve_merge').y + byId('parallel_retrieve_merge').h/2 + byId('generate').y + byId('generate').h/2) / 2 - 2" :opacity="bothOff() ? 0.6 : 0.04" fill="#f59e0b" class="text-[9px]">均关</text>
        <text x="215" :y="byId('check_hallucination').y + byId('check_hallucination').h + 26" :opacity="reflectionBypassOpacity()" :fill="reflectionBypassColor()" class="text-[9px]">绕过</text>

        <!-- ═══ 分支：grade → web_search / transform_query ═══ -->
        <g stroke="#475569" stroke-width="1.5" fill="none">
          <!-- grade → transform_query（左） -->
          <path :d="`M ${byId('grade_documents').x} ${byId('grade_documents').y+22} C ${byId('grade_documents').x} ${byId('grade_documents').y+30}, ${byId('transform_query').x+byId('transform_query').w/2} ${byId('transform_query').y-5}, ${byId('transform_query').x+byId('transform_query').w/2} ${byId('transform_query').y}`"
            marker-end="url(#arr)"/>
          <!-- grade → web_search（右） -->
          <path :d="`M ${byId('grade_documents').x+byId('grade_documents').w} ${byId('grade_documents').y+22} C ${byId('grade_documents').x+byId('grade_documents').w} ${byId('grade_documents').y+30}, ${byId('web_search').x+byId('web_search').w/2} ${byId('web_search').y-5}, ${byId('web_search').x+byId('web_search').w/2} ${byId('web_search').y}`"
            marker-end="url(#arr)"/>
        </g>

        <!-- ═══ web_search → generate ═══ -->
        <line :x1="byId('web_search').x + byId('web_search').w/2"
              :y1="byId('web_search').y + byId('web_search').h"
              :x2="byId('web_search').x + byId('web_search').w/2"
              :y2="byId('web_search').y + byId('web_search').h + 7"
          stroke="#475569" stroke-width="1.5" fill="none" marker-end="url(#arr)"/>
        <path :d="`M ${byId('web_search').x + byId('web_search').w/2} ${byId('web_search').y + byId('web_search').h + 7} C ${byId('web_search').x + byId('web_search').w/2} ${byId('web_search').y + byId('web_search').h + 11}, ${byId('generate').x + byId('generate').w/2 + 30} ${byId('generate').y - 1}, ${byId('generate').x + byId('generate').w/2 + 30} ${byId('generate').y}`"
          stroke="#475569" stroke-width="1.5" fill="none" marker-end="url(#arr)"/>

        <!-- ═══ 回环曲线 ═══ -->
        <!-- transform_query → retrieve（循环回检索） -->
        <path :d="`M ${byId('transform_query').x} ${byId('transform_query').y+22} C ${byId('transform_query').x} ${byId('transform_query').y+30}, -5 ${byId('transform_query').y+5}, -5 160 C -5 100, 38 100, 38 100`"
          fill="none" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="4 2" marker-end="url(#arrAmber)"/>
        <!-- check_hallucination → generate（幻觉重试） -->
        <path :d="`M ${byId('check_hallucination').x + byId('check_hallucination').w/2 + 30} ${byId('check_hallucination').y + byId('check_hallucination').h + 3} C ${byId('check_hallucination').x + byId('check_hallucination').w/2 + 42} ${byId('check_hallucination').y + byId('check_hallucination').h + 3}, ${byId('check_hallucination').x + byId('check_hallucination').w/2 + 42} ${byId('generate').y + byId('generate').h/2}, ${byId('check_hallucination').x + byId('check_hallucination').w/2 + 30} ${byId('generate').y + byId('generate').h/2}`"
          fill="none" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="4 2" marker-end="url(#arrAmber)"/>

        <!-- ═══ 分支标签 ═══ -->
        <text x="186" :y="byId('grade_documents').y + byId('grade_documents').h + 34" class="text-[9px] fill-green-500/70">相关</text>
        <text x="226" :y="byId('grade_documents').y + byId('grade_documents').h + 20" class="text-[9px] fill-orange-400/70">不相关+联网</text>
        <text x="42"  :y="byId('grade_documents').y + byId('grade_documents').h + 20" class="text-[9px] fill-orange-400/70">不相关</text>

        <!-- ═══ 回环标签 ═══ -->
        <text x="22" y="210" class="text-[9px] fill-amber-500/80" transform="rotate(-90 22 210)">回检索</text>
        <text x="215" y="345" class="text-[9px] fill-red-400/80">重试</text>

        <!-- ═══ END ═══ -->
        <rect x="150" y="440" width="60" height="24" rx="12" ry="12" fill="#1e293b" stroke="#475569" stroke-width="1.5"/>
        <text x="180" y="452" text-anchor="middle" dominant-baseline="central" class="text-[12px] fill-slate-400">END</text>

        <!-- ═══ check → END ═══ -->
        <line :x1="byId('check_hallucination').x + byId('check_hallucination').w/2"
              :y1="byId('check_hallucination').y + byId('check_hallucination').h"
              :x2="byId('check_hallucination').x + byId('check_hallucination').w/2"
              y2="440"
          stroke="#475569" stroke-width="1.5" fill="none" marker-end="url(#arr)"/>

        <!-- ═══ 节点列表 ═══ -->
        <g v-for="n in NODES" :key="n.id" class="flow-node" :class="{ 'cursor-pointer': true }"
          @click.stop="onNodeClick(n, $event)"
          :style="flow.selectedNodeId.value === n.id ? 'filter: brightness(1.3)' : ''">
          <rect
            :x="n.x" :y="n.y" :width="n.w" :height="n.h" rx="4" ry="4"
            :fill="FILL[state(n.id)]"
            :stroke="STROKE[state(n.id)]"
            :stroke-width="state(n.id) === 'active' || flow.selectedNodeId.value === n.id ? 2 : 1"
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

      <!-- ═══ 节点信息 Popover（Teleport 到 body 避免被 sidebar 层叠上下文遮挡） ═══ -->
      <Teleport to="body">
        <div v-if="flow.selectedNodeId.value"
          class="node-popover fixed z-[9999] bg-slate-800 border border-slate-600 rounded-lg shadow-2xl w-64 overflow-hidden"
          :style="{ left: popoverLeft + 'px', top: popoverTop + 'px' }">
          <!-- 标题 -->
          <div class="flex items-center justify-between px-3 py-2 border-b border-slate-700">
            <span class="text-xs font-semibold text-slate-200">
              {{ byId(flow.selectedNodeId.value)?.label }}
            </span>
            <button @click.stop="closePopover"
              class="p-0.5 rounded hover:bg-slate-700 text-slate-500 hover:text-slate-300 transition-colors">
              <X class="w-3 h-3" />
            </button>
          </div>
          <!-- 状态 -->
          <div class="px-3 py-1.5 border-b border-slate-700/50">
            <span class="text-[10px] text-slate-500 mr-2">状态</span>
            <span class="text-[11px]" :class="nodeStatusColor(flow.selectedNodeId.value)">
              {{ nodeStatusText(flow.selectedNodeId.value) }}
            </span>
          </div>
          <!-- I/O 数据（已执行节点） -->
          <template v-if="isNodeExecuted(flow.selectedNodeId.value)">
          <div class="px-3 py-2 space-y-1.5">
            <!-- 输入 -->
            <div>
              <div class="flex items-center gap-1 cursor-pointer select-none hover:bg-slate-700/30 rounded px-1 -mx-1"
                @click.stop="popoverShowInput = !popoverShowInput">
                <span class="text-[10px] font-mono text-blue-400/70">{{ popoverShowInput ? '▾' : '▸' }}</span>
                <span class="text-[10px] font-medium text-blue-400/70">输入</span>
              </div>
              <template v-if="popoverShowInput">
                <!-- 字符串 -->
                <p v-if="selectedNodeInfo && !isArray(selectedNodeInfo.input)"
                  class="mt-0.5 text-[10px] text-slate-400 leading-relaxed whitespace-pre-wrap break-all bg-slate-900/50 rounded px-1.5 py-1">
                  {{ selectedNodeInfo.input }}
                </p>
                <!-- 数组逐条列表 -->
                <ul v-else-if="selectedNodeInfo && isArray(selectedNodeInfo.input)"
                  class="mt-0.5 space-y-1 max-h-40 overflow-y-auto">
                  <li v-for="(item, i) in selectedNodeInfo.input" :key="i"
                    class="text-[10px] text-slate-400 leading-relaxed whitespace-pre-wrap break-all bg-slate-900/50 rounded px-1.5 py-1">
                    {{ item }}
                  </li>
                </ul>
                <p v-else class="mt-0.5 text-[10px] text-slate-500 italic">无输入数据</p>
              </template>
            </div>
            <!-- 输出 -->
            <div>
              <div class="flex items-center gap-1 cursor-pointer select-none hover:bg-slate-700/30 rounded px-1 -mx-1"
                @click.stop="popoverShowOutput = !popoverShowOutput">
                <span class="text-[10px] font-mono text-emerald-400/70">{{ popoverShowOutput ? '▾' : '▸' }}</span>
                <span class="text-[10px] font-medium text-emerald-400/70">输出</span>
              </div>
              <template v-if="popoverShowOutput">
                <!-- 字符串 -->
                <p v-if="selectedNodeInfo && !isArray(selectedNodeInfo.output)"
                  class="mt-0.5 text-[10px] text-slate-400 leading-relaxed whitespace-pre-wrap break-all bg-slate-900/50 rounded px-1.5 py-1">
                  {{ selectedNodeInfo.output }}
                </p>
                <!-- 数组逐条列表 -->
                <ul v-else-if="selectedNodeInfo && isArray(selectedNodeInfo.output)"
                  class="mt-0.5 space-y-1 max-h-40 overflow-y-auto">
                  <li v-for="(item, i) in selectedNodeInfo.output" :key="i"
                    class="text-[10px] text-slate-400 leading-relaxed whitespace-pre-wrap break-all bg-slate-900/50 rounded px-1.5 py-1">
                    {{ item }}
                  </li>
                </ul>
                <p v-else class="mt-0.5 text-[10px] text-slate-500 italic">无输出数据</p>
              </template>
            </div>
          </div>
          <div v-if="!selectedNodeInfo" class="px-3 py-2">
            <p class="text-[10px] text-slate-500 italic">节点已执行，但本次查询未产生详细数据</p>
          </div>
        </template>
        <!-- 未执行提示 -->
        <div v-else class="px-3 py-2">
          <p class="text-[10px] text-slate-500 leading-relaxed">
            {{ getNotExecutedReason(flow.selectedNodeId.value) }}
          </p>
        </div>
      </div>
      </Teleport>
    </div>

    <!-- 点击提示 -->
    <div class="px-3 py-1.5 border-t border-slate-700/30">
      <p class="text-[10px] text-slate-600 text-center leading-relaxed">
        💡 点击流程图中的节点可查看输入 / 输出详情
      </p>
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
