<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { MessageCircle, Database, Network, X } from 'lucide-vue-next'
import * as flow from '../../composables/agentFlowState'
import NodeDataRenderer from './NodeDataRenderer.vue'

const route = useRoute()
const navItems = [
  { path: '/', label: '对话', icon: MessageCircle },
  { path: '/documents', label: '知识库', icon: Database },
]

// ── SVG 节点布局 ──
// Flow:
//   START → cache_exact（精准缓存，纵向）
//              ↓ [未命中] → cache_semantic（语义缓存）
//              ├─ [精准命中] → cache_replay（输出回放，虚拟节点） → END
//              ├─ [语义命中] → cache_replay（输出回放） → END
//              └─ [全部未命中] → analyze_kg_intent
//                    ↓ (总线)
//            → retrieve | bm25 | 多角度查询 | 图谱(意图分析自动)
//                    ↓ (全部收敛到合并)
//            → merge → rerank → grade（仅当查询重写或联网搜索任一开启时运行）
//                     → [相关] → judge_complexity (LLM_MODEL_FAST)
//                     → [不相关] → transform_query（默认关闭，手动开启且最多 1 次）─ retry → retrieve
//                               → web_search ──────────── → judge
//                     两者都关 → 跳过 grade，merge/rerank 直达 judge
//            judge_complexity → [SIMPLE] → generate_simple (LLM_MODEL_FAST)
//                             → [COMPLEX] → generate_complex (LLM_MODEL_STRONG)
//            generate_simple/complex → [反思] → check_hallucination
//            check_hallucination → cache_store（缓存写入，虚拟节点） → END
//            幻觉 FAILED：不重试，直接返回（流程图不画重试回环）
//   viewBox 0 0 460 740
type NodeState = 'active' | 'done' | 'disabled' | 'skipped' | 'pending'
interface N { id: string; label: string; x: number; y: number; w: number; h: number }
const NODES: N[] = [
  // 缓存查询拆分为两个纵向顺序的展示节点（均由服务层 cache_lookup 虚拟节点的数据驱动）
  { id: 'cache_exact',          label: '精准缓存',   x: 188, y: 50,  w: 84, h: 30 },
  { id: 'cache_semantic',       label: '语义缓存',   x: 188, y: 100, w: 84, h: 30 },
  { id: 'cache_replay',         label: '输出回放',   x: 330, y: 75,  w: 84, h: 30 },
  // 入口（主链居中 x=230）
  { id: 'analyze_kg_intent',   label: '意图分析',   x: 190, y: 170, w: 80, h: 30 },
  // 检索策略行（平级，语义检索必选，其余可选）
  { id: 'retrieve',             label: '语义检索',   x: 44,  y: 230, w: 80, h: 26 },
  { id: 'bm25_retrieve',        label: 'BM25',       x: 138, y: 230, w: 68, h: 26 },
  { id: 'multi_query_retrieve', label: '多角度查询',  x: 220, y: 230, w: 88, h: 26 },
  { id: 'kg_retrieve',          label: '图谱检索',   x: 322, y: 230, w: 88, h: 26 },
  // 合并
  { id: 'parallel_retrieve_merge',label:'检索合并',  x: 190, y: 292, w: 80, h: 30 },
  // 可选节点（主链，可被 bypass）
  { id: 'rerank_documents',     label: '重排序',     x: 190, y: 352, w: 80, h: 30 },
  { id: 'grade_documents',      label: '文档评估',   x: 190, y: 412, w: 80, h: 30 },
  // 分支节点（同排：左-查询重写，中-复杂度判定，右-联网搜索）
  { id: 'transform_query',      label: '查询重写',   x: 58,  y: 472, w: 80, h: 26 },
  { id: 'judge_complexity',     label: '复杂度判定', x: 190, y: 472, w: 80, h: 26 },
  { id: 'web_search',           label: '联网搜索',   x: 322, y: 472, w: 80, h: 26 },
  // 并联生成节点
  { id: 'generate_simple',      label: '简单生成',   x: 112, y: 532, w: 80, h: 26 },
  { id: 'generate_complex',     label: '复杂生成',   x: 268, y: 532, w: 80, h: 26 },
  { id: 'check_hallucination',  label: '幻觉检测',   x: 190, y: 594, w: 80, h: 30 },
  // 缓存写回虚拟节点
  { id: 'cache_store',          label: '缓存写入',   x: 190, y: 654, w: 80, h: 30 },
]
const byId = (id: string): N => NODES.find(n => n.id === id)!

// 拆分展示节点对应的服务层数据键（cache_lookup 虚拟节点数据）
const DATA_KEY: Record<string, string> = {
  cache_exact: 'cache_lookup',
  cache_semantic: 'cache_lookup',
}
const CACHE_SUB_NODES = new Set(['cache_exact', 'cache_semantic'])

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
  // 文档评估的有效开关：仅当查询重写或联网搜索任一开启时才运行
  if (id === 'grade_documents') return gradeRuns()
  const k = ENABLED[id]
  return k ? (flow[k].value as boolean) : true
}

// 文档评估仅在查询重写或联网搜索任一开启时运行（两者都关则跳过，直接进生成判定）
function gradeRuns(): boolean {
  return (
    flow.enableGradeDocuments.value &&
    (flow.enableTransformQuery.value || flow.enableWebSearch.value)
  )
}

// ── 缓存拆分子节点状态（由 cache_lookup 节点数据驱动） ──
function cacheNodeState(id: string): NodeState | null {
  if (!CACHE_SUB_NODES.has(id)) return null
  const info = flow.nodeDataMap.value[DATA_KEY[id]]
  if (!info || typeof info.output !== 'object' || info.output === null) return null
  const out = info.output as Record<string, unknown>
  const checked = id === 'cache_exact' ? out.exact_checked : out.semantic_checked
  // 该层未执行（如精准命中后语义层无需运行）→ 回落默认状态，
  // 与其他未被调度到的节点一致显示为"待执行"
  return checked ? 'done' : null
}

function cacheBadgeText(id: string): string {
  if (!CACHE_SUB_NODES.has(id)) return ''
  const info = flow.nodeDataMap.value[DATA_KEY[id]]
  if (!info || typeof info.output !== 'object' || info.output === null) return ''
  const out = info.output as Record<string, unknown>
  const hit = id === 'cache_exact' ? out.exact_hit : out.semantic_hit
  const checked = id === 'cache_exact' ? out.exact_checked : out.semantic_checked
  // 该层未执行 → 不显示命中/未中徽标（节点保持单行"待执行"样式）
  if (checked !== true) return ''
  return hit ? '命中' : '未中'
}

function cacheBadgeColor(id: string): string {
  const text = cacheBadgeText(id)
  if (text === '命中') return '#34d399'
  if (text === '未中') return '#94a3b8'
  return ''
}

function state(id: string): NodeState {
  const sub = cacheNodeState(id)
  if (sub) return sub
  if (!enabled(id)) return 'disabled'
  if (flow.skippedNodes.value.includes(id)) return 'skipped'
  if (flow.completedNodes.value.includes(id)) return 'done'
  if (flow.currentNode.value === id) return 'active'
  return 'pending'
}

// ── 缓存命中高亮：命中时点亮 cache_lookup → cache_replay 分支，主链置灰 ──
const cacheHit = computed(() =>
  flow.completedNodes.value.includes('cache_replay')
)
const mainChainOpacity = computed(() => (cacheHit.value ? 0.15 : 1))

// ── 缓存拆分子节点分支高亮 ──
const cacheInfo = computed<Record<string, unknown> | null>(() => {
  const info = flow.nodeDataMap.value['cache_lookup']
  if (!info || typeof info.output !== 'object' || info.output === null) return null
  return info.output as Record<string, unknown>
})
const exactChecked = computed(() => cacheInfo.value?.exact_checked === true)
const exactHit = computed(() => cacheInfo.value?.exact_hit === true)
const semanticChecked = computed(() => cacheInfo.value?.semantic_checked === true)
const semanticHit = computed(() => cacheInfo.value?.semantic_hit === true)

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
  if (s === 'disabled' && id === 'grade_documents') {
    return '查询重写和联网搜索均关闭，文档评估被跳过'
  }
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
  const info = flow.nodeDataMap.value[DATA_KEY[id] ?? id]
  if (!info) return undefined
  // 缓存拆分子节点显示各自的执行耗时（精准/语义分层毫秒数）
  if (CACHE_SUB_NODES.has(id) && info.output && typeof info.output === 'object') {
    const out = info.output as Record<string, unknown>
    const ms = id === 'cache_exact' ? out.exact_ms : out.semantic_ms
    if (typeof ms === 'number') {
      return { ...info, durationMs: ms }
    }
    // 该层未执行（如精准命中后语义层无需运行）→ 不显示耗时
    return { input: info.input, output: info.output }
  }
  return info
}

function durationColor(ms: number): string {
  if (ms < 200) return 'bg-emerald-900/40 text-emerald-400'
  if (ms < 1000) return 'bg-amber-900/40 text-amber-400'
  return 'bg-red-900/40 text-red-400'
}

function durationFill(ms: number): string {
  if (ms < 200) return '#34d399'
  if (ms < 1000) return '#fbbf24'
  return '#f87171'
}

function nodeDuration(id: string): number | undefined {
  return flow.nodeDataMap.value[DATA_KEY[id] ?? id]?.durationMs
}

function nodeDurationText(id: string): string {
  const ms = nodeDuration(id)
  return ms != null ? `${ms.toFixed(0)}ms` : ''
}

function nodeDurationColor(id: string): string {
  const ms = nodeDuration(id)
  return ms != null ? durationFill(ms) : ''
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
  <aside class="w-80 flex-shrink-0 glass border-r border-slate-700/50 flex flex-col h-full">
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

      <svg viewBox="0 0 460 740" class="w-full" style="max-height: 580px">
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
        <rect x="195" y="10" width="70" height="26" rx="13" ry="13" fill="#1e293b" stroke="#475569" stroke-width="1.5"/>
        <text x="230" y="23" text-anchor="middle" dominant-baseline="central" class="text-[13px] fill-slate-400">START</text>

        <!-- ═══ START → cache_exact ═══ -->
        <line x1="230" y1="36" x2="230" y2="50" stroke="#475569" stroke-width="1.5" fill="none" marker-end="url(#arr)"/>

        <!-- ═══ cache_exact → cache_semantic（精准未命中，纵向流转到语义层） ═══ -->
        <line :x1="byId('cache_exact').x + byId('cache_exact').w/2"
              :y1="byId('cache_exact').y + byId('cache_exact').h"
              :x2="byId('cache_semantic').x + byId('cache_semantic').w/2"
              :y2="byId('cache_semantic').y - 2"
          stroke="#475569" stroke-width="1.5" fill="none"
          :opacity="exactChecked && !exactHit ? 1 : 0.12" marker-end="url(#arr)"/>

        <!-- ═══ cache_exact → cache_replay（精准命中，琥珀色虚线高亮） ═══ -->
        <path :d="`M ${byId('cache_exact').x + byId('cache_exact').w} ${byId('cache_exact').y + byId('cache_exact').h/2} L ${byId('cache_replay').x - 2} ${byId('cache_replay').y + 8}`" fill="none"
          :stroke="exactHit ? '#f59e0b' : '#475569'" stroke-width="1.4" stroke-dasharray="5 3"
          :opacity="exactHit ? 0.9 : 0.04" marker-end="url(#arrAmber)"/>

        <!-- ═══ cache_semantic → cache_replay（语义命中，琥珀色虚线高亮） ═══ -->
        <path :d="`M ${byId('cache_semantic').x + byId('cache_semantic').w} ${byId('cache_semantic').y + byId('cache_semantic').h/2} L ${byId('cache_replay').x - 2} ${byId('cache_replay').y + byId('cache_replay').h - 8}`" fill="none"
          :stroke="semanticHit ? '#f59e0b' : '#475569'" stroke-width="1.4" stroke-dasharray="5 3"
          :opacity="semanticHit ? 0.9 : 0.04" marker-end="url(#arrAmber)"/>

        <!-- ═══ cache_replay → END（右侧下行通道） ═══ -->
        <path d="M 372 105 L 430 105 L 430 694 L 230 694 L 230 704" fill="none" stroke="#475569"
          stroke-width="1.5" :opacity="cacheHit ? 1 : 0.25" marker-end="url(#arr)"/>

        <!-- ═══ 主链（命中时整体置灰） ═══ -->
        <g :opacity="mainChainOpacity">

        <!-- ═══ cache_semantic → analyze_kg_intent（全部未命中，主链入口） ═══ -->
        <line :x1="byId('cache_semantic').x + byId('cache_semantic').w/2"
              :y1="byId('cache_semantic').y + byId('cache_semantic').h"
              :x2="byId('analyze_kg_intent').x + byId('analyze_kg_intent').w/2"
              :y2="byId('analyze_kg_intent').y - 2"
          stroke="#475569" stroke-width="1.5" fill="none" marker-end="url(#arr)"/>

        <!-- ═══ 意图分析 → 检索总线（语义必选，其余可选扇出） ═══ -->
        <g stroke="#475569" stroke-width="1.5" fill="none">
          <!-- 意图分析 底部连到总线 -->
          <line :x1="byId('analyze_kg_intent').x + byId('analyze_kg_intent').w/2"
                :y1="byId('analyze_kg_intent').y + byId('analyze_kg_intent').h"
                :x2="byId('analyze_kg_intent').x + byId('analyze_kg_intent').w/2"
                y2="220"/>
          <!-- 总线水平 -->
          <line x1="84" y1="220" x2="366" y2="220"/>
          <!-- 总线 → 语义检索（必选，始终可见） -->
          <line x1="84"  y1="220" x2="84"  y2="230" marker-end="url(#arrCyan)"/>
          <!-- 总线 → bm25（可选，关闭时连线熄灭） -->
          <line x1="172" y1="220" x2="172" y2="230" :opacity="enabled('bm25_retrieve') ? 1 : 0.12" marker-end="url(#arrCyan)"/>
          <!-- 总线 → 多角度查询（可选，关闭时连线熄灭） -->
          <line x1="264" y1="220" x2="264" y2="230" :opacity="enabled('multi_query_retrieve') ? 1 : 0.12" marker-end="url(#arrCyan)"/>
          <!-- 总线 → 图谱检索（意图分析自动，关闭时连线熄灭） -->
          <line x1="366" y1="220" x2="366" y2="230" :opacity="enabled('kg_retrieve') ? 1 : 0.12" marker-end="url(#arrCyan)"/>
        </g>

        <!-- ═══ 4 路检索 → merge 收敛（可选策略关闭时对应连线熄灭） ═══ -->
        <g stroke="#475569" stroke-width="1.5" fill="none">
          <!-- 语义检索 → merge -->
          <line x1="84"  y1="256" x2="190" y2="292" marker-end="url(#arr)"/>
          <!-- bm25 → merge -->
          <line x1="172" y1="256" x2="210" y2="292" :opacity="enabled('bm25_retrieve') ? 1 : 0.12" marker-end="url(#arr)"/>
          <!-- 多角度查询 → merge -->
          <line x1="264" y1="256" x2="250" y2="292" :opacity="enabled('multi_query_retrieve') ? 1 : 0.12" marker-end="url(#arr)"/>
          <!-- 图谱检索 → merge -->
          <line x1="366" y1="256" x2="270" y2="292" :opacity="enabled('kg_retrieve') ? 1 : 0.12" marker-end="url(#arr)"/>
        </g>

        <!-- ═══ 主链箭头 merge → rerank → grade → judge → [simple|complex] → check → cache_store → END ═══ -->
        <g stroke="#475569" stroke-width="1.5" fill="none">
          <!-- merge → rerank -->
          <line x1="230" :y1="byId('parallel_retrieve_merge').y + byId('parallel_retrieve_merge').h"
                x2="230" :y2="byId('rerank_documents').y"
            marker-end="url(#arr)" :opacity="mainEdgeOpacity('merge_to_rerank')"/>
          <!-- rerank → grade -->
          <line x1="230" :y1="byId('rerank_documents').y + byId('rerank_documents').h"
                x2="230" :y2="byId('grade_documents').y"
            marker-end="url(#arr)" :opacity="mainEdgeOpacity('rerank_to_grade')"/>
          <!-- grade → judge_complexity（相关） -->
          <line x1="230" :y1="byId('grade_documents').y + byId('grade_documents').h"
                x2="230" :y2="byId('judge_complexity').y"
            marker-end="url(#arr)" :opacity="mainEdgeOpacity('grade_to_gen')"/>
          <!-- judge → generate_simple（SIMPLE 分支左） -->
          <path :d="`M 230 ${byId('judge_complexity').y + byId('judge_complexity').h} L 152 ${byId('generate_simple').y - 2}`" marker-end="url(#arr)" />
          <!-- judge → generate_complex（COMPLEX 分支右） -->
          <path :d="`M 230 ${byId('judge_complexity').y + byId('judge_complexity').h} L 308 ${byId('generate_complex').y - 2}`" marker-end="url(#arr)" />
          <!-- generate_simple → check（左汇聚） -->
          <path :d="`M 152 ${byId('generate_simple').y + byId('generate_simple').h} L 200 ${byId('check_hallucination').y}`" :opacity="enabled('check_hallucination') ? 1 : 0.12" />
          <!-- generate_complex → check（右汇聚） -->
          <path :d="`M 308 ${byId('generate_complex').y + byId('generate_complex').h} L 260 ${byId('check_hallucination').y}`" :opacity="enabled('check_hallucination') ? 1 : 0.12" />
          <!-- 汇聚 → check_hallucination -->
          <line x1="230" y1="582" x2="230" :y2="byId('check_hallucination').y" :opacity="enabled('check_hallucination') ? 1 : 0.12" marker-end="url(#arr)"/>
        </g>

        <!-- ═══ Bypass 绕过线（右侧虚线弧线） ═══ -->
        <!-- case 4: merge → 绕过 rerank → grade -->
        <path :d="`M 270 ${byId('parallel_retrieve_merge').y + byId('parallel_retrieve_merge').h/2} C 296 ${byId('parallel_retrieve_merge').y + byId('parallel_retrieve_merge').h/2}, 296 ${byId('grade_documents').y + byId('grade_documents').h/2}, 270 ${byId('grade_documents').y + byId('grade_documents').h/2}`"
          fill="none" :stroke="bypassColor('rerank')" stroke-width="1.2" stroke-dasharray="4 3"
          :opacity="bypassOpacity('rerank')" marker-end="url(#arrAmber)"/>
        <!-- case 3: rerank → 绕过 grade → judge_complexity -->
        <path :d="`M 270 ${byId('rerank_documents').y + byId('rerank_documents').h/2} C 296 ${byId('rerank_documents').y + byId('rerank_documents').h/2}, 296 ${byId('judge_complexity').y + byId('judge_complexity').h/2}, 270 ${byId('judge_complexity').y + byId('judge_complexity').h/2}`"
          fill="none" :stroke="bypassColor('grade')" stroke-width="1.2" stroke-dasharray="4 3"
          :opacity="bypassOpacity('grade')" marker-end="url(#arrAmber)"/>
        <!-- case 2: merge → 绕过 rerank 和 grade → judge_complexity -->
        <path :d="`M 270 ${byId('parallel_retrieve_merge').y + byId('parallel_retrieve_merge').h/2} C 306 ${byId('parallel_retrieve_merge').y + byId('parallel_retrieve_merge').h/2}, 306 ${byId('judge_complexity').y + byId('judge_complexity').h/2}, 270 ${byId('judge_complexity').y + byId('judge_complexity').h/2}`"
          fill="none" stroke="#f59e0b" stroke-width="1.2" stroke-dasharray="4 3"
          :opacity="bothOff() ? 0.55 : 0.04" marker-end="url(#arrAmber)"/>
        <!-- generate_simple/complex → 绕过 check → cache_store -->
        <path :d="`M 270 ${byId('check_hallucination').y - 14} C 300 ${byId('check_hallucination').y - 14}, 300 ${byId('cache_store').y + byId('cache_store').h/2}, 270 ${byId('cache_store').y + byId('cache_store').h/2}`"
          fill="none" :stroke="reflectionBypassColor()" stroke-width="1.2" stroke-dasharray="4 3"
          :opacity="reflectionBypassOpacity()" marker-end="url(#arrAmber)"/>

        <!-- ═══ Bypass 标签 ═══ -->
        <text x="278" :y="byId('rerank_documents').y + byId('rerank_documents').h/2 - 5" :opacity="bypassOpacity('rerank')" :fill="bypassColor('rerank')" class="text-[10px]">绕过</text>
        <text x="278" :y="byId('grade_documents').y + byId('grade_documents').h + 14" :opacity="bypassOpacity('grade')" :fill="bypassColor('grade')" class="text-[10px]">绕过</text>
        <text x="288" :y="(byId('parallel_retrieve_merge').y + byId('parallel_retrieve_merge').h/2 + byId('judge_complexity').y + byId('judge_complexity').h/2) / 2 - 2" :opacity="bothOff() ? 0.6 : 0.04" fill="#f59e0b" class="text-[10px]">均关</text>
        <text x="282" :y="(byId('check_hallucination').y - 14 + byId('cache_store').y + byId('cache_store').h/2) / 2 - 4" :opacity="reflectionBypassOpacity()" :fill="reflectionBypassColor()" class="text-[10px]">绕过</text>

        <!-- ═══ 分支：grade → web_search / transform_query ═══ -->
        <g stroke="#475569" stroke-width="1.5" fill="none">
          <!-- grade → transform_query（左） -->
          <path :d="`M ${byId('grade_documents').x} ${byId('grade_documents').y + byId('grade_documents').h/2} C ${byId('grade_documents').x} ${byId('grade_documents').y + 40}, ${byId('transform_query').x + byId('transform_query').w/2} ${byId('transform_query').y - 12}, ${byId('transform_query').x + byId('transform_query').w/2} ${byId('transform_query').y}`"
            marker-end="url(#arr)" :opacity="gradeRuns() && enabled('transform_query') ? 1 : 0.12"/>
          <!-- grade → web_search（右） -->
          <path :d="`M ${byId('grade_documents').x + byId('grade_documents').w} ${byId('grade_documents').y + byId('grade_documents').h/2} C ${byId('grade_documents').x + byId('grade_documents').w} ${byId('grade_documents').y + 40}, ${byId('web_search').x + byId('web_search').w/2} ${byId('web_search').y - 12}, ${byId('web_search').x + byId('web_search').w/2} ${byId('web_search').y}`"
            marker-end="url(#arr)" :opacity="gradeRuns() && enabled('web_search') ? 1 : 0.12"/>
        </g>

        <!-- ═══ web_search → judge_complexity ═══ -->
        <path :d="`M ${byId('web_search').x + byId('web_search').w/2} ${byId('web_search').y + byId('web_search').h} L ${byId('web_search').x + byId('web_search').w/2} 510 L 270 510 L 270 ${byId('judge_complexity').y + byId('judge_complexity').h/2}`"
          stroke="#475569" stroke-width="1.5" fill="none"
          :opacity="enabled('web_search') ? 1 : 0.12" marker-end="url(#arr)"/>

        <!-- ═══ 回环曲线 ═══ -->
        <!-- transform_query → retrieve（循环回检索） -->
        <path :d="`M ${byId('transform_query').x} ${byId('transform_query').y + byId('transform_query').h/2} C ${byId('transform_query').x} ${byId('transform_query').y + 36}, 24 ${byId('transform_query').y + 36}, 24 300 C 24 270, 60 256, 84 256`"
          fill="none" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="4 2"
          :opacity="enabled('transform_query') ? 0.85 : 0.12" marker-end="url(#arrAmber)"/>

        <!-- ═══ 分支标签 ═══ -->
        <text x="196" :y="byId('grade_documents').y + 46" :opacity="gradeRuns() ? 1 : 0" class="text-[10px] fill-green-500/70">相关</text>
        <text x="286" :y="byId('grade_documents').y + 24" :opacity="gradeRuns() ? 1 : 0" class="text-[10px] fill-orange-400/70">不相关+联网</text>
        <text x="100" :y="byId('grade_documents').y + 24" :opacity="gradeRuns() ? 1 : 0" class="text-[10px] fill-orange-400/70">不相关</text>
        <!-- judge 分支标签 -->
        <text x="140" :y="byId('judge_complexity').y + 20" class="text-[10px] fill-cyan-400/70">SIMPLE</text>
        <text x="288" :y="byId('judge_complexity').y + 20" class="text-[10px] fill-amber-400/70">COMPLEX</text>

        <!-- ═══ 回环标签 ═══ -->
        <text x="14" y="390" class="text-[10px] fill-amber-500/80" transform="rotate(-90 14 390)">回检索</text>

        <!-- ═══ END ═══ -->
        <rect x="195" y="704" width="70" height="26" rx="13" ry="13" fill="#1e293b" stroke="#475569" stroke-width="1.5"/>
        <text x="230" y="717" text-anchor="middle" dominant-baseline="central" class="text-[13px] fill-slate-400">END</text>

        <!-- ═══ check → cache_store → END ═══ -->
        <line x1="230" :y1="byId('check_hallucination').y + byId('check_hallucination').h"
              x2="230" :y2="byId('cache_store').y"
          stroke="#475569" stroke-width="1.5" fill="none" :opacity="enabled('check_hallucination') ? 1 : 0.12" marker-end="url(#arr)"/>
        <line x1="230" :y1="byId('cache_store').y + byId('cache_store').h"
              x2="230" y2="704"
          stroke="#475569" stroke-width="1.5" fill="none" marker-end="url(#arr)"/>

        </g>

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
          <template v-if="cacheBadgeText(n.id)">
            <!-- 缓存拆分子节点：两行布局（节点名 + 命中/未中/跳过徽标） -->
            <text
              :x="n.x + n.w/2" :y="n.y + 10"
              text-anchor="middle" dominant-baseline="central"
              :fill="TEXT[state(n.id)]"
              class="text-[11px]"
            >{{ n.label }}</text>
            <text
              :x="n.x + n.w/2" :y="n.y + 19"
              text-anchor="middle" dominant-baseline="central"
              :fill="cacheBadgeColor(n.id)"
              class="text-[9px] font-mono"
            >{{ cacheBadgeText(n.id) }}</text>
          </template>
          <text
            v-else
            :x="n.x + n.w/2" :y="n.y + n.h/2"
            text-anchor="middle" dominant-baseline="central"
            :fill="TEXT[state(n.id)]"
            :style="state(n.id) === 'disabled' ? 'text-decoration: line-through' : ''"
            class="text-[13px]"
          >{{ n.label }}</text>
          <text v-if="state(n.id) === 'done'"
            :x="n.x + n.w - 6" :y="n.y + n.h/2"
            text-anchor="middle" dominant-baseline="central"
            class="text-[10px] fill-emerald-300"
          >✓</text>
          <text v-if="state(n.id) === 'done' && nodeDurationText(n.id) && !CACHE_SUB_NODES.has(n.id)"
            :x="n.x + n.w + 3" :y="n.y + n.h/2"
            text-anchor="start" dominant-baseline="central"
            :fill="nodeDurationColor(n.id)"
            class="text-[9px] font-mono"
          >{{ nodeDurationText(n.id) }}</text>
        </g>
      </svg>

      <!-- ════════════════════════════════════════════════════════════
               节点信息 Popover（Teleport 到 body）
               自适应渲染：string / string[] / Record<string, unknown>
               ════════════════════════════════════════════════════════════ -->
      <Teleport to="body">
        <div v-if="flow.selectedNodeId.value"
          class="node-popover fixed z-[9999] bg-slate-800 border border-slate-600 rounded-lg shadow-2xl w-72 overflow-hidden"
          :style="{ left: popoverLeft + 'px', top: popoverTop + 'px' }">
          <!-- 标题 + 关闭 -->
          <div class="flex items-center justify-between px-3 py-2 border-b border-slate-700">
            <span class="text-xs font-semibold text-slate-200">
              {{ byId(flow.selectedNodeId.value)?.label }}
            </span>
            <button @click.stop="closePopover"
              class="p-0.5 rounded hover:bg-slate-700 text-slate-500 hover:text-slate-300 transition-colors">
              <X class="w-3 h-3" />
            </button>
          </div>
          <!-- 状态 + 执行时间 -->
          <div class="px-3 py-1.5 border-b border-slate-700/50 flex items-center gap-3">
            <span class="text-[10px] text-slate-500">状态</span>
            <span class="text-[11px]" :class="nodeStatusColor(flow.selectedNodeId.value)">
              {{ nodeStatusText(flow.selectedNodeId.value) }}
            </span>
            <span v-if="selectedNodeInfo?.durationMs" class="ml-auto">
              <span class="text-[9px] font-mono px-1.5 py-0.5 rounded"
                :class="durationColor(selectedNodeInfo.durationMs)">
                {{ (selectedNodeInfo.durationMs).toFixed(1) }} ms
              </span>
            </span>
          </div>
          <!-- 已执行 → 展示 I/O -->
          <template v-if="isNodeExecuted(flow.selectedNodeId.value)">
            <template v-if="selectedNodeInfo">
              <!-- 输入 -->
              <div class="px-3 py-2 border-b border-slate-700/30">
                <div class="flex items-center gap-1 cursor-pointer select-none hover:bg-slate-700/30 rounded px-1 -mx-1"
                  @click.stop="popoverShowInput = !popoverShowInput">
                  <span class="text-[10px] font-mono text-blue-400/70">{{ popoverShowInput ? '▾' : '▸' }}</span>
                  <span class="text-[10px] font-medium text-blue-400/70">输入</span>
                </div>
                <div v-if="popoverShowInput" class="mt-1">
                  <NodeDataRenderer :val="selectedNodeInfo.input" />
                </div>
              </div>
              <!-- 输出 -->
              <div class="px-3 py-2">
                <div class="flex items-center gap-1 cursor-pointer select-none hover:bg-slate-700/30 rounded px-1 -mx-1"
                  @click.stop="popoverShowOutput = !popoverShowOutput">
                  <span class="text-[10px] font-mono text-emerald-400/70">{{ popoverShowOutput ? '▾' : '▸' }}</span>
                  <span class="text-[10px] font-medium text-emerald-400/70">输出</span>
                </div>
                <div v-if="popoverShowOutput" class="mt-1">
                  <NodeDataRenderer :val="selectedNodeInfo.output" />
                </div>
              </div>
            </template>
            <div v-else class="px-3 py-2">
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
