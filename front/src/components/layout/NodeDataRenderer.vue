<script setup lang="ts">
import { ref } from 'vue'

/**
 * 递归渲染节点 I/O 数据（可读性优化版）
 *
 * - Object: key-value 键值对，嵌套对象可折叠
 * - Array: 计数徽章 + 列表项
 * - String: 短文本直接展示，长文本预览 + [展开全文]
 * - Number / Boolean: 内联展示
 */

const props = defineProps<{ val: unknown; depth?: number }>()

// 展开状态
const expandedObjects = ref<Record<string, boolean>>({})
const expandedContent = ref<Record<string, boolean>>({})

function toggleObj(key: string) {
  expandedObjects.value[key] = !expandedObjects.value[key]
}
function toggleContent(key: string) {
  expandedContent.value[key] = !expandedContent.value[key]
}

// ── 工具函数 ──
function isArray(v: unknown): v is unknown[] {
  return Array.isArray(v)
}
function isPlainObject(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === 'object' && !Array.isArray(v)
}
function isPrimitive(v: unknown): v is string | number | boolean {
  return typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean'
}

const MAX_PREVIEW_LEN = 200
const isLongString = (s: string) => s.length > MAX_PREVIEW_LEN
const truncate = (s: string) => s.slice(0, MAX_PREVIEW_LEN) + '...'

/** 将 backend 下划线/驼峰 key 格式化为可读标签 */
function formatLabel(k: string): string {
  const LABELS: Record<string, string> = {
    query: '查询',
    method: '检索方法',
    result_count: '结果数量',
    total_documents: '文档总数',
    documents: '文档列表',
    source: '来源',
    content: '文档内容',
    content_length: '内容长度',
    score: '相关性评分',
    metadata: '元数据',
    note: '说明',
    context: '检索上下文',
    context_length: '上下文长度',
    has_result: '检索结果',
    kg_intent: 'KG 意图',
    explanation: '判定说明',
    strategies: '检索策略',
    strategies_count: '策略数量',
    original_query: '原始查询',
    rewritten_query: '改写后查询',
    changed: '已改写',
    iteration: '迭代次数',
    max_iterations: '最大迭代',
    all_relevant: '全部相关',
    verdict: '判定结论',
    action: '后续动作',
    total_web_results: '联网结果数',
    total_results: '搜索结果数',
    results: '搜索结果',
    documents_count: '文档数量',
    reranked_count: '重排数量',
    context_documents: '上下文文档',
    context_documents_count: '上下文文档数',
    answer_length: '回答长度',
    answer: '生成回答',
    faithfulness_score: '忠实度分数',
    passed: '检测通过',
    kg_retrieve: '图谱检索',
    retrieval_method: '检索方法',
    durationMs: '执行时间',
  }
  return LABELS[k] ?? k.replace(/_/g, ' ').replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/^\w/, c => c.toUpperCase())
}

/** 格式化数值单位 */
function formatValue(v: number): string {
  if (typeof v !== 'number') return String(v)
  return v >= 1000 ? `${(v / 1000).toFixed(1)}k` : String(v)
}

function pluralize(count: number, noun: string): string {
  return `${count} ${noun}${count !== 1 ? '' : ''}`
}
</script>

<template>
  <!-- ═══════════════ Object: key-value 键值对 ═══════════════ -->
  <template v-if="isPlainObject(val)">
    <div class="space-y-1 max-h-60 overflow-y-auto">
      <div v-for="(v, rawKey) in val" :key="String(rawKey)"
        class="rounded px-1.5 py-0.5 hover:bg-slate-800/60 transition-colors">

        <!-- 嵌套对象 → 可折叠区块 -->
        <template v-if="isPlainObject(v)">
          <div class="flex items-center gap-1 cursor-pointer select-none"
            @click.stop="toggleObj(String(rawKey))">
            <span class="text-[9px] text-slate-500 transition-transform"
              :class="expandedObjects[String(rawKey)] ? 'rotate-90' : ''">▶</span>
            <span class="text-[10px] font-medium text-slate-300">
              {{ formatLabel(String(rawKey)) }}
            </span>
            <span v-if="!expandedObjects[String(rawKey)]"
              class="text-[9px] text-slate-600 ml-1">
              {{ Object.keys(v).length }} 个字段
            </span>
          </div>
          <div v-if="expandedObjects[String(rawKey)]" class="ml-2 mt-0.5 border-l border-slate-700/40 pl-2">
            <NodeDataRenderer :val="v" :depth="(depth ?? 0) + 1" />
          </div>
        </template>

        <!-- 数组 → 计数 + 列表 -->
        <template v-else-if="isArray(v)">
          <div class="flex items-center gap-1">
            <span class="text-[10px] font-medium text-slate-300">
              {{ formatLabel(String(rawKey)) }}
            </span>
            <span v-if="v.length > 0"
              class="text-[9px] font-mono px-1 rounded bg-slate-700/50 text-slate-400">
              {{ v.length }}
            </span>
            <span v-else class="text-[9px] text-slate-600 italic">无</span>
          </div>
          <div v-if="v.length > 0" class="mt-0.5 space-y-0.5 ml-1">
            <NodeDataRenderer :val="v" :depth="(depth ?? 0) + 1" />
          </div>
        </template>

        <!-- 长字符串 → 预览 + 展开 -->
        <template v-else-if="typeof v === 'string' && isLongString(v)">
          <div class="flex items-start gap-1">
            <span class="text-[9px] text-slate-500 font-mono shrink-0 mt-0.5">
              {{ formatLabel(String(rawKey)) }}:
            </span>
            <div class="min-w-0 flex-1">
              <p v-if="!expandedContent[String(rawKey)]"
                class="text-[10px] text-slate-300 leading-relaxed whitespace-pre-wrap break-all">
                {{ truncate(v) }}
                <button @click.stop="toggleContent(String(rawKey))"
                  class="text-[9px] text-amber-400/80 hover:text-amber-300 ml-0.5 underline decoration-dotted">
                  展开全文
                </button>
              </p>
              <div v-else>
                <p class="text-[10px] text-slate-300 leading-relaxed whitespace-pre-wrap break-all max-h-40 overflow-y-auto">
                  {{ v }}
                </p>
                <button @click.stop="toggleContent(String(rawKey))"
                  class="text-[9px] text-amber-400/80 hover:text-amber-300 mt-0.5 underline decoration-dotted">
                  收起
                </button>
              </div>
            </div>
          </div>
        </template>

        <!-- 普通 primitive 值 → 内联 -->
        <template v-else>
          <div class="flex items-start gap-1">
            <span class="text-[9px] text-slate-500 font-mono shrink-0 mt-0.5">
              {{ formatLabel(String(rawKey)) }}:
            </span>
            <span v-if="isPrimitive(v)"
              class="text-[10px] text-slate-200 leading-relaxed break-all">
              {{ typeof v === 'number' ? formatValue(v) : String(v) }}
            </span>
            <span v-else-if="v === null"
              class="text-[10px] text-slate-600 italic">无</span>
            <span v-else
              class="text-[10px] text-slate-400">{{ String(v) }}</span>
          </div>
        </template>
      </div>
    </div>
  </template>

  <!-- ═══════════════ Array: 列表 ═══════════════ -->
  <ul v-else-if="isArray(val)" class="space-y-0.5 max-h-60 overflow-y-auto">
    <li v-for="(item, i) in val" :key="i"
      class="rounded px-1.5 py-0.5 hover:bg-slate-800/60 transition-colors">

      <!-- 对象数组元素 → 可折叠卡片 -->
      <template v-if="isPlainObject(item)">
        <div class="flex items-center gap-1 cursor-pointer select-none"
          @click.stop="toggleObj(`arr_${i}`)">
          <span class="text-[9px] text-sky-400/70 transition-transform"
            :class="expandedObjects[`arr_${i}`] ? 'rotate-90' : ''">▶</span>
          <!-- 优先展示 source/文件名作为标题 -->
          <span class="text-[10px] font-medium text-sky-300">
            {{ (item.source as string) || `#${i + 1}` }}
          </span>
          <span v-if="item.score != null" 
            class="text-[9px] font-mono px-1 rounded bg-sky-900/30 text-sky-400">
            {{ (item.score as number).toFixed(3) }}
          </span>
          <span v-if="item.content_length"
            class="text-[9px] text-slate-500 ml-auto">
            {{ formatValue(item.content_length as number) }} 字符
          </span>
        </div>
        <div v-if="expandedObjects[`arr_${i}`]" class="ml-2 mt-0.5 border-l border-sky-700/30 pl-2">
          <NodeDataRenderer :val="item" :depth="(depth ?? 0) + 1" />
        </div>
      </template>

      <!-- 数组元素为 primitive -->
      <span v-else-if="isPrimitive(item)"
        class="text-[10px] text-slate-300 whitespace-pre-wrap break-all">
        {{ item }}
      </span>

      <!-- 数组元素为嵌套数组 -->
      <template v-else-if="isArray(item)">
        <div class="flex items-center gap-1 cursor-pointer select-none"
          @click.stop="toggleObj(`arr_${i}`)">
          <span class="text-[9px] text-slate-500 transition-transform"
            :class="expandedObjects[`arr_${i}`] ? 'rotate-90' : ''">▶</span>
          <span class="text-[10px] text-slate-400">#{{ i + 1 }}</span>
          <span class="text-[9px] text-slate-600">({{ item.length }} 项)</span>
        </div>
        <div v-if="expandedObjects[`arr_${i}`]" class="ml-2 mt-0.5">
          <NodeDataRenderer :val="item" />
        </div>
      </template>

      <!-- 兜底 -->
      <span v-else class="text-[10px] text-slate-500">{{ String(item) }}</span>
    </li>
  </ul>

  <!-- ═══════════════ String: 纯文本 ═══════════════ -->
  <p v-else-if="typeof val === 'string'"
    class="text-[10px] text-slate-300 leading-relaxed whitespace-pre-wrap break-all
      bg-slate-900/50 rounded px-1.5 py-1 max-h-52 overflow-y-auto">
    {{ val }}
  </p>

  <!-- ═══════════════ 兜底 ═══════════════ -->
  <p v-else class="text-[10px] text-slate-500 italic">无数据</p>
</template>
