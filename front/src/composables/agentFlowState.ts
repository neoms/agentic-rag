/** Agent 流程图全局共享状态
 *
 * 使用独立 ref() 而非 reactive()，确保跨模块的数组赋值能正确触发响应式。
 */
import { ref } from 'vue'
import type { Ref } from 'vue'

export const completedNodes: Ref<string[]> = ref([])
export const currentNode: Ref<string | null> = ref(null)
/** Config-disabled 之外的 runtime skip (e.g. enable_kg=true but kg_intent=false) */
export const skippedNodes: Ref<string[]> = ref([])
export const enableRerank: Ref<boolean> = ref(true)
export const enableGradeDocuments: Ref<boolean> = ref(true)
export const enableTransformQuery: Ref<boolean> = ref(true)
export const enableWebSearch: Ref<boolean> = ref(false)
export const enableReflection: Ref<boolean> = ref(true)
export const enableBm25: Ref<boolean> = ref(true)
export const enableMultiQuery: Ref<boolean> = ref(false)

/** 每轮提问的节点 I/O 数据记录（仅前端，新提问时清空）
 *  key = 节点 ID (如 'retrieve')
 *  value = { input: string | string[], output: string | string[] }
 *  当为 string[] 时，前端以列表形式逐条展示
 */
export type NodeDataInfo = {
  input: string | string[];
  output: string | string[];
}
export const nodeDataMap: Ref<Record<string, NodeDataInfo>> = ref({})

/** 当前选中的流程图节点 ID（用于 popover 展示） */
export const selectedNodeId: Ref<string | null> = ref(null)
