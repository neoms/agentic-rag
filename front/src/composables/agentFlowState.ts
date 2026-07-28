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
export const enableBm25: Ref<boolean> = ref(false)
export const enableHyde: Ref<boolean> = ref(false)
export const enableMultiQuery: Ref<boolean> = ref(false)
export const enableKg: Ref<boolean> = ref(false)
