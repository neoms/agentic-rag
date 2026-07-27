/** Agent 流程图全局共享状态
 *
 * 使用独立 ref() 而非 reactive()，确保跨模块的数组赋值能正确触发响应式。
 */
import { ref } from 'vue'
import type { Ref } from 'vue'

export const completedNodes: Ref<string[]> = ref([])
export const currentNode: Ref<string | null> = ref(null)
export const enableRerank: Ref<boolean> = ref(true)
export const enableGradeDocuments: Ref<boolean> = ref(true)
export const enableTransformQuery: Ref<boolean> = ref(true)
export const enableWebSearch: Ref<boolean> = ref(false)
export const enableReflection: Ref<boolean> = ref(true)
