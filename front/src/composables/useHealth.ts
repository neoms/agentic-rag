import { ref, onMounted, onUnmounted } from 'vue'
import type { HealthResponse } from '../types'
import { getHealth } from '../api/health'

export function useHealth() {
  const health = ref<HealthResponse | null>(null)
  const error = ref<string | null>(null)
  const loading = ref(false)
  let timer: ReturnType<typeof setInterval> | null = null

  async function fetchHealth() {
    loading.value = true
    error.value = null
    const result = await getHealth()
    if (result) {
      health.value = result
      error.value = null
    } else {
      health.value = null
      error.value = '无法获取服务状态'
    }
    loading.value = false
  }

  onMounted(() => {
    fetchHealth()
    timer = setInterval(fetchHealth, 30000)
  })

  onUnmounted(() => {
    if (timer) {
      clearInterval(timer)
      timer = null
    }
  })

  return { health, error, loading, refresh: fetchHealth }
}
