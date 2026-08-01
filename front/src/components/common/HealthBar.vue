<script setup lang="ts">
import { computed } from 'vue'
import { useHealth } from '../../composables/useHealth'
import { Activity, RefreshCw } from 'lucide-vue-next'

const { health, error, loading, refresh } = useHealth()

const statusColor = computed(() => {
  if (error.value) return 'text-red-400'
  if (health.value?.status === 'degraded') return 'text-amber-400'
  return 'text-emerald-400'
})

const statusText = computed(() => {
  if (error.value) return '离线'
  if (health.value?.status === 'degraded') return '降级'
  return '正常'
})

const statusBg = computed(() => {
  if (error.value) return 'bg-red-400/10'
  if (health.value?.status === 'degraded') return 'bg-amber-400/10'
  return 'bg-emerald-400/10'
})
</script>

<template>
  <div class="glass border-b border-slate-700/50">
    <div class="flex items-center justify-between px-6 py-3">
      <div class="flex items-center gap-4">
        <!-- 状态指示 -->
        <div class="flex items-center gap-2">
          <div class="flex items-center gap-1.5" :class="[statusBg, 'px-2.5 py-1 rounded-full']">
            <Activity :class="['w-3.5 h-3.5', statusColor]" />
            <span class="text-xs font-medium" :class="statusColor">{{ statusText }}</span>
          </div>
          <button
            @click="refresh"
            :class="['p-1 rounded-md transition-colors hover:bg-slate-700/50', loading && 'animate-spin']"
            title="刷新状态"
          >
            <RefreshCw class="w-3.5 h-3.5 text-slate-500" />
          </button>
        </div>

        <!-- 错误状态 -->
        <div v-if="error" class="text-xs text-red-400 flex items-center gap-1.5">
          <span>{{ error }}</span>
        </div>
      </div>
    </div>
  </div>
</template>
