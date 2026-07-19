<script setup lang="ts">
import Sidebar from './Sidebar.vue'
</script>

<template>
  <Sidebar />
  <div class="flex-1 flex flex-col min-w-0 relative">
    <header v-if="$slots.top" class="flex-shrink-0">
      <slot name="top" />
    </header>
    <main class="flex-1 overflow-hidden relative">
      <slot />
    </main>
  </div>
</template>
