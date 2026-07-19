<script setup lang="ts">
import AppLayout from './components/layout/AppLayout.vue'
import HealthBar from './components/common/HealthBar.vue'
</script>

<template>
  <div class="flex h-screen overflow-hidden">
    <AppLayout>
      <template #top>
        <HealthBar />
      </template>
      <template #default>
        <router-view />
      </template>
    </AppLayout>
  </div>
</template>
