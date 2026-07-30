<template>
  <div v-if="authStore.isResolving" class="auth-bootstrap" role="status">正在确认身份…</div>
  <AppShell v-else-if="authStore.isLoggedIn" />
  <router-view v-else />
</template>

<script setup>
import AppShell from './AppShell.vue';
import { useAuthStore } from '../store/auth';

const authStore = useAuthStore();
</script>

<style scoped>
.auth-bootstrap {
  min-height: 100vh;
  display: grid;
  place-items: center;
  background: var(--color-paper);
  color: var(--color-ink-soft);
  font-size: 14px;
}
</style>
