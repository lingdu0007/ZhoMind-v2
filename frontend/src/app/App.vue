<template>
  <div class="layout-shell">
    <aside class="layout-sidebar">
      <div class="brand">
        <span class="logo-dot" />
        <span>ZhoMind</span>
      </div>
      <nav class="nav-list">
        <RouterLink
          v-for="item in navItems"
          :key="item.path"
          :to="item.path"
          class="nav-link"
          :class="{ active: currentPath.startsWith(item.path) }"
        >
          <component :is="item.icon" class="nav-icon" />
          <span>{{ item.label }}</span>
        </RouterLink>
      </nav>
    </aside>
    <main class="layout-main">
      <div class="grid-12">
        <section class="col-span-8 col-start-3 main-slot">
          <router-view />
        </section>
      </div>
    </main>
  </div>
</template>

<script setup>
import { computed } from 'vue';
import { useRoute } from 'vue-router';
import { MessageCircle, FileText, SlidersHorizontal } from 'lucide-vue-next';

const route = useRoute();
const currentPath = computed(() => route.path);
const navItems = [
  { path: '/chat', label: '聊天', icon: MessageCircle },
  { path: '/documents', label: '文档', icon: FileText },
  { path: '/config', label: '配置', icon: SlidersHorizontal }
];
</script>

<style scoped>
.brand {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 18px;
  font-weight: 600;
}

.logo-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: var(--cta);
  display: inline-block;
}

.nav-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.nav-link {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 10px;
  color: var(--text-muted);
}

.nav-link.active {
  background: #e2e8f0;
  color: var(--text-strong);
}

.nav-icon {
  width: 18px;
  height: 18px;
}

.main-slot {
  padding-bottom: 80px;
}
</style>
