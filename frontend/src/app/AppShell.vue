<template>
  <div class="workbench-shell">
    <aside class="workbench-shell__rail">
      <div class="workbench-shell__brand" aria-label="ZhoMind">
        <span aria-hidden="true" class="workbench-shell__mark" />
        <span class="workbench-shell__brand-name">ZhoMind</span>
      </div>

      <nav class="workbench-shell__nav" aria-label="主导航">
        <RouterLink class="workbench-shell__nav-item" to="/chat" title="对话工作区" aria-label="对话工作区">
          <MessageCircle :size="20" aria-hidden="true" />
          <span>对话工作区</span>
        </RouterLink>
        <RouterLink class="workbench-shell__nav-item" to="/knowledge" title="知识地图" aria-label="知识地图">
          <BookOpen :size="20" aria-hidden="true" />
          <span>知识地图</span>
        </RouterLink>
        <RouterLink
          v-if="authStore.isAdmin"
          class="workbench-shell__nav-item"
          to="/reviews"
          title="知识复核"
          aria-label="知识复核"
        >
          <ListChecks :size="20" aria-hidden="true" />
          <span>知识复核</span>
        </RouterLink>
        <RouterLink
          v-if="authStore.isAdmin"
          class="workbench-shell__nav-item"
          to="/documents"
          title="文档库"
          aria-label="文档库"
        >
          <LibraryBig :size="20" aria-hidden="true" />
          <span>文档库</span>
        </RouterLink>
        <RouterLink
          v-if="authStore.isAdmin"
          class="workbench-shell__nav-item"
          to="/jobs"
          title="构建任务"
          aria-label="构建任务"
        >
          <ClipboardList :size="20" aria-hidden="true" />
          <span>构建任务</span>
        </RouterLink>
        <RouterLink
          v-if="authStore.isAdmin"
          class="workbench-shell__nav-item workbench-shell__nav-item--reviewed-bundles"
          to="/reviewed-bundles"
          title="Reviewed Release Bundles"
          aria-label="Reviewed Release Bundles"
        >
          <PackageCheck :size="20" aria-hidden="true" />
          <span>Reviewed Release Bundles</span>
        </RouterLink>
        <RouterLink
          v-if="authStore.canAccessSystemSettings"
          class="workbench-shell__nav-item"
          to="/config"
          title="系统设置"
          aria-label="系统设置"
        >
          <Settings2 :size="20" aria-hidden="true" />
          <span>系统设置</span>
        </RouterLink>
      </nav>

      <div class="workbench-shell__identity">
        <span class="workbench-shell__identity-name">{{ authStore.username }}</span>
        <span class="workbench-shell__identity-role">{{ roleLabel }}</span>
        <button type="button" title="退出登录" aria-label="退出登录" @click="signOut">
          <LogOut :size="18" aria-hidden="true" />
        </button>
      </div>
    </aside>

    <main class="workbench-shell__main">
      <p v-if="accessMessage" class="workbench-shell__notice" role="status">{{ accessMessage }}</p>
      <router-view />
    </main>
  </div>
</template>

<script setup>
import { computed } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { BookOpen, ClipboardList, LibraryBig, ListChecks, LogOut, MessageCircle, PackageCheck, Settings2 } from 'lucide-vue-next';
import { clearProtectedSession } from './protected-session';
import { useAuthStore } from '../store/auth';

const authStore = useAuthStore();
const route = useRoute();
const router = useRouter();

const roleLabel = computed(() => (authStore.isAdmin ? '系统管理员' : '知识用户'));
const accessMessage = computed(() => {
  if (route.query.notice === 'admin-required') return '当前账户无权访问该工作区，已返回对话工作区。';
  if (route.query.notice === 'settings-unavailable') return '系统设置当前不可用，已返回对话工作区。';
  return '';
});

const signOut = async () => {
  try {
    await authStore.logout();
  } catch {
    // A failed revocation must never leave an authorized workbench visible.
  } finally {
    clearProtectedSession();
    await router.replace({ name: 'authentication-entry' });
  }
};
</script>

<style scoped>
.workbench-shell {
  min-height: 100vh;
  display: grid;
  grid-template-columns: 84px minmax(0, 1fr);
  background: var(--color-paper);
}

.workbench-shell__rail {
  position: sticky;
  top: 0;
  height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 28px;
  padding: 22px 12px 16px;
  border-right: 1px solid var(--color-rule);
  background: var(--color-paper-muted);
}

.workbench-shell__brand {
  display: grid;
  place-items: center;
  width: 40px;
  height: 40px;
}

.workbench-shell__mark {
  width: 22px;
  height: 22px;
  border: 1px solid var(--color-copper);
  box-shadow: -4px 4px 0 var(--color-paper-muted), -4px 4px 0 1px var(--color-copper), -8px 8px 0 var(--color-paper-muted), -8px 8px 0 1px var(--color-copper);
}

.workbench-shell__brand-name {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

.workbench-shell__nav {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.workbench-shell__nav-item,
.workbench-shell__identity button {
  width: 40px;
  height: 40px;
  display: grid;
  place-items: center;
  border: 1px solid transparent;
  border-radius: var(--radius-control);
  color: var(--color-ink-soft);
}

.workbench-shell__nav-item.router-link-active {
  border-color: var(--color-rule);
  background: var(--color-paper-raised);
  color: var(--color-copper-strong);
}

.workbench-shell__nav-item span {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

.workbench-shell__identity {
  margin-top: auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 7px;
}

.workbench-shell__identity-name,
.workbench-shell__identity-role {
  max-width: 56px;
  overflow: hidden;
  color: var(--color-ink-soft);
  font-size: 11px;
  line-height: 1.2;
  text-align: center;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.workbench-shell__identity-role {
  color: var(--color-moss);
}

.workbench-shell__identity button {
  border: 0;
  background: transparent;
  cursor: pointer;
}

.workbench-shell__main {
  min-width: 0;
  padding: 36px clamp(24px, 5vw, 72px) 56px;
}

.workbench-shell__notice {
  max-width: 960px;
  margin: 0 auto 20px;
  padding: 10px 12px;
  border-left: 3px solid var(--color-warning);
  background: var(--color-warning-soft);
  color: var(--color-warning);
  font-size: 13px;
}

@media (max-width: 640px) {
  .workbench-shell {
    grid-template-columns: 1fr;
    grid-template-rows: auto minmax(0, 1fr);
  }

  .workbench-shell__rail {
    position: static;
    height: auto;
    flex-direction: row;
    justify-content: space-between;
    gap: 8px;
    padding: 10px 8px;
    border-right: 0;
    border-bottom: 1px solid var(--color-rule);
  }

  .workbench-shell__nav {
    margin-left: auto;
    flex-direction: row;
    gap: 4px;
  }

  .workbench-shell__nav-item--reviewed-bundles {
    display: none;
  }

  .workbench-shell__identity {
    margin: 0 0 0 4px;
    flex-direction: row;
  }

  .workbench-shell__identity-name,
  .workbench-shell__identity-role {
    display: none;
  }

  .workbench-shell__main {
    padding: 24px 16px 32px;
  }
}
</style>
