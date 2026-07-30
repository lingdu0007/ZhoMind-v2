import { createRouter, createWebHistory } from 'vue-router';

import AuthEntryPage from '../pages/AuthEntryPage.vue';
import ChatPage from '../pages/ChatPage.vue';
import UploadPage from '../pages/UploadPage.vue';
import ConfigPage from '../pages/ConfigPage.vue';
import { clearProtectedSession } from './protected-session';
import { useAuthStore } from '../store/auth';

const routes = [
  { path: '/', redirect: '/chat' },
  { path: '/auth', name: 'authentication-entry', component: AuthEntryPage, meta: { public: true } },
  { path: '/chat', name: 'chat', component: ChatPage, meta: { requiresAuth: true } },
  { path: '/documents', name: 'documents', component: UploadPage, meta: { requiresAuth: true, requiresAdmin: true } },
  { path: '/config', name: 'config', component: ConfigPage, meta: { requiresAuth: true, requiresAdmin: true, unavailable: true } }
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

router.beforeEach(async (to) => {
  const authStore = useAuthStore();

  if (authStore.isResolving) {
    try {
      await authStore.refreshMe();
    } catch {
      clearProtectedSession();
    }
  }

  if (to.meta.public) {
    return authStore.isLoggedIn ? { name: 'chat' } : true;
  }

  if (!to.meta.requiresAuth) return true;

  if (!authStore.isLoggedIn) {
    clearProtectedSession();
    return { name: 'authentication-entry' };
  }

  if (to.meta.requiresAdmin && !authStore.isAdmin) {
    return { name: 'chat', query: { notice: 'admin-required' } };
  }

  if (to.meta.unavailable) {
    return { name: 'chat' };
  }

  return true;
});

export default router;
