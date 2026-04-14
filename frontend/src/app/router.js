import { createRouter, createWebHistory } from 'vue-router';

import ChatPage from '../pages/ChatPage.vue';
import UploadPage from '../pages/UploadPage.vue';
import ConfigPage from '../pages/ConfigPage.vue';

const routes = [
  { path: '/', redirect: '/chat' },
  { path: '/chat', name: 'chat', component: ChatPage },
  { path: '/documents', name: 'documents', component: UploadPage },
  { path: '/config', name: 'config', component: ConfigPage }
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

export default router;
