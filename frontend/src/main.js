import { createApp } from 'vue';
import { createPinia } from 'pinia';
import ElementPlus from 'element-plus';
import 'element-plus/dist/index.css';

import App from './app/App.vue';
import router from './app/router';
import { clearProtectedSession } from './app/protected-session';
import './styles/global.css';

const app = createApp(App);

app.use(createPinia());
app.use(router);
app.use(ElementPlus);

window.addEventListener('zhomind:auth-invalid', () => {
  clearProtectedSession();
  if (router.currentRoute.value.name !== 'authentication-entry') {
    router.replace({ name: 'authentication-entry' });
  }
});

app.mount('#app');
