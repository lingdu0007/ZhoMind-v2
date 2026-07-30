<template>
  <main class="auth-entry">
    <section class="auth-entry__intro" aria-labelledby="auth-title">
      <div class="auth-entry__mark" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>
      <p class="auth-entry__eyebrow">ZhoMind</p>
      <h1 id="auth-title">身份验证</h1>
      <p class="auth-entry__description">进入团队知识库，检索已有资料并保留可核查的阅读路径。</p>
    </section>

    <section class="auth-entry__form" aria-label="账户访问">
      <div class="auth-entry__switch" role="tablist" aria-label="认证模式">
        <button
          id="sign-in-tab"
          class="auth-entry__mode"
          :class="{ 'auth-entry__mode--active': mode === 'login' }"
          type="button"
          role="tab"
          :aria-selected="mode === 'login'"
          @click="mode = 'login'"
        >
          登录
        </button>
        <button
          id="registration-tab"
          class="auth-entry__mode"
          :class="{ 'auth-entry__mode--active': mode === 'register' }"
          type="button"
          role="tab"
          :aria-selected="mode === 'register'"
          @click="mode = 'register'"
        >
          注册
        </button>
      </div>

      <form class="auth-entry__fields" @submit.prevent="submit">
        <div class="auth-entry__field">
          <label for="auth-username">用户名</label>
          <input id="auth-username" v-model.trim="form.username" autocomplete="username" required />
        </div>
        <div class="auth-entry__field">
          <label for="auth-password">密码</label>
          <input id="auth-password" v-model="form.password" type="password" autocomplete="current-password" required />
        </div>

        <fieldset v-if="mode === 'register'" class="auth-entry__role-picker">
          <legend>注册身份</legend>
          <label>
            <input v-model="form.role" type="radio" value="user" />
            <span>知识用户</span>
          </label>
          <label>
            <input v-model="form.role" type="radio" value="admin" />
            <span>系统管理员</span>
          </label>
        </fieldset>

        <div v-if="mode === 'register' && form.role === 'admin'" class="auth-entry__field">
          <label for="auth-admin-code">管理员邀请码</label>
          <input id="auth-admin-code" v-model.trim="form.adminCode" autocomplete="off" required />
        </div>

        <p v-if="errorMessage" class="auth-entry__error" role="alert">{{ errorMessage }}</p>
        <button class="auth-entry__submit" type="submit" :disabled="authStore.loading">
          {{ authStore.loading ? '正在验证…' : mode === 'login' ? '登录' : '完成注册' }}
        </button>
      </form>
    </section>
  </main>
</template>

<script setup>
import { reactive, ref } from 'vue';
import { useRouter } from 'vue-router';
import { useAuthStore } from '../store/auth';

const router = useRouter();
const authStore = useAuthStore();
const mode = ref('login');
const errorMessage = ref('');
const form = reactive({
  username: '',
  password: '',
  role: 'user',
  adminCode: ''
});

const formatAuthError = (error) => {
  if (error?.code === 'AUTH_INVALID_CREDENTIALS') return '用户名或密码不正确，请核对后重试。';
  if (error?.code === 'AUTH_FORBIDDEN') return '管理员邀请码无效，请核对后重试。';
  if (error?.code === 'VALIDATION_ERROR') return '请填写用户名和密码。';
  return error?.message || '认证未完成，请稍后重试。';
};

const submit = async () => {
  errorMessage.value = '';
  try {
    if (mode.value === 'login') {
      await authStore.login({ username: form.username, password: form.password });
    } else {
      const payload = {
        username: form.username,
        password: form.password,
        role: form.role
      };
      if (form.role === 'admin') payload.admin_code = form.adminCode;
      await authStore.register(payload);
    }
    await router.replace({ name: 'chat' });
  } catch (error) {
    errorMessage.value = formatAuthError(error);
  }
};
</script>

<style scoped>
.auth-entry {
  min-height: 100vh;
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(360px, 440px);
  background: var(--color-paper);
}

.auth-entry__intro {
  padding: clamp(48px, 12vh, 136px) clamp(32px, 10vw, 160px);
  border-right: 1px solid var(--color-rule);
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.auth-entry__mark {
  width: 36px;
  height: 32px;
  position: relative;
  margin-bottom: 28px;
}

.auth-entry__mark span {
  position: absolute;
  width: 22px;
  height: 22px;
  border: 1px solid var(--color-copper);
  background: var(--color-paper-raised);
}

.auth-entry__mark span:nth-child(1) { left: 0; top: 8px; }
.auth-entry__mark span:nth-child(2) { left: 7px; top: 4px; }
.auth-entry__mark span:nth-child(3) { left: 14px; top: 0; }

.auth-entry__eyebrow {
  margin: 0 0 12px;
  color: var(--color-ink-soft);
  font-size: 13px;
  font-weight: 600;
}

h1 {
  margin: 0;
  font-family: var(--font-display);
  font-size: 34px;
  font-weight: 600;
  line-height: 1.3;
}

.auth-entry__description {
  max-width: 420px;
  margin: 16px 0 0;
  color: var(--color-ink-soft);
  line-height: 1.8;
}

.auth-entry__form {
  align-self: center;
  width: min(100%, 440px);
  padding: 48px;
}

.auth-entry__switch {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  border-bottom: 1px solid var(--color-rule);
  margin-bottom: 28px;
}

.auth-entry__mode {
  padding: 10px 0;
  border: 0;
  border-bottom: 2px solid transparent;
  background: transparent;
  color: var(--color-ink-soft);
  font: inherit;
  cursor: pointer;
}

.auth-entry__mode--active {
  border-bottom-color: var(--color-copper);
  color: var(--color-ink);
  font-weight: 600;
}

.auth-entry__fields {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.auth-entry__field,
.auth-entry__role-picker {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin: 0;
  padding: 0;
  border: 0;
}

.auth-entry__field label,
.auth-entry__role-picker legend {
  color: var(--color-ink-soft);
  font-size: 13px;
  font-weight: 600;
}

.auth-entry__field input {
  min-height: 40px;
  padding: 8px 10px;
  border: 1px solid var(--color-rule);
  border-radius: var(--radius-control);
  background: var(--color-paper-raised);
  color: var(--color-ink);
  font: inherit;
}

.auth-entry__role-picker {
  gap: 10px;
}

.auth-entry__role-picker label {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--color-ink);
  cursor: pointer;
}

.auth-entry__error {
  margin: -2px 0 0;
  color: var(--color-danger);
  font-size: 13px;
  line-height: 1.6;
}

.auth-entry__submit {
  min-height: 42px;
  border: 1px solid var(--color-copper);
  border-radius: var(--radius-control);
  background: var(--color-copper);
  color: var(--color-paper-raised);
  font: inherit;
  font-weight: 600;
  cursor: pointer;
}

.auth-entry__submit:disabled {
  cursor: wait;
  opacity: 0.65;
}

@media (max-width: 720px) {
  .auth-entry {
    grid-template-columns: 1fr;
  }

  .auth-entry__intro {
    min-height: 38vh;
    padding: 48px 24px 32px;
    border-right: 0;
    border-bottom: 1px solid var(--color-rule);
  }

  .auth-entry__form {
    padding: 32px 24px 48px;
  }
}
</style>
