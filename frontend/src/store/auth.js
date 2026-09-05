import { defineStore } from 'pinia';
import { apiAdapter } from '../api/adapters';

export const useAuthStore = defineStore('auth', {
  state: () => ({
    username: '',
    role: '',
    capabilities: { system_settings: false },
    token: localStorage.getItem('access_token') || '',
    loading: false,
    status: localStorage.getItem('access_token') ? 'resolving' : 'anonymous'
  }),
  getters: {
    isLoggedIn: (state) => state.status === 'authenticated' && Boolean(state.token),
    isAdmin: (state) => state.status === 'authenticated' && state.role === 'admin',
    canAccessSystemSettings: (state) =>
      state.status === 'authenticated' && state.role === 'admin' && state.capabilities.system_settings,
    isResolving: (state) => state.status === 'resolving'
  },
  actions: {
    setToken(authResp) {
      const token = authResp?.access_token || authResp?.token || '';
      if (!token) throw new Error('认证响应未返回访问令牌');

      this.token = token;
      this.username = '';
      this.role = '';
      this.capabilities = { system_settings: false };
      this.status = 'resolving';

      localStorage.setItem('access_token', token);
      localStorage.removeItem('username');
      localStorage.removeItem('role');
    },
    setIdentity(identity) {
      const username = identity?.username || '';
      const role = identity?.role || '';
      if (!username || !['user', 'admin'].includes(role)) {
        throw new Error('认证身份信息无效');
      }

      this.username = username;
      this.role = role;
      this.capabilities = { system_settings: identity?.capabilities?.system_settings === true };
      this.status = 'authenticated';

      localStorage.setItem('username', username);
      localStorage.setItem('role', role);
    },
    clearAuth() {
      this.token = '';
      this.username = '';
      this.role = '';
      this.capabilities = { system_settings: false };
      this.status = 'anonymous';
      localStorage.removeItem('access_token');
      localStorage.removeItem('username');
      localStorage.removeItem('role');
    },
    async login(payload) {
      return this.authenticate(apiAdapter.login, payload);
    },
    async register(payload) {
      return this.authenticate(apiAdapter.register, payload);
    },
    async logout() {
      await apiAdapter.logout();
    },
    async authenticate(request, payload) {
      this.loading = true;
      try {
        const data = await request(payload);
        this.setToken(data);
        await this.refreshMe();
      } finally {
        this.loading = false;
      }
    },
    async refreshMe() {
      if (!this.token) {
        this.status = 'anonymous';
        return false;
      }

      this.status = 'resolving';
      try {
        const data = await apiAdapter.getCurrentUser();
        this.setIdentity(data);
        return true;
      } catch (error) {
        this.clearAuth();
        throw error;
      }
    }
  }
});
