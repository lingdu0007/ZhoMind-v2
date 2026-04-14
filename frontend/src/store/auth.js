import { defineStore } from 'pinia';
import { apiAdapter } from '../api/adapters';

export const useAuthStore = defineStore('auth', {
  state: () => ({
    username: localStorage.getItem('username') || '',
    role: localStorage.getItem('role') || '',
    token: localStorage.getItem('access_token') || '',
    loading: false
  }),
  getters: {
    isLoggedIn: (state) => Boolean(state.token),
    isAdmin: (state) => state.role === 'admin'
  },
  actions: {
    setAuth(authResp) {
      const token = authResp?.access_token || authResp?.token || '';
      const username = authResp?.username || '';
      const role = authResp?.role || 'user';

      this.token = token;
      this.username = username;
      this.role = role;

      localStorage.setItem('access_token', token);
      localStorage.setItem('username', username);
      localStorage.setItem('role', role);
    },
    clearAuth() {
      this.token = '';
      this.username = '';
      this.role = '';
      localStorage.removeItem('access_token');
      localStorage.removeItem('username');
      localStorage.removeItem('role');
    },
    async login(payload) {
      this.loading = true;
      try {
        const data = await apiAdapter.login(payload);
        this.setAuth(data);
      } finally {
        this.loading = false;
      }
    },
    async register(payload) {
      this.loading = true;
      try {
        const data = await apiAdapter.register(payload);
        this.setAuth(data);
      } finally {
        this.loading = false;
      }
    },
    async refreshMe() {
      if (!this.token) return;
      const data = await apiAdapter.getCurrentUser();
      this.username = data?.username || this.username;
      this.role = data?.role || this.role;
      localStorage.setItem('username', this.username);
      localStorage.setItem('role', this.role);
    }
  }
});
