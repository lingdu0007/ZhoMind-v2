import { defineStore } from 'pinia';
import { defaultConfig } from '../types';

const CONFIG_KEY = 'rag_config';

export const useConfigStore = defineStore('config', {
  state: () => ({
    config: { ...defaultConfig },
    loading: false
  }),
  actions: {
    async fetchConfig() {
      this.loading = true;
      try {
        const raw = localStorage.getItem(CONFIG_KEY);
        if (!raw) return;
        const parsed = JSON.parse(raw);
        this.config = { ...defaultConfig, ...parsed };
      } finally {
        this.loading = false;
      }
    },
    async saveConfig() {
      this.loading = true;
      try {
        localStorage.setItem(CONFIG_KEY, JSON.stringify(this.config));
      } finally {
        this.loading = false;
      }
    }
  }
});
