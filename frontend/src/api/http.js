import axios from 'axios';

const DIRECT_BACKEND_BASE_URL = 'http://127.0.0.1:8000/api/v1';

const trimTrailingSlash = (value) => value.replace(/\/+$/, '');

export const resolveApiBaseURL = () => {
  const envBase = import.meta.env.VITE_API_BASE_URL;
  if (envBase && envBase.trim()) {
    return trimTrailingSlash(envBase.trim());
  }

  // Nginx 同源语义：页面由 8001 提供时统一走 /api，再由 Nginx 转 /api/v1。
  if (window.location.port === '8001') {
    return '/api';
  }

  // 直连后端语义：显式带 /api/v1。
  return DIRECT_BACKEND_BASE_URL;
};

const baseURL = resolveApiBaseURL();

const http = axios.create({
  baseURL,
  timeout: 30000
});

http.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

http.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error?.response?.status;
    const payload = error?.response?.data || {};
    if (status === 401) {
      localStorage.removeItem('access_token');
      localStorage.removeItem('username');
      localStorage.removeItem('role');
    }

    const normalizedError = new Error(payload.message || payload.detail || error.message || '请求失败');
    normalizedError.status = status || 0;
    normalizedError.code = payload.code || '';
    normalizedError.detail = payload.detail;
    normalizedError.request_id = payload.request_id || '';
    return Promise.reject(normalizedError);
  }
);

export default http;
