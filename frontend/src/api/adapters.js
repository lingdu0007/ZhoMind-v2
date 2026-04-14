import http, { resolveApiBaseURL } from './http';
import { createSSEParser, normalizeSSEFrame } from './sse';

export const apiAdapter = {
  // Auth
  async register(payload) {
    const { data } = await http.post('/auth/register', payload);
    return data;
  },
  async login(payload) {
    const { data } = await http.post('/auth/login', payload);
    return data;
  },
  async getCurrentUser() {
    const { data } = await http.get('/auth/me');
    return data;
  },

  // Chat & sessions
  async chat(payload) {
    const { data } = await http.post('/chat', payload);
    return data;
  },
  async listSessions() {
    const { data } = await http.get('/sessions');
    return data;
  },
  async getSessionMessages(sessionId) {
    const { data } = await http.get(`/sessions/${encodeURIComponent(sessionId)}`);
    return data;
  },
  async deleteSession(sessionId) {
    const { data } = await http.delete(`/sessions/${encodeURIComponent(sessionId)}`);
    return data;
  },

  // Documents (admin)
  async listDocuments(params) {
    const { data } = await http.get('/documents', { params });
    return data;
  },
  async uploadDocument(formData) {
    const { data } = await http.post('/documents/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return data;
  },
  async buildDocument(documentId, payload) {
    const { data } = await http.post(`/documents/${encodeURIComponent(documentId)}/build`, payload);
    return data;
  },
  async batchBuildDocuments(payload) {
    const { data } = await http.post('/documents/batch-build', payload);
    return data;
  },
  async batchDeleteDocuments(payload) {
    const { data } = await http.post('/documents/batch-delete', payload);
    return data;
  },
  async getDocumentChunks(documentId, params) {
    const { data } = await http.get(`/documents/${encodeURIComponent(documentId)}/chunks`, { params });
    return data;
  },
  async deleteDocument(filename) {
    const { data } = await http.delete(`/documents/${encodeURIComponent(filename)}`);
    return data;
  },

  // Document async jobs (admin)
  async listDocumentJobs(params) {
    const { data } = await http.get('/documents/jobs', { params });
    return data;
  },
  async getDocumentJob(jobId) {
    const { data } = await http.get(`/documents/jobs/${encodeURIComponent(jobId)}`);
    return data;
  },
  async cancelDocumentJob(jobId) {
    const { data } = await http.post(`/documents/jobs/${encodeURIComponent(jobId)}/cancel`);
    return data;
  }
};

export const streamChat = async ({ message, session_id, signal }, handlers = {}) => {
  const token = localStorage.getItem('access_token');
  const base = resolveApiBaseURL();
  const response = await fetch(`${base}/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {})
    },
    body: JSON.stringify({ message, session_id }),
    signal
  });

  if (!response.ok || !response.body) {
    throw new Error(`流式请求失败: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let doneDispatched = false;

  const dispatch = (event) => {
    if (!event) return;

    if (event.type === 'done') {
      doneDispatched = true;
      handlers.onDone?.();
      return;
    }

    if (event.type === 'content') {
      handlers.onContent?.(event.content || event.delta || '');
      return;
    }

    if (event.type === 'rag_step') {
      handlers.onRagStep?.(event.step ?? event.data ?? event);
      return;
    }

    if (event.type === 'trace') {
      handlers.onTrace?.(event.trace ?? event.data ?? event);
      return;
    }

    if (event.type === 'error') {
      handlers.onError?.(event.error || event.detail || '流式响应错误');
      return;
    }

    handlers.onUnknown?.(event);
  };

  const parser = createSSEParser((frame) => {
    const event = normalizeSSEFrame(frame);
    dispatch(event);
  });

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    parser.feed(decoder.decode(value, { stream: true }));
    if (doneDispatched) return;
  }

  parser.feed(decoder.decode());
  parser.finish();
  if (!doneDispatched) {
    handlers.onDone?.();
  }
};
