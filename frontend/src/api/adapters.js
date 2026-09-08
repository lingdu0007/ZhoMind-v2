import http, { notifyAuthInvalid, resolveApiBaseURL } from './http';
import { createSSEParser, normalizeSSEFrame } from './sse';

const unwrapData = (payload) => payload?.data ?? payload;
const abortStreamError = () => {
  const error = new Error('流式请求已取消。');
  error.name = 'AbortError';
  return error;
};
const awaitWithAbort = (operation, signal) => {
  if (!signal) return operation;
  if (signal.aborted) return Promise.reject(abortStreamError());

  return new Promise((resolve, reject) => {
    const rejectAbort = () => reject(abortStreamError());
    signal.addEventListener('abort', rejectAbort, { once: true });
    operation.then(
      (result) => {
        signal.removeEventListener('abort', rejectAbort);
        resolve(result);
      },
      (error) => {
        signal.removeEventListener('abort', rejectAbort);
        reject(error);
      }
    );
  });
};

export const apiAdapter = {
  // Auth
  async register(payload) {
    const { data } = await http.post('/auth/register', payload);
    return unwrapData(data);
  },
  async login(payload) {
    const { data } = await http.post('/auth/login', payload);
    return unwrapData(data);
  },
  async logout() {
    const { data } = await http.post('/auth/logout');
    return unwrapData(data);
  },
  async getCurrentUser() {
    const { data } = await http.get('/auth/me');
    return unwrapData(data);
  },

  // Chat & sessions
  async chat(payload) {
    const { data } = await http.post('/chat', payload);
    return unwrapData(data);
  },
  async listSessions() {
    const { data } = await http.get('/sessions');
    return unwrapData(data);
  },
  async getSessionMessages(sessionId) {
    const { data } = await http.get(`/sessions/${encodeURIComponent(sessionId)}`);
    return unwrapData(data);
  },
  async deleteSession(sessionId) {
    const { data } = await http.delete(`/sessions/${encodeURIComponent(sessionId)}`);
    return unwrapData(data);
  },
  async getKnowledgeMap() {
    const { data } = await http.get('/knowledge-map');
    return unwrapData(data);
  },
  async getKnowledgeMapEntry(entryId) {
    const { data } = await http.get(`/knowledge-map/${encodeURIComponent(entryId)}`);
    return unwrapData(data);
  },
  async submitKnowledgeFeedback(payload) {
    const { data } = await http.post('/knowledge-feedback', payload);
    return unwrapData(data);
  },
  async listKnowledgeFeedback(answerId = undefined) {
    const params =
      typeof answerId === 'string' && answerId.trim()
        ? { answer_id: answerId }
        : undefined;
    const { data } = await http.get('/knowledge-feedback', {
      params
    });
    return unwrapData(data);
  },
  async deleteKnowledgeFeedback(signalId) {
    const { data } = await http.delete(`/knowledge-feedback/${encodeURIComponent(signalId)}`);
    return unwrapData(data);
  },

  // Knowledge review queue (admin)
  async listKnowledgeReviewQueue() {
    const { data } = await http.get('/knowledge-review-queue');
    return unwrapData(data);
  },
  async classifyKnowledgeReviewItem(itemId, payload) {
    const { data } = await http.patch(`/knowledge-review-queue/${encodeURIComponent(itemId)}`, payload);
    return unwrapData(data);
  },

  // Documents (admin)
  async listDocuments(params) {
    const { data } = await http.get('/documents', { params });
    return unwrapData(data);
  },
  async uploadDocument(formData) {
    const { data } = await http.post('/documents/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return unwrapData(data);
  },
  async buildDocument(documentId, payload) {
    const { data } = await http.post(`/documents/${encodeURIComponent(documentId)}/build`, payload);
    return unwrapData(data);
  },
  async batchBuildDocuments(payload) {
    const { data } = await http.post('/documents/batch-build', payload);
    return unwrapData(data);
  },
  async batchDeleteDocuments(payload) {
    const { data } = await http.post('/documents/batch-delete', payload);
    return unwrapData(data);
  },
  async getDocumentChunks(documentId, params) {
    const { data } = await http.get(`/documents/${encodeURIComponent(documentId)}/chunks`, { params });
    return unwrapData(data);
  },
  async deleteDocument(filename) {
    const { data } = await http.delete(`/documents/${encodeURIComponent(filename)}`);
    return unwrapData(data);
  },

  // Document async jobs (admin)
  async listDocumentJobs(params) {
    const { data } = await http.get('/documents/jobs', { params });
    return unwrapData(data);
  },
  async getDocumentJob(jobId) {
    const { data } = await http.get(`/documents/jobs/${encodeURIComponent(jobId)}`);
    return unwrapData(data);
  },
  async cancelDocumentJob(jobId) {
    const { data } = await http.post(`/documents/jobs/${encodeURIComponent(jobId)}/cancel`);
    return unwrapData(data);
  },

  // Reviewed Release Bundles and isolated Candidate Builds (admin)
  async listReviewedReleaseBundles() {
    const { data } = await http.get('/reviewed-release-bundles');
    return unwrapData(data);
  },
  async importReviewedReleaseBundle(manifest) {
    const { data } = await http.post('/reviewed-release-bundles/import', manifest);
    return unwrapData(data);
  },
  async getReviewedReleaseBundle(bundleId) {
    const { data } = await http.get(`/reviewed-release-bundles/${encodeURIComponent(bundleId)}`);
    return unwrapData(data);
  },
  async getReviewedBundleJob(jobId) {
    const { data } = await http.get(`/reviewed-release-bundles/jobs/${encodeURIComponent(jobId)}`);
    return unwrapData(data);
  },
  async dispatchReviewedBundleJob(jobId) {
    const { data } = await http.post(`/reviewed-release-bundles/jobs/${encodeURIComponent(jobId)}/dispatch`);
    return unwrapData(data);
  },
  async retryReviewedBundleJob(jobId) {
    const { data } = await http.post(`/reviewed-release-bundles/jobs/${encodeURIComponent(jobId)}/retry`);
    return unwrapData(data);
  },
  async cancelReviewedBundleJob(jobId) {
    const { data } = await http.post(`/reviewed-release-bundles/jobs/${encodeURIComponent(jobId)}/cancel`);
    return unwrapData(data);
  },
  async getReviewedCandidateInspection(candidateId) {
    const { data } = await http.get(
      `/reviewed-release-bundles/candidates/${encodeURIComponent(candidateId)}/inspection`
    );
    return unwrapData(data);
  },
  async inspectReviewedCandidate(candidateId) {
    const { data } = await http.post(
      `/reviewed-release-bundles/candidates/${encodeURIComponent(candidateId)}/inspection`
    );
    return unwrapData(data);
  },
  async acceptReviewedCandidate(candidateId) {
    const { data } = await http.post(
      `/reviewed-release-bundles/candidates/${encodeURIComponent(candidateId)}/acceptance`
    );
    return unwrapData(data);
  },
  async getReviewedCandidatePublicationEligibility(candidateId) {
    const { data } = await http.get(
      `/reviewed-release-bundles/candidates/${encodeURIComponent(candidateId)}/publication-eligibility`
    );
    return unwrapData(data);
  },
  async getReviewedCandidatePublication(candidateId) {
    const { data } = await http.get(
      `/reviewed-release-bundles/candidates/${encodeURIComponent(candidateId)}/publication`
    );
    return unwrapData(data);
  },
  async publishReviewedCandidateBatch(payload) {
    const { data } = await http.post('/reviewed-release-bundles/publication-batches', payload);
    return unwrapData(data);
  },

  // System Settings (admin, application-gated)
  async getGenerationRoute() {
    const { data } = await http.get('/settings/generation-route');
    return unwrapData(data);
  },
  async saveGenerationRoute(payload) {
    const { data } = await http.put('/settings/generation-route', payload);
    return unwrapData(data);
  },
  async activateGenerationRoute(payload) {
    const { data } = await http.post('/settings/generation-route/activate', payload);
    return unwrapData(data);
  },
  async getSystemSettingsDraft() {
    const { data } = await http.get('/settings/draft');
    return unwrapData(data);
  },
  async saveSystemSettingsDraft(payload) {
    const { data } = await http.put('/settings/draft', payload);
    return unwrapData(data);
  },
  async applySystemSettingsVersion(version) {
    const { data } = await http.post('/settings/apply', { version });
    return unwrapData(data);
  }
};

export const streamChat = async (
  { message, session_id, query_conditions, inherit_conditions, signal, token },
  handlers = {}
) => {
  const authToken = token || localStorage.getItem('access_token');
  const base = resolveApiBaseURL();
  const payload = { message, session_id };
  if (query_conditions !== undefined) payload.query_conditions = query_conditions;
  if (inherit_conditions === true) payload.inherit_conditions = true;
  const response = await awaitWithAbort(
    fetch(`${base}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(authToken ? { Authorization: `Bearer ${authToken}` } : {})
      },
      body: JSON.stringify(payload),
      signal
    }),
    signal
  );

  if (!response.ok || !response.body) {
    if (response.status === 401) notifyAuthInvalid();
    let messageText = `流式请求失败: ${response.status}`;
    let code = '';
    let requestId = '';

    try {
      const payload = await response.clone().json();
      const envelopeData = payload?.data || {};
      messageText =
        payload?.message ||
        payload?.detail ||
        envelopeData?.message ||
        envelopeData?.detail ||
        messageText;
      code = payload?.code || envelopeData?.code || '';
      requestId = payload?.request_id || envelopeData?.request_id || '';
    } catch {
      // ignore json parse failure and keep fallback message
    }

    const error = new Error(messageText);
    error.status = response.status;
    error.code = code;
    error.request_id = requestId;
    throw error;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let doneDispatched = false;
  const cancelReader = () => {
    void reader.cancel().catch(() => {
      // A network cancellation may already have closed the reader.
    });
  };
  const readWithAbort = () => {
    return awaitWithAbort(reader.read(), signal);
  };
  signal?.addEventListener('abort', cancelReader, { once: true });

  const dispatch = (event) => {
    if (!event) return;

    if (event.type === 'protocol_error') {
      throw new Error(event.error || 'stream emitted a malformed terminal done event');
    }

    if (doneDispatched && event.type !== 'unknown') {
      throw new Error(
        event.type === 'done'
          ? 'stream emitted a repeated terminal done event'
          : 'stream emitted a semantic event after terminal done'
      );
    }

    if (event.type === 'done') {
      handlers.onDone?.();
      doneDispatched = true;
      return;
    }

    if (event.type === 'content') {
      handlers.onContent?.(event.content || event.delta || '');
      return;
    }

    if (event.type === 'answer_identity') {
      handlers.onAnswerIdentity?.(event.answer_id || '');
      return;
    }

    if (event.type === 'answer_execution') {
      handlers.onAnswerExecution?.(event.answer_execution);
      return;
    }

    if (event.type === 'outcome') {
      handlers.onOutcome?.(event.outcome || '');
      return;
    }

    if (event.type === 'insufficient_evidence_reply') {
      handlers.onInsufficientEvidenceReply?.(event.insufficient_evidence_reply);
      return;
    }

    if (event.type === 'stage') {
      handlers.onStage?.({ stage: event.stage || '', message: event.message || '' });
      return;
    }

    if (event.type === 'evidence_summary') {
      handlers.onEvidenceSummary?.(event.evidence_summary);
      return;
    }

    if (event.type === 'retrieval_diagnostics') {
      handlers.onRetrievalDiagnostics?.(event.retrieval_diagnostics);
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

  try {
    if (signal?.aborted) {
      cancelReader();
      throw abortStreamError();
    }
    while (true) {
      const { done, value } = await readWithAbort();
      if (signal?.aborted) throw abortStreamError();
      if (done) break;

      parser.feed(decoder.decode(value, { stream: true }));
    }

    if (signal?.aborted) throw abortStreamError();
    parser.feed(decoder.decode());
    parser.finish();
    if (!doneDispatched) {
      const error = new Error('流式响应在完成前中断。');
      error.code = 'CHAT_STREAM_INTERRUPTED';
      handlers.onError?.(error);
      throw error;
    }
  } finally {
    signal?.removeEventListener('abort', cancelReader);
  }
};
