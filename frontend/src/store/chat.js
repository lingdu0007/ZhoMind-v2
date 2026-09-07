import { defineStore } from 'pinia';
import { apiAdapter, streamChat } from '../api/adapters';
import {
  formatStreamError,
  getDoneStatus,
  validateCompletedStreamProjection,
  validateExecutionTurnBinding
} from './chat-state';

const retainedExecutionStatus = (execution, outcome) => {
  const state = execution?.state || '';
  if (state === 'stopped') return '回答已停止，内容不完整';
  if (state === 'failed') return '回答失败，可重试';
  if (state === 'throttled') return '请求受限，请稍后重试';
  if (state === 'rejected') return '请求被拒绝';
  if (state === 'completed') return getDoneStatus({ outcome });
  return '';
};

const cloneQueryConditions = (conditions) => {
  if (!Array.isArray(conditions)) return undefined;
  return conditions.map((condition) => ({
    condition_id: condition?.condition_id || '',
    field: condition?.field || '',
    operator: condition?.operator || '',
    value: condition?.value || ''
  }));
};

const clearUntrustedCompletedProjection = (message) => {
  if (!message) return;
  if (message.answer_execution?.state === 'completed') {
    message.answer_execution = null;
  }
  message.outcome = '';
  message.rejected = false;
  message.evidence_summary = null;
  message.insufficient_evidence_reply = null;
  message.retrieval_diagnostics = null;
};

const receivedCompletedProjection = (terminalProjection) =>
  terminalProjection.execution?.state === 'completed' ||
  Boolean(
    terminalProjection.outcome ||
      terminalProjection.evidenceSummary ||
      terminalProjection.insufficientEvidenceReply
  );

export const useChatStore = defineStore('chat', {
  state: () => ({
    messages: [],
    loading: false,
    sessions: [],
    activeSessionId: '',
    streamController: null,
    streamTick: 0
  }),
  actions: {
    clearWorkspaceState() {
      this.streamController?.abort();
      this.messages = [];
      this.sessions = [];
      this.activeSessionId = '';
      this.streamController = null;
      this.loading = false;
      this.streamTick = 0;
    },
    async loadSessions() {
      const data = await apiAdapter.listSessions();
      this.sessions = data?.sessions || data?.items || data?.data || [];
    },
    async loadSessionMessages(sessionId) {
      if (!sessionId) return;
      const data = await apiAdapter.getSessionMessages(sessionId);
      this.activeSessionId = sessionId;
      const rawMessages = data?.messages || data?.items || data?.data || [];
      this.messages = rawMessages.map((item) => ({
        id: item?.id || '',
        role: item?.type === 'user' ? 'user' : 'assistant',
        content: item?.content || '',
        timestamp: item?.timestamp,
        answer_execution: item?.answer_execution || null,
        outcome: item?.outcome || '',
        evidence_summary: item?.evidence_summary || null,
        insufficient_evidence_reply: item?.insufficient_evidence_reply || null,
        retrieval_diagnostics: item?.retrieval_diagnostics || null,
        streaming: false,
        isThinking: false,
        rejected: item?.outcome === 'insufficient_evidence_reply',
        reject_reason: '',
        failed: item?.answer_execution?.state === 'failed',
        status: retainedExecutionStatus(item?.answer_execution, item?.outcome)
      }));
    },
    async deleteSession(sessionId) {
      const result = await apiAdapter.deleteSession(sessionId);
      if (result?.deleted !== true) {
        throw new Error('会话未删除，请刷新后重试。');
      }
      if (this.activeSessionId === sessionId) {
        this.activeSessionId = '';
        this.messages = [];
      }
      await this.loadSessions();
      return result;
    },
    stopStreaming() {
      if (this.streamController) {
        this.streamController.abort();
      }
    },
    async sendMessage(question, options = {}) {
      const normalizedQuestion = typeof question === 'string' ? question.trim() : '';
      if (!normalizedQuestion || this.loading) return;

      if (!this.activeSessionId || this.activeSessionId === 'default_session') {
        this.activeSessionId = `session_${Date.now()}`;
      }

      const priorSessionExecutions = this.messages
        .map((message) => message?.answer_execution)
        .filter((execution) => execution && typeof execution === 'object' && !Array.isArray(execution));
      const submittedTurn = {
        question: normalizedQuestion,
        query_conditions: cloneQueryConditions(options?.query_conditions),
        inherit_conditions: options?.inherit_conditions === true,
        known_executions: priorSessionExecutions
      };
      const userMessage = {
        role: 'user',
        content: normalizedQuestion,
        answer_execution: null,
        requested_query_conditions: cloneQueryConditions(options?.query_conditions),
        requested_inherit_conditions: options?.inherit_conditions === true,
        admission_observed: false,
        retry_authority_invalid: false
      };
      this.messages.push(userMessage);
      const localStreamId = `stream_${Date.now()}_${Math.random().toString(16).slice(2)}`;
      const assistantMessage = {
        id: '',
        local_stream_id: localStreamId,
        role: 'assistant',
        content: '',
        answer_execution: null,
        outcome: '',
        evidence_summary: null,
        insufficient_evidence_reply: null,
        retrieval_diagnostics: null,
        streaming: true,
        isThinking: true,
        rejected: false,
        reject_reason: '',
        failed: false,
        status: '思考中...'
      };
      this.messages.push(assistantMessage);

      const submittedSessionId = this.activeSessionId;
      const getAssistantMsg = () =>
        this.activeSessionId === submittedSessionId
          ? this.messages.find((message) => message?.local_stream_id === localStreamId) || null
          : null;
      const terminalProjection = {
        answerId: '',
        execution: null,
        outcome: '',
        evidenceSummary: null,
        insufficientEvidenceReply: null
      };
      const invalidateRetryAuthority = () => {
        userMessage.answer_execution = null;
        userMessage.retry_authority_invalid = true;
      };
      const requiresRecoveredRetryAuthority = () =>
        receivedCompletedProjection(terminalProjection) ||
        (userMessage.admission_observed === true && userMessage.answer_execution === null);

      this.loading = true;
      const streamController = new AbortController();
      this.streamController = streamController;

      try {
        await streamChat(
          {
            message: normalizedQuestion,
            session_id: this.activeSessionId || undefined,
            query_conditions: options?.query_conditions,
            inherit_conditions: options?.inherit_conditions === true,
            signal: streamController.signal,
            token: options?.token || ''
          },
          {
            onAnswerIdentity: (answerId) => {
              const assistantMsg = getAssistantMsg();
              if (!assistantMsg) return;
              if (typeof answerId !== 'string' || !answerId) {
                throw new Error('stream answer identity is malformed');
              }
              if (terminalProjection.answerId) {
                throw new Error('stream answer identity was repeated');
              }
              terminalProjection.answerId = answerId;
              assistantMsg.id = answerId;
              userMessage.admission_observed = true;
              this.streamTick += 1;
            },
            onAnswerExecution: (execution) => {
              const assistantMsg = getAssistantMsg();
              if (!assistantMsg) return;
              if (!execution || typeof execution !== 'object' || Array.isArray(execution)) {
                throw new Error('stream answer execution is malformed');
              }
              if (terminalProjection.execution) {
                throw new Error('stream answer execution was repeated');
              }
              try {
                validateExecutionTurnBinding(execution, submittedTurn);
              } catch (error) {
                invalidateRetryAuthority();
                throw error;
              }
              userMessage.admission_observed = true;
              terminalProjection.execution = execution;
              assistantMsg.answer_execution = execution;
              if (execution.state !== 'completed') {
                userMessage.answer_execution = execution;
                userMessage.retry_authority_invalid = false;
              }
              this.streamTick += 1;
            },
            onOutcome: (outcome) => {
              const assistantMsg = getAssistantMsg();
              if (!assistantMsg) return;
              if (typeof outcome !== 'string' || !outcome) {
                throw new Error('stream completed outcome is malformed');
              }
              if (terminalProjection.outcome) {
                throw new Error('stream completed outcome was repeated');
              }
              terminalProjection.outcome = outcome;
              assistantMsg.outcome = outcome;
              assistantMsg.rejected = outcome === 'insufficient_evidence_reply';
              this.streamTick += 1;
            },
            onInsufficientEvidenceReply: (insufficientEvidenceReply) => {
              const assistantMsg = getAssistantMsg();
              if (!assistantMsg) return;
              if (
                !insufficientEvidenceReply ||
                typeof insufficientEvidenceReply !== 'object' ||
                Array.isArray(insufficientEvidenceReply)
              ) {
                throw new Error('stream structured insufficiency reply is malformed');
              }
              if (terminalProjection.insufficientEvidenceReply) {
                throw new Error('stream structured insufficiency reply was repeated');
              }
              terminalProjection.insufficientEvidenceReply = insufficientEvidenceReply;
              assistantMsg.insufficient_evidence_reply = insufficientEvidenceReply;
              this.streamTick += 1;
            },
            onStage: (stage) => {
              const assistantMsg = getAssistantMsg();
              if (!assistantMsg) return;
              userMessage.admission_observed = true;
              if (assistantMsg.isThinking && stage?.message) {
                assistantMsg.status = stage.message;
              }
              this.streamTick += 1;
            },
            onContent: (chunk) => {
              const assistantMsg = getAssistantMsg();
              if (!assistantMsg) return;
              assistantMsg.isThinking = false;
              assistantMsg.streaming = true;
              assistantMsg.status = '生成中...';
              assistantMsg.content += chunk || '';
              this.streamTick += 1;
            },
            onEvidenceSummary: (evidenceSummary) => {
              const assistantMsg = getAssistantMsg();
              if (!assistantMsg) return;
              if (!evidenceSummary || typeof evidenceSummary !== 'object' || Array.isArray(evidenceSummary)) {
                throw new Error('stream evidence summary is malformed');
              }
              if (terminalProjection.evidenceSummary) {
                throw new Error('stream evidence summary was repeated');
              }
              terminalProjection.evidenceSummary = evidenceSummary;
              assistantMsg.evidence_summary = evidenceSummary;
              this.streamTick += 1;
            },
            onRetrievalDiagnostics: (diagnostics) => {
              const assistantMsg = getAssistantMsg();
              if (!assistantMsg) return;
              assistantMsg.retrieval_diagnostics = diagnostics || null;
              this.streamTick += 1;
            },
            onError: (err) => {
              const assistantMsg = getAssistantMsg();
              if (!assistantMsg) return;
              if (requiresRecoveredRetryAuthority()) {
                invalidateRetryAuthority();
              }
              clearUntrustedCompletedProjection(assistantMsg);
              assistantMsg.streaming = false;
              assistantMsg.isThinking = false;
              assistantMsg.failed = true;
              assistantMsg.status = '回答失败，可重试';
              if (!assistantMsg.content) {
                assistantMsg.content = `请求失败：${formatStreamError(err)}`;
              }
              this.streamTick += 1;
            },
            onDone: () => {
              const assistantMsg = getAssistantMsg();
              if (!assistantMsg) return;
              if (assistantMsg.failed) return;
              validateCompletedStreamProjection(assistantMsg, submittedTurn);
              userMessage.answer_execution = terminalProjection.execution;
              userMessage.admission_observed = true;
              userMessage.retry_authority_invalid = false;
              assistantMsg.streaming = false;
              assistantMsg.isThinking = false;
              assistantMsg.status = getDoneStatus(assistantMsg);
              this.streamTick += 1;
            }
          }
        );
      } catch (error) {
        const assistantMsg = getAssistantMsg();
        if (!assistantMsg) return;
        assistantMsg.streaming = false;
        assistantMsg.isThinking = false;
        if (error?.name === 'AbortError' && !receivedCompletedProjection(terminalProjection)) {
          assistantMsg.status = '回答已停止，内容不完整';
          if (!assistantMsg.content) {
            assistantMsg.content = '回答已停止，未生成可保留的内容。';
          }
        } else {
          if (requiresRecoveredRetryAuthority()) {
            invalidateRetryAuthority();
          }
          clearUntrustedCompletedProjection(assistantMsg);
          assistantMsg.failed = true;
          assistantMsg.status = '回答失败，可重试';
          if (!assistantMsg.content) {
            assistantMsg.content = `请求失败：${formatStreamError(error)}`;
          }
        }
        this.streamTick += 1;
      } finally {
        if (this.streamController === streamController) {
          this.streamController = null;
          this.loading = false;
        }

        // 保持轻量同步：只刷新会话列表，不覆盖当前正在展示的流式文本
        await this.loadSessions();
      }
    }
  }
});
