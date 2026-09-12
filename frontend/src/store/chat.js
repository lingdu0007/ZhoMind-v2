import { defineStore } from 'pinia';
import { apiAdapter, streamChat } from '../api/adapters';
import {
  findRecoveredClosedExecution,
  validateClosedAssistantProjection,
  validateCompletedStreamProjection,
  validateExecutionTurnBinding,
  validateHistoryUserExecutionProjection
} from './chat-state';

const terminalExecutionStates = new Set(['stopped', 'failed', 'throttled', 'rejected']);
let activeStreamController = null;

const wait = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
const closedSessionRecoveryDelays = [100, 150, 250, 400, 600, 850, 1150, 1500, 1900];

const cloneQueryConditions = (conditions) => {
  if (!Array.isArray(conditions)) return undefined;
  return conditions.map((condition) => ({
    condition_id: condition?.condition_id || '',
    field: condition?.field || '',
    operator: condition?.operator || '',
    value: condition?.value || ''
  }));
};

const clearUntrustedClosedProjection = (message) => {
  if (!message) return;
  message.answer_execution = null;
  message.outcome = '';
  message.evidence_summary = null;
  message.insufficient_evidence_reply = null;
  message.retrieval_diagnostics = null;
  message.content = '';
  message.pending_content = '';
  message.local_terminal_state = '';
  message.local_retryable = false;
};

const isTerminalExecution = (execution) => terminalExecutionStates.has(execution?.state);

const historyContractFailure = (item) => ({
  id: item?.id || '',
  role: 'assistant',
  content: '',
  answer_execution: null,
  outcome: '',
  evidence_summary: null,
  insufficient_evidence_reply: null,
  retrieval_diagnostics: null,
  streaming: false,
  isThinking: false,
  contract_error: '该历史回答没有可验证的闭合执行记录，已阻止展示其内容。'
});

const historyExecutionId = (execution) =>
  typeof execution?.id === 'string' && execution.id ? execution.id : '';

const invalidateHistoryExecutionGroup = (projected, group) => {
  for (const index of group.users) {
    if (projected[index]?.role === 'user') {
      projected[index].answer_execution = null;
    }
  }
  for (const index of group.assistants) {
    projected[index] = historyContractFailure(projected[index]);
  }
};

const projectHistoryMessages = (rawMessages) => {
  const projected = rawMessages.map((item) => {
    const role = item?.type === 'user' ? 'user' : 'assistant';
    const base = {
      id: item?.id || '',
      role,
      content: typeof item?.content === 'string' ? item.content : '',
      timestamp: item?.timestamp,
      answer_execution: item?.answer_execution || null,
      outcome: item?.outcome || '',
      evidence_summary: item?.evidence_summary || null,
      insufficient_evidence_reply: item?.insufficient_evidence_reply || null,
      retrieval_diagnostics: item?.retrieval_diagnostics || null,
      streaming: false,
      isThinking: false,
      contract_error: ''
    };

    if (role === 'user') return base;

    try {
      validateClosedAssistantProjection(base);
      return base;
    } catch {
      return historyContractFailure(item);
    }
  });

  const executionGroups = new Map();
  rawMessages.forEach((item, index) => {
    const execution = item?.answer_execution;
    if (execution === undefined || execution === null) return;
    const executionId = historyExecutionId(execution);
    if (!executionId) {
      if (projected[index]?.role === 'user') projected[index].answer_execution = null;
      return;
    }
    const group = executionGroups.get(executionId) || { users: [], assistants: [] };
    if (projected[index]?.role === 'user') group.users.push(index);
    else group.assistants.push(index);
    executionGroups.set(executionId, group);
  });

  for (const group of executionGroups.values()) {
    try {
      if (group.assistants.length === 0) {
        if (group.users.length !== 1) {
          throw new Error('standalone history execution has duplicate user bindings');
        }
        validateHistoryUserExecutionProjection(projected[group.users[0]], projected);
        continue;
      }
      if (group.users.length !== 1 || group.assistants.length !== 1) {
        throw new Error('assistant-backed history execution has incomplete or duplicate message bindings');
      }
      const userMessage = projected[group.users[0]];
      const assistantMessage = projected[group.assistants[0]];
      if (
        !userMessage?.answer_execution ||
        assistantMessage?.contract_error ||
        userMessage.answer_execution.assistant_message_id !== assistantMessage?.id
      ) {
        throw new Error('assistant-backed history execution has contradictory message bindings');
      }
      validateHistoryUserExecutionProjection(userMessage, projected);
    } catch {
      invalidateHistoryExecutionGroup(projected, group);
    }
  }
  return projected;
};

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
      activeStreamController?.abort();
      this.messages = [];
      this.sessions = [];
      this.activeSessionId = '';
      this.streamController = null;
      activeStreamController = null;
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
      this.messages = projectHistoryMessages(Array.isArray(rawMessages) ? rawMessages : []);
    },
    async recoverClosedSession(sessionId, submittedTurn, knownExecutionIds, expectedAssistantMessageId) {
      if (
        !sessionId ||
        typeof expectedAssistantMessageId !== 'string' ||
        !expectedAssistantMessageId
      ) {
        return false;
      }
      for (const delay of closedSessionRecoveryDelays) {
        await wait(delay);
        try {
          const data = await apiAdapter.getSessionMessages(sessionId);
          const rawMessages = data?.messages || data?.items || data?.data || [];
          const recovered = projectHistoryMessages(Array.isArray(rawMessages) ? rawMessages : []);
          const recoveredExecution = findRecoveredClosedExecution(
            recovered,
            submittedTurn,
            knownExecutionIds,
            expectedAssistantMessageId
          );
          if (!recoveredExecution) {
            continue;
          }
          if (this.activeSessionId === sessionId) {
            this.messages = recovered;
          }
          return true;
        } catch {
          // The cancellation path is allowed to race the terminal persistence write.
        }
      }
      return false;
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
      activeStreamController?.abort();
    },
    async sendMessage(question, options = {}) {
      const normalizedQuestion = typeof question === 'string' ? question.trim() : '';
      if (!normalizedQuestion || this.loading) return;

      if (!this.activeSessionId || this.activeSessionId === 'default_session') {
        this.activeSessionId = `session_${crypto.randomUUID()}`;
      }

      const priorExecutionIds = this.messages
        .map((message) => message?.answer_execution)
        .filter((execution) => execution && typeof execution === 'object' && !Array.isArray(execution))
        .map((execution) => execution.id)
        .filter((executionId) => typeof executionId === 'string' && executionId);
      const priorCompletedExecutions = this.messages
        .filter(
          (message) =>
            message?.role === 'assistant' &&
            !message.contract_error &&
            message.answer_execution?.state === 'completed'
        )
        .map((message) => message.answer_execution);
      const submittedTurn = {
        question: normalizedQuestion,
        query_conditions: cloneQueryConditions(options?.query_conditions),
        inherit_conditions: options?.inherit_conditions === true,
        known_executions: priorCompletedExecutions
      };
      const localStreamId = `stream_${Date.now()}_${Math.random().toString(16).slice(2)}`;
      const userMessage = {
        local_stream_id: localStreamId,
        role: 'user',
        content: normalizedQuestion,
        answer_execution: null,
        requested_query_conditions: cloneQueryConditions(options?.query_conditions),
        requested_inherit_conditions: options?.inherit_conditions === true,
        admission_observed: false,
        retry_authority_invalid: false
      };
      this.messages.push(userMessage);
      const assistantMessage = {
        id: '',
        local_stream_id: localStreamId,
        role: 'assistant',
        content: '',
        pending_content: '',
        answer_execution: null,
        outcome: '',
        evidence_summary: null,
        insufficient_evidence_reply: null,
        retrieval_diagnostics: null,
        streaming: true,
        isThinking: true,
        contract_error: '',
        status: '思考中...',
        local_terminal_state: '',
        local_retryable: false
      };
      this.messages.push(assistantMessage);

      const submittedSessionId = this.activeSessionId;
      const getAssistantMsg = () =>
        this.activeSessionId === submittedSessionId
          ? this.messages.find(
              (message) => message?.role === 'assistant' && message?.local_stream_id === localStreamId
            ) || null
          : null;
      const getUserMsg = () =>
        this.activeSessionId === submittedSessionId
          ? this.messages.find(
              (message) => message?.role === 'user' && message?.local_stream_id === localStreamId
            ) || null
          : null;
      const terminalProjection = {
        answerId: '',
        execution: null,
        outcome: '',
        evidenceSummary: null,
        insufficientEvidenceReply: null,
        protocolError: false,
        doneObserved: false,
        localTerminalSettled: false
      };
      const invalidateRetryAuthority = () => {
        const currentUserMessage = getUserMsg();
        if (!currentUserMessage) return;
        currentUserMessage.answer_execution = null;
        currentUserMessage.retry_authority_invalid = true;
      };
      const requiresRecoveredRetryAuthority = () =>
        Boolean(terminalProjection.execution || terminalProjection.answerId || terminalProjection.outcome) ||
        (getUserMsg()?.admission_observed === true && getUserMsg()?.answer_execution === null);
      const hasCompletedTerminalFields = () =>
        terminalProjection.execution?.state === 'completed' ||
        Boolean(
          terminalProjection.outcome ||
            terminalProjection.evidenceSummary ||
            terminalProjection.insufficientEvidenceReply
        );
      const canSettlePreAdmissionFailure = () =>
        !terminalProjection.protocolError &&
        !terminalProjection.doneObserved &&
        !terminalProjection.answerId &&
        !terminalProjection.execution &&
        !terminalProjection.outcome &&
        getUserMsg()?.admission_observed !== true;
      const failClosed = (assistantMsg) => {
        if (!assistantMsg) return;
        clearUntrustedClosedProjection(assistantMsg);
        assistantMsg.streaming = false;
        assistantMsg.isThinking = false;
        assistantMsg.contract_error = '流式结果没有形成可验证的闭合执行记录，已阻止展示其内容。';
        assistantMsg.status = '';
      };
      const settlePreAdmissionFailure = (assistantMsg, state) => {
        const currentUserMessage = getUserMsg();
        if (!assistantMsg || !currentUserMessage) return;
        clearUntrustedClosedProjection(assistantMsg);
        assistantMsg.streaming = false;
        assistantMsg.isThinking = false;
        assistantMsg.contract_error = '';
        assistantMsg.status = '';
        assistantMsg.local_terminal_state = state;
        assistantMsg.local_retryable =
          currentUserMessage.requested_inherit_conditions !== true &&
          (currentUserMessage.requested_query_conditions === undefined ||
            Array.isArray(currentUserMessage.requested_query_conditions));
        terminalProjection.localTerminalSettled = true;
      };
      const localFailureState = (error) => {
        if (['CHAT_CONCURRENCY_LIMIT_REACHED', 'CHAT_QUEUE_FULL', 'CHAT_MEMBER_LIMIT'].includes(error?.code)) return 'throttled';
        if (error?.code === 'CHAT_REQUEST_REJECTED') return 'rejected';
        return 'failed';
      };

      this.loading = true;
      const streamController = new AbortController();
      activeStreamController = streamController;
      this.streamController = true;

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
              const currentUserMessage = getUserMsg();
              if (currentUserMessage) currentUserMessage.admission_observed = true;
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
              const currentUserMessage = getUserMsg();
              if (currentUserMessage) currentUserMessage.admission_observed = true;
              terminalProjection.execution = execution;
              assistantMsg.answer_execution = execution;
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
              const currentUserMessage = getUserMsg();
              if (currentUserMessage) currentUserMessage.admission_observed = true;
              if (assistantMsg.isThinking && stage?.message) {
                assistantMsg.status = stage.message;
              }
              this.streamTick += 1;
            },
            onContent: (chunk) => {
              const assistantMsg = getAssistantMsg();
              if (!assistantMsg) return;
              if (terminalProjection.localTerminalSettled) {
                terminalProjection.protocolError = true;
                failClosed(assistantMsg);
                this.streamTick += 1;
                return;
              }
              assistantMsg.isThinking = false;
              assistantMsg.streaming = true;
              assistantMsg.status = '正在等待闭合结果…';
              assistantMsg.pending_content += chunk || '';
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
              if (terminalProjection.localTerminalSettled) return;
              if (isTerminalExecution(terminalProjection.execution)) {
                assistantMsg.isThinking = false;
                assistantMsg.status = '';
                this.streamTick += 1;
                return;
              }
              if (canSettlePreAdmissionFailure()) {
                settlePreAdmissionFailure(assistantMsg, localFailureState(err));
                this.streamTick += 1;
                return;
              }
              terminalProjection.protocolError = true;
              if (requiresRecoveredRetryAuthority()) invalidateRetryAuthority();
              failClosed(assistantMsg);
              this.streamTick += 1;
            },
            onDone: () => {
              const assistantMsg = getAssistantMsg();
              if (!assistantMsg) return;
              if (terminalProjection.localTerminalSettled) {
                terminalProjection.doneObserved = true;
                return;
              }
              if (terminalProjection.protocolError) {
                throw new Error('stream produced an error before a closed terminal execution');
              }
              if (!terminalProjection.execution) {
                if (canSettlePreAdmissionFailure()) {
                  terminalProjection.doneObserved = true;
                  settlePreAdmissionFailure(assistantMsg, 'failed');
                  this.streamTick += 1;
                  return;
                }
                throw new Error('stream has no closed answer execution');
              }
              terminalProjection.doneObserved = true;

              if (terminalProjection.execution.state === 'completed') {
                const completedMessage = {
                  ...assistantMsg,
                  content: assistantMsg.pending_content
                };
                validateCompletedStreamProjection(completedMessage, submittedTurn);
                assistantMsg.content = completedMessage.content;
              } else {
                const terminalMessage = {
                  ...assistantMsg,
                  content: assistantMsg.content || assistantMsg.pending_content
                };
                validateClosedAssistantProjection(terminalMessage, submittedTurn);
                assistantMsg.content = '';
                assistantMsg.pending_content = '';
              }

              const currentUserMessage = getUserMsg();
              if (!currentUserMessage) {
                throw new Error('stream user message is no longer available');
              }
              currentUserMessage.answer_execution = terminalProjection.execution;
              currentUserMessage.admission_observed = true;
              currentUserMessage.retry_authority_invalid = false;
              const persistenceFailureWithoutAssistant =
                terminalProjection.execution.state === 'failed' &&
                terminalProjection.execution.failure_code === 'ANSWER_EXECUTION_PERSISTENCE_FAILED' &&
                (terminalProjection.execution.assistant_message_id === undefined ||
                  terminalProjection.execution.assistant_message_id === null);
              if (persistenceFailureWithoutAssistant) {
                const assistantIndex = this.messages.indexOf(assistantMsg);
                if (assistantIndex >= 0) this.messages.splice(assistantIndex, 1);
              } else {
                assistantMsg.streaming = false;
                assistantMsg.isThinking = false;
                assistantMsg.status = '';
              }
              this.streamTick += 1;
            }
          }
        );
      } catch (error) {
        const assistantMsg = getAssistantMsg();
        if (!assistantMsg) return;
        if (canSettlePreAdmissionFailure()) {
          settlePreAdmissionFailure(
            assistantMsg,
            error?.name === 'AbortError' ? 'stopped' : localFailureState(error)
          );
          this.streamTick += 1;
          return;
        }
        if (requiresRecoveredRetryAuthority()) invalidateRetryAuthority();
        if (
          error?.name === 'AbortError' &&
          !hasCompletedTerminalFields() &&
          terminalProjection.answerId
        ) {
          clearUntrustedClosedProjection(assistantMsg);
          assistantMsg.streaming = true;
          assistantMsg.isThinking = false;
          assistantMsg.contract_error = '';
          assistantMsg.status = '';
          this.streamTick += 1;
          const recovered = await this.recoverClosedSession(
            submittedSessionId,
            submittedTurn,
            new Set(priorExecutionIds),
            terminalProjection.answerId
          );
          if (recovered) return;
        }
        failClosed(assistantMsg);
        this.streamTick += 1;
      } finally {
        if (activeStreamController === streamController) {
          activeStreamController = null;
          this.streamController = null;
          this.loading = false;
        }

        // 保持轻量同步：只刷新会话列表，不覆盖当前正在展示的流式文本
        await this.loadSessions();
      }
    }
  }
});
