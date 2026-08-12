import { defineStore } from 'pinia';
import { apiAdapter, streamChat } from '../api/adapters';
import { formatStreamError, getDoneStatus } from './chat-state';

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
        evidence_summary: item?.evidence_summary || null,
        retrieval_diagnostics: item?.retrieval_diagnostics || null,
        streaming: false,
        isThinking: false,
        rejected: false,
        reject_reason: '',
        status: ''
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
      if (!question?.trim() || this.loading) return;

      if (!this.activeSessionId || this.activeSessionId === 'default_session') {
        this.activeSessionId = `session_${Date.now()}`;
      }

      this.messages.push({ role: 'user', content: question });
      const assistantIndex = this.messages.length;
      this.messages.push({
        id: '',
        role: 'assistant',
        content: '',
        evidence_summary: null,
        retrieval_diagnostics: null,
        streaming: true,
        isThinking: true,
        rejected: false,
        reject_reason: '',
        failed: false,
        status: '思考中...'
      });

      const getAssistantMsg = () => this.messages[assistantIndex];

      this.loading = true;
      this.streamController = new AbortController();

      try {
        await streamChat(
          {
            message: question,
            session_id: this.activeSessionId || undefined,
            signal: this.streamController.signal,
            token: options?.token || ''
          },
          {
            onAnswerIdentity: (answerId) => {
              const assistantMsg = getAssistantMsg();
              if (!assistantMsg) return;
              assistantMsg.id = answerId || '';
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
              assistantMsg.evidence_summary = evidenceSummary || null;
              if (evidenceSummary?.coverage === 'insufficient') {
                assistantMsg.rejected = true;
                assistantMsg.status = '证据不足，进入拒答';
              }
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
              assistantMsg.streaming = false;
              assistantMsg.isThinking = false;
              if (assistantMsg.rejected && !assistantMsg.content) {
                assistantMsg.content = '未检索到足够相关的知识片段，请补充更具体的问题或关键词。';
              }
              if (!assistantMsg.failed) {
                assistantMsg.status = getDoneStatus(assistantMsg);
              }
              this.streamTick += 1;
            }
          }
        );
      } catch (error) {
        const assistantMsg = getAssistantMsg();
        if (!assistantMsg) return;
        assistantMsg.streaming = false;
        assistantMsg.isThinking = false;
        if (error?.name === 'AbortError') {
          assistantMsg.status = '回答已停止，内容不完整';
          if (!assistantMsg.content) {
            assistantMsg.content = '回答已停止，未生成可保留的内容。';
          }
        } else {
          assistantMsg.failed = true;
          assistantMsg.status = '回答失败，可重试';
          if (!assistantMsg.content) {
            assistantMsg.content = `请求失败：${formatStreamError(error)}`;
          }
        }
        this.streamTick += 1;
      } finally {
        this.streamController = null;
        this.loading = false;

        // 保持轻量同步：只刷新会话列表，不覆盖当前正在展示的流式文本
        await this.loadSessions();
      }
    }
  }
});
