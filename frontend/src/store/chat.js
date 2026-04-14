import { defineStore } from 'pinia';
import { apiAdapter, streamChat } from '../api/adapters';
import { extractRejectReason, formatStreamError, getDoneStatus } from './chat-state';

const toText = (value) => {
  if (typeof value === 'string') return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
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
    async loadSessions() {
      const data = await apiAdapter.listSessions();
      this.sessions = data?.sessions || data?.items || data?.data || [];
      if (!this.activeSessionId && this.sessions.length > 0) {
        const preferred =
          this.sessions.find((item) => (item.session_id || item.id) !== 'default_session') ||
          this.sessions[0];
        const candidate = preferred.session_id || preferred.id || '';
        if (candidate && candidate !== 'default_session') {
          this.activeSessionId = candidate;
        }
      }
    },
    async loadSessionMessages(sessionId) {
      if (!sessionId) return;
      const data = await apiAdapter.getSessionMessages(sessionId);
      this.activeSessionId = sessionId;
      const rawMessages = data?.messages || data?.items || data?.data || [];
      this.messages = rawMessages.map((item) => ({
        role: item?.type === 'user' ? 'user' : 'assistant',
        content: item?.content || '',
        timestamp: item?.timestamp,
        rag_trace: item?.rag_trace || null,
        rag_steps: [],
        streaming: false,
        isThinking: false,
        rejected: false,
        reject_reason: '',
        status: ''
      }));
    },
    async deleteSession(sessionId) {
      await apiAdapter.deleteSession(sessionId);
      if (this.activeSessionId === sessionId) {
        this.activeSessionId = '';
        this.messages = [];
      }
      await this.loadSessions();
    },
    stopStreaming() {
      if (this.streamController) {
        this.streamController.abort();
      }
    },
    async sendMessage(question) {
      if (!question?.trim() || this.loading) return;

      if (!this.activeSessionId || this.activeSessionId === 'default_session') {
        this.activeSessionId = `session_${Date.now()}`;
      }

      this.messages.push({ role: 'user', content: question });
      const assistantIndex = this.messages.length;
      this.messages.push({
        role: 'assistant',
        content: '',
        rag_trace: null,
        rag_steps: [],
        streaming: true,
        isThinking: true,
        rejected: false,
        reject_reason: '',
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
            signal: this.streamController.signal
          },
          {
            onContent: (chunk) => {
              const assistantMsg = getAssistantMsg();
              if (!assistantMsg) return;
              assistantMsg.isThinking = false;
              assistantMsg.streaming = true;
              assistantMsg.status = '生成中...';
              assistantMsg.content += chunk || '';
              this.streamTick += 1;
            },
            onRagStep: (step) => {
              const assistantMsg = getAssistantMsg();
              if (!assistantMsg) return;
              assistantMsg.rag_steps.push(toText(step));
              const rejectReason = extractRejectReason(step);
              if (rejectReason) {
                assistantMsg.rejected = true;
                assistantMsg.reject_reason = rejectReason;
                assistantMsg.status = '证据不足，进入拒答';
              }
              this.streamTick += 1;
            },
            onTrace: (trace) => {
              const assistantMsg = getAssistantMsg();
              if (!assistantMsg) return;
              assistantMsg.rag_trace = trace;
            },
            onError: (err) => {
              const assistantMsg = getAssistantMsg();
              if (!assistantMsg) return;
              assistantMsg.streaming = false;
              assistantMsg.isThinking = false;
              assistantMsg.status = '生成失败';
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
        if (error?.name === 'AbortError') {
          assistantMsg.status = '已停止';
          assistantMsg.content = assistantMsg.content
            ? `${assistantMsg.content}(回答已被终止)`
            : '(已终止回答)';
        } else {
          assistantMsg.status = '生成失败';
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
