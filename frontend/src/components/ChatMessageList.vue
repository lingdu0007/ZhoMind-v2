<template>
  <section ref="listRef" class="message-list" aria-label="对话内容">
    <div v-if="messages.length === 0" class="empty">
      <p class="empty__eyebrow">内部知识库</p>
      <h2>从团队知识开始提问</h2>
      <p>可查询部署规范、事故手册、产品决策和运行流程。</p>
    </div>
    <article
      v-for="(msg, idx) in messages"
      :key="idx"
      class="msg"
      :class="`msg--${msg.role}`"
      :aria-label="msg.role === 'user' ? '用户消息' : '助手消息'"
    >
      <header class="msg-head">
        <span class="role" :class="msg.role">{{ msg.role === 'user' ? '你' : '助手' }}</span>
        <span v-if="msg.isThinking" class="status-dot" role="status">正在检索与生成回答</span>
        <span v-else-if="msg.streaming" class="status-dot" role="status">正在生成回答</span>
        <span v-else-if="msg.status" class="status-text" :class="statusClass(msg.status)" role="status">{{ msg.status }}</span>
      </header>
      <div class="content">{{ msg.content }}</div>
      <div v-if="msg.rejected" class="reject-tip">拒答原因：知识片段不足，建议补充关键词或限定范围。</div>
      <button v-if="msg.failed" type="button" class="retry-button" :disabled="retryDisabled" @click="$emit('retry', idx)">重试</button>

      <section v-if="msg.evidence_summary" class="evidence-summary" aria-label="证据摘要">
        <div class="evidence-summary__heading">
          <span>证据摘要</span>
          <strong>{{ getEvidenceCoverageLabel(msg.evidence_summary.coverage) }}</strong>
        </div>
        <p class="evidence-summary__count">{{ msg.evidence_summary.source_count || 0 }} 个来源</p>
        <ul v-if="msg.evidence_summary.sources?.length" class="evidence-summary__sources">
          <li v-for="source in msg.evidence_summary.sources" :key="source.source_id">
            <button
              type="button"
              class="evidence-summary__source"
              :aria-label="`查看来源 ${getEvidenceSourceLabel(source)}`"
              @click="$emit('open-source', { source, trigger: $event.currentTarget })"
            >
              <span>{{ getEvidenceSourceLabel(source) }}</span>
              <span>查看摘录</span>
            </button>
          </li>
        </ul>
        <p v-else class="evidence-summary__empty">没有可供核对的来源摘录。</p>
      </section>
      <RetrievalDiagnostics v-if="showDiagnostics && msg.retrieval_diagnostics" :diagnostics="msg.retrieval_diagnostics" />
    </article>
  </section>
</template>

<script setup>
import { nextTick, ref, watch } from 'vue';
import { getEvidenceCoverageLabel, getEvidenceSourceLabel } from '../app/evidence-summary';
import RetrievalDiagnostics from './RetrievalDiagnostics.vue';

const listRef = ref(null);

const statusClass = (status) => {
  if (!status) return '';
  if (status.includes('失败')) return 'status-error';
  if (status.includes('拒答')) return 'status-reject';
  if (status.includes('停止')) return 'status-stop';
  return '';
};

const props = defineProps({
  messages: {
    type: Array,
    default: () => []
  },
  retryDisabled: {
    type: Boolean,
    default: false
  },
  showDiagnostics: {
    type: Boolean,
    default: false
  }
});

defineEmits(['retry', 'open-source']);

const scrollToBottom = async () => {
  await nextTick();
  if (!listRef.value) return;
  listRef.value.scrollTop = listRef.value.scrollHeight;
};

watch(
  () => props.messages.map((m) => `${m.role}|${m.content?.length || 0}|${m.status || ''}`).join(';'),
  () => {
    scrollToBottom();
  }
);
</script>

<style scoped>
.message-list {
  min-height: 468px;
  display: flex;
  flex-direction: column;
  border-top: 1px solid var(--color-rule);
  border-bottom: 1px solid var(--color-rule);
  max-height: min(62vh, 720px);
  overflow-y: auto;
}

.empty {
  margin: auto 0;
  padding: 40px 0;
  color: var(--color-ink-soft);
  text-align: center;
}

.empty__eyebrow {
  margin: 0 0 8px;
  color: var(--color-copper-strong);
  font-family: var(--font-mono);
  font-size: 11px;
  line-height: 1.5;
}

.empty h2 {
  margin: 0;
  color: var(--color-ink);
  font-family: var(--font-display);
  font-size: 22px;
  font-weight: 600;
  line-height: 1.45;
}

.empty > p:last-child {
  max-width: 360px;
  margin: 12px auto 0;
  font-size: 14px;
  line-height: 1.75;
}

.msg {
  width: min(100%, 680px);
  padding: 20px 0;
  border-bottom: 1px solid var(--color-rule);
}

.msg--user {
  align-self: flex-end;
  width: min(78%, 560px);
  margin: 16px 0 12px;
  padding: 14px 16px;
  border: 1px solid var(--color-rule);
  border-radius: var(--radius-control);
  background: var(--color-paper-muted);
}

.msg--assistant {
  align-self: flex-start;
}

.role {
  font-size: 13px;
  font-weight: 600;
  color: var(--color-ink-soft);
}

.role.user {
  color: var(--color-copper-strong);
}

.role.assistant {
  color: var(--color-moss);
}

.msg-head {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.content {
  color: var(--color-ink);
  white-space: pre-wrap;
  font-size: 15px;
  line-height: 1.8;
}

.stream-status {
  margin-top: 8px;
  font-size: 12px;
  color: var(--cta);
}

.status-dot,
.status-text {
  font-size: 12px;
  color: var(--color-copper-strong);
}

.status-error {
  color: var(--color-danger);
}

.status-reject {
  color: var(--color-warning);
}

.status-stop {
  color: var(--color-ink-soft);
}

.reject-tip {
  margin-top: 12px;
  padding: 10px 12px;
  border-left: 3px solid var(--color-warning);
  background: var(--color-warning-soft);
  color: var(--color-warning);
  font-size: 13px;
  line-height: 1.65;
}

.retry-button {
  min-height: 32px;
  margin-top: 12px;
  padding: 4px 10px;
  border: 1px solid var(--color-rule);
  border-radius: var(--radius-control);
  background: var(--color-paper-raised);
  color: var(--color-ink);
  font: inherit;
  font-size: 13px;
  cursor: pointer;
  transition: color 180ms ease-out, background-color 180ms ease-out, border-color 180ms ease-out, transform 180ms ease-out;
}

.retry-button:hover:not(:disabled) {
  border-color: var(--color-copper);
  color: var(--color-copper-strong);
}

.retry-button:active:not(:disabled) {
  transform: translateY(1px);
}

.retry-button:disabled {
  cursor: not-allowed;
  opacity: 0.58;
}

.evidence-summary {
  margin-top: 12px;
  padding: 12px 16px;
  border-left: 3px solid var(--color-moss);
  background: var(--color-moss-soft);
}

.evidence-summary__heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  color: var(--color-ink);
  font-size: 13px;
}

.evidence-summary__heading strong {
  color: var(--color-moss);
  font-size: 12px;
}

.evidence-summary__count,
.evidence-summary__empty {
  margin: 8px 0 0;
  color: var(--color-ink-soft);
  font-size: 12px;
  line-height: 1.6;
}

.evidence-summary__sources {
  margin: 0;
  padding: 8px 0 0;
  list-style: none;
}

.evidence-summary__source {
  width: 100%;
  min-height: 36px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 8px 0;
  border: 0;
  border-top: 1px solid color-mix(in srgb, var(--color-moss) 28%, transparent);
  background: transparent;
  color: var(--color-ink);
  cursor: pointer;
  font: inherit;
  font-size: 13px;
  text-align: left;
  transition: transform 180ms ease-out;
}

.evidence-summary__source:hover {
  color: var(--color-copper-strong);
}

.evidence-summary__source:active {
  transform: translateY(1px);
}

.evidence-summary__source span:first-child {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.evidence-summary__source span:last-child {
  flex: 0 0 auto;
  color: var(--color-moss);
  font-size: 12px;
}

@media (max-width: 640px) {
  .message-list {
    min-height: 390px;
    max-height: none;
  }

  .msg,
  .msg--user {
    width: 100%;
  }

  .empty {
    padding: 32px 12px;
  }
}
</style>
