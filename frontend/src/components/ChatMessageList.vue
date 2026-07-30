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

      <div v-if="msg.rag_steps?.length" class="steps">
        <strong>检索步骤</strong>
        <ul>
          <li v-for="(step, i) in msg.rag_steps" :key="i">{{ step }}</li>
        </ul>
      </div>

      <div v-if="msg.rag_trace" class="trace">
        <el-collapse>
          <el-collapse-item title="RAG Trace" name="trace">
            <pre>{{ formatTrace(msg.rag_trace) }}</pre>
          </el-collapse-item>
        </el-collapse>
      </div>
    </article>
  </section>
</template>

<script setup>
import { nextTick, ref, watch } from 'vue';

const listRef = ref(null);

const formatTrace = (trace) => {
  if (typeof trace === 'string') return trace;
  try {
    return JSON.stringify(trace, null, 2);
  } catch {
    return String(trace);
  }
};

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
  }
});

defineEmits(['retry']);

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

.steps {
  margin-top: 8px;
  font-size: 13px;
  color: var(--color-ink-soft);
}

.ref-title {
  font-weight: 600;
  margin-bottom: 4px;
}

.trace {
  margin-top: 12px;
  color: var(--color-ink-soft);
}

pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 12px;
  line-height: 1.5;
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
