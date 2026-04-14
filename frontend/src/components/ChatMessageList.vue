<template>
  <div ref="listRef" class="message-list card">
    <div v-if="messages.length === 0" class="empty">开始提问，系统会基于知识库回复。</div>
    <article v-for="(msg, idx) in messages" :key="idx" class="msg">
      <header class="msg-head">
        <span class="role" :class="msg.role">{{ msg.role === 'user' ? '你' : '助手' }}</span>
        <span v-if="msg.isThinking" class="status-dot">思考中...</span>
        <span v-else-if="msg.streaming" class="status-dot">Streaming…</span>
        <span v-else-if="msg.status" class="status-text" :class="statusClass(msg.status)">{{ msg.status }}</span>
      </header>
      <div class="content">{{ msg.content }}</div>
      <div v-if="msg.rejected" class="reject-tip">拒答原因：知识片段不足，建议补充关键词或限定范围。</div>

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
  </div>
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
  }
});

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
  padding: 24px;
  min-height: 480px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  max-height: calc(70vh);
  overflow-y: auto;
}

.empty {
  color: var(--text-muted);
}

.msg {
  border: 1px solid var(--line-soft);
  border-radius: 18px;
  padding: 20px;
  box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
}

.role {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-muted);
}

.role.user {
  color: var(--cta);
}

.role.assistant {
  color: var(--primary);
}

.msg-head {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.content {
  white-space: pre-wrap;
  line-height: 1.6;
}

.stream-status {
  margin-top: 8px;
  font-size: 12px;
  color: var(--cta);
}

.status-dot,
.status-text {
  font-size: 12px;
  color: var(--cta);
}

.status-error {
  color: var(--danger);
}

.status-reject {
  color: #b45309;
}

.status-stop {
  color: var(--text-muted);
}

.reject-tip {
  margin-top: 8px;
  font-size: 12px;
  color: #92400e;
  background: #fffbeb;
  border: 1px solid #fcd34d;
  border-radius: 8px;
  padding: 8px 10px;
}

.steps {
  margin-top: 8px;
  font-size: 13px;
  color: var(--text-muted);
}

.ref-title {
  font-weight: 600;
  margin-bottom: 4px;
}

.trace {
  margin-top: 8px;
}

pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 12px;
  line-height: 1.5;
}
</style>
