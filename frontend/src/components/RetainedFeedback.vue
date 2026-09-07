<template>
  <section v-if="visible" class="retained-feedback" aria-label="已保留反馈">
    <header class="retained-feedback__header">
      <div>
        <h2>已保留反馈</h2>
      </div>
      <span v-if="signals.length" class="retained-feedback__count">{{ signals.length }}</span>
    </header>
    <p v-if="loading && !signals.length" class="retained-feedback__status" role="status">正在加载已保留反馈…</p>
    <p v-if="error" class="retained-feedback__status retained-feedback__status--error" role="alert">
      {{ error }}
    </p>
    <ul v-if="signals.length" class="retained-feedback__items">
      <li v-for="signal in signals" :key="signal.id" class="retained-feedback__item">
        <div class="retained-feedback__summary">
          <strong>{{ outcomeLabel(signal.outcome) }}</strong>
          <span>{{ feedbackLabel(signal.label) }}</span>
        </div>
        <dl>
          <div v-if="signal.entry_id">
            <dt>关联条目</dt>
            <dd>{{ signal.entry_id }}</dd>
          </div>
          <div v-else-if="signal.gap_context?.reason">
            <dt>缺口原因</dt>
            <dd>{{ signal.gap_context.reason }}</dd>
          </div>
          <div v-if="signal.knowledge_edition">
            <dt>知识版本</dt>
            <dd>{{ signal.knowledge_edition }}</dd>
          </div>
          <div>
            <dt>提交时间</dt>
            <dd>{{ submittedAt(signal.created_at) }}</dd>
          </div>
          <div>
            <dt>保留</dt>
            <dd>{{ signal.retention_days }} 天</dd>
          </div>
        </dl>
        <button type="button" :disabled="deletingId === signal.id" @click="withdraw(signal)">
          <Trash2 :size="15" aria-hidden="true" />
          <span>{{ deletingId === signal.id ? '撤回中' : '撤回反馈' }}</span>
        </button>
      </li>
    </ul>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { Trash2 } from 'lucide-vue-next';
import { apiAdapter } from '../api/adapters';

const labels = new Map([
  ['helpful', '有帮助'],
  ['insufficient_evidence', '证据不足'],
  ['outdated', '已过时'],
  ['out_of_scope', '超出范围']
]);
const outcomes = new Map([
  ['evidence_gated_answer', 'Supported by published knowledge'],
  ['insufficient_evidence_reply', 'Insufficient Evidence Reply']
]);

const signals = ref([]);
const loading = ref(false);
const error = ref('');
const deletingId = ref('');
let loadVersion = 0;

const visible = computed(() => loading.value || Boolean(error.value) || signals.value.length > 0);

const normalizeSignals = (items) => {
  if (!Array.isArray(items)) return [];
  return items.filter(
    (signal) =>
      signal &&
      typeof signal.id === 'string' &&
      signal.id &&
      labels.has(signal.label) &&
      (signal.outcome === undefined || signal.outcome === null || outcomes.has(signal.outcome)) &&
      typeof signal.created_at === 'string' &&
      Number.isInteger(signal.retention_days) &&
      signal.retention_days > 0 &&
      (typeof signal.entry_id === 'string' || signal.entry_id === null) &&
      (typeof signal.knowledge_edition === 'string' || signal.knowledge_edition === null)
  );
};

const feedbackLabel = (label) => labels.get(label) || '';
const outcomeLabel = (outcome) => outcomes.get(outcome) || '历史反馈';
const submittedAt = (value) => {
  const date = new Date(value);
  return Number.isNaN(date.valueOf())
    ? ''
    : new Intl.DateTimeFormat('zh-CN', { dateStyle: 'medium', timeStyle: 'short' }).format(date);
};

const load = async () => {
  const currentLoadVersion = ++loadVersion;
  loading.value = true;
  error.value = '';
  try {
    const response = await apiAdapter.listKnowledgeFeedback();
    if (currentLoadVersion === loadVersion) signals.value = normalizeSignals(response?.items);
  } catch (requestError) {
    if (currentLoadVersion === loadVersion) {
      signals.value = [];
      error.value = requestError.message || '已保留反馈加载失败。';
    }
  } finally {
    if (currentLoadVersion === loadVersion) loading.value = false;
  }
};

const withdraw = async (signal) => {
  if (!signal || deletingId.value) return;
  loadVersion += 1;
  deletingId.value = signal.id;
  error.value = '';
  try {
    const response = await apiAdapter.deleteKnowledgeFeedback(signal.id);
    if (response?.deleted !== true) throw new Error('反馈撤回未确认。');
    signals.value = signals.value.filter((item) => item.id !== signal.id);
    window.dispatchEvent(new Event('knowledge-feedback-changed'));
  } catch (requestError) {
    error.value = requestError.message || '反馈撤回失败。';
  } finally {
    deletingId.value = '';
  }
};

const refreshAfterFeedbackChange = () => {
  void load();
};

onMounted(() => {
  void load();
  window.addEventListener('knowledge-feedback-changed', refreshAfterFeedbackChange);
});

onBeforeUnmount(() => {
  window.removeEventListener('knowledge-feedback-changed', refreshAfterFeedbackChange);
});
</script>

<style scoped>
.retained-feedback {
  display: grid;
  gap: 12px;
  padding: 16px 0;
  border-top: 1px solid var(--color-rule);
  border-bottom: 1px solid var(--color-rule);
}

.retained-feedback__header,
.retained-feedback__summary,
.retained-feedback__item > button {
  display: flex;
  align-items: center;
}

.retained-feedback__header {
  justify-content: space-between;
  gap: 16px;
}

.retained-feedback__header h2,
.retained-feedback__status,
.retained-feedback__item dl {
  margin: 0;
}

.retained-feedback__header h2 {
  color: var(--color-ink);
  font-size: 14px;
  font-weight: 600;
  line-height: 1.5;
}

.retained-feedback__status {
  color: var(--color-ink-soft);
  font-size: 12px;
  line-height: 1.5;
}

.retained-feedback__status--error {
  color: var(--color-danger);
}

.retained-feedback__count {
  display: inline-flex;
  min-width: 24px;
  min-height: 24px;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--color-rule);
  color: var(--color-ink-soft);
  font-size: 12px;
}

.retained-feedback__items {
  display: grid;
  gap: 0;
  margin: 0;
  padding: 0;
  list-style: none;
  border-top: 1px solid var(--color-rule);
}

.retained-feedback__item {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 10px 16px;
  padding: 12px 0;
  border-bottom: 1px solid var(--color-rule);
}

.retained-feedback__summary {
  gap: 8px;
  min-width: 0;
}

.retained-feedback__summary strong {
  min-width: 0;
  color: var(--color-ink);
  font-size: 13px;
  font-weight: 600;
  overflow-wrap: anywhere;
}

.retained-feedback__summary span {
  color: var(--color-ink-soft);
  font-size: 12px;
}

.retained-feedback__item dl {
  display: grid;
  grid-column: 1 / -1;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px 16px;
}

.retained-feedback__item dl div {
  min-width: 0;
}

.retained-feedback__item dt,
.retained-feedback__item dd {
  margin: 0;
  overflow-wrap: anywhere;
  font-size: 12px;
  line-height: 1.5;
}

.retained-feedback__item dt {
  color: var(--color-ink-soft);
}

.retained-feedback__item dd {
  color: var(--color-ink);
}

.retained-feedback__item > button {
  align-self: start;
  min-height: 32px;
  gap: 6px;
  padding: 4px 10px;
  border: 1px solid var(--color-rule);
  border-radius: var(--radius-control);
  background: var(--color-paper-raised);
  color: var(--color-ink);
  cursor: pointer;
  font: inherit;
  font-size: 12px;
}

.retained-feedback__item > button:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

@media (max-width: 640px) {
  .retained-feedback__item {
    grid-template-columns: minmax(0, 1fr);
  }

  .retained-feedback__item > button {
    justify-self: start;
  }

  .retained-feedback__item dl {
    grid-template-columns: minmax(0, 1fr);
    gap: 6px;
  }
}
</style>
