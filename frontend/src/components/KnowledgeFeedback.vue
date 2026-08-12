<template>
  <section v-if="entryOptions.length && message.id" class="knowledge-feedback" aria-label="知识反馈">
    <div v-if="submitted" class="knowledge-feedback__submitted" role="status">
      <CheckCircle2 :size="16" aria-hidden="true" />
      <span>反馈已提交</span>
      <span>保留 {{ submitted.retention_days }} 天</span>
    </div>
    <form v-else @submit.prevent="submit">
      <div class="knowledge-feedback__heading">
        <span>知识反馈</span>
        <select v-if="entryOptions.length > 1" v-model="entryId" aria-label="关联知识条目">
          <option v-for="entry in entryOptions" :key="entry.id" :value="entry.id">{{ entry.title }}</option>
        </select>
      </div>
      <div class="knowledge-feedback__labels" role="radiogroup" aria-label="反馈类型">
        <button
          v-for="option in labelOptions"
          :key="option.value"
          type="button"
          role="radio"
          :aria-checked="label === option.value"
          :class="{ active: label === option.value }"
          @click="label = option.value"
        >
          <component :is="option.icon" :size="15" aria-hidden="true" />
          <span>{{ option.text }}</span>
        </button>
      </div>
      <textarea
        v-model="note"
        aria-label="补充说明（可选）"
        maxlength="500"
        rows="2"
        placeholder="补充说明（可选）"
      />
      <div class="knowledge-feedback__actions">
        <span v-if="error" role="alert">{{ error }}</span>
        <button type="submit" :disabled="!label || loading">
          <Send :size="15" aria-hidden="true" />
          <span>{{ loading ? '提交中' : '提交反馈' }}</span>
        </button>
      </div>
    </form>
  </section>
</template>

<script setup>
import { computed, ref, watch } from 'vue';
import { Ban, CheckCircle2, CircleHelp, Clock3, Send } from 'lucide-vue-next';
import { apiAdapter } from '../api/adapters';

const props = defineProps({
  message: {
    type: Object,
    required: true
  }
});

const labelOptions = [
  { value: 'helpful', text: '有帮助', icon: CheckCircle2 },
  { value: 'insufficient_evidence', text: '证据不足', icon: CircleHelp },
  { value: 'outdated', text: '已过时', icon: Clock3 },
  { value: 'out_of_scope', text: '超出范围', icon: Ban }
];

const entryOptions = computed(() => {
  const entries = new Map();
  for (const source of props.message?.evidence_summary?.sources || []) {
    const id = source?.entry_id || source?.metadata?.entry_id;
    if (!id) continue;
    entries.set(id, { id, title: source?.entry_title || source?.metadata?.entry_title || id });
  }
  return [...entries.values()];
});

const entryId = ref('');
const label = ref('');
const note = ref('');
const error = ref('');
const loading = ref(false);
const submitted = ref(null);

watch(entryOptions, (entries) => {
  if (!entries.some((entry) => entry.id === entryId.value)) entryId.value = entries[0]?.id || '';
}, { immediate: true });

const submit = async () => {
  if (!label.value || !entryId.value || loading.value) return;
  loading.value = true;
  error.value = '';
  try {
    submitted.value = await apiAdapter.submitKnowledgeFeedback({
      answer_id: props.message.id,
      entry_id: entryId.value,
      label: label.value,
      note: note.value.trim() || null
    });
  } catch (requestError) {
    error.value = requestError.message || '反馈提交失败。';
  } finally {
    loading.value = false;
  }
};
</script>

<style scoped>
.knowledge-feedback {
  margin-top: 16px;
  padding-top: 14px;
  border-top: 1px solid var(--color-rule);
}

.knowledge-feedback form {
  display: grid;
  gap: 10px;
}

.knowledge-feedback__heading,
.knowledge-feedback__actions,
.knowledge-feedback__submitted {
  display: flex;
  align-items: center;
  gap: 10px;
}

.knowledge-feedback__heading {
  justify-content: space-between;
  color: var(--color-ink-soft);
  font-size: 12px;
  font-weight: 600;
}

.knowledge-feedback__heading select,
.knowledge-feedback__actions button,
.knowledge-feedback textarea {
  border: 1px solid var(--color-rule);
  border-radius: var(--radius-control);
  background: var(--color-paper-raised);
  color: var(--color-ink);
  font: inherit;
}

.knowledge-feedback__heading select {
  min-width: 0;
  max-width: 240px;
  height: 32px;
  padding: 0 8px;
  font-size: 12px;
}

.knowledge-feedback__labels {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  border: 1px solid var(--color-rule);
  border-radius: var(--radius-control);
  overflow: hidden;
}

.knowledge-feedback__labels button {
  min-width: 0;
  min-height: 36px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 4px 8px;
  border: 0;
  border-right: 1px solid var(--color-rule);
  background: var(--color-paper-raised);
  color: var(--color-ink-soft);
  cursor: pointer;
  font: inherit;
  font-size: 12px;
}

.knowledge-feedback__labels button:last-child {
  border-right: 0;
}

.knowledge-feedback__labels button.active {
  background: var(--color-moss-soft);
  color: var(--color-moss);
  font-weight: 600;
}

.knowledge-feedback textarea {
  width: 100%;
  min-height: 54px;
  resize: vertical;
  padding: 8px 10px;
  font-size: 13px;
  line-height: 1.5;
}

.knowledge-feedback__actions {
  justify-content: flex-end;
  min-height: 32px;
}

.knowledge-feedback__actions > span {
  margin-right: auto;
  color: var(--color-danger);
  font-size: 12px;
}

.knowledge-feedback__actions button {
  min-height: 32px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  cursor: pointer;
  font-size: 12px;
}

.knowledge-feedback__actions button:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.knowledge-feedback__submitted {
  color: var(--color-moss);
  font-size: 12px;
}

.knowledge-feedback__submitted span:last-child {
  margin-left: auto;
  color: var(--color-ink-soft);
}

@media (max-width: 640px) {
  .knowledge-feedback__labels {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .knowledge-feedback__labels button:nth-child(2) {
    border-right: 0;
  }

  .knowledge-feedback__labels button:nth-child(-n + 2) {
    border-bottom: 1px solid var(--color-rule);
  }
}
</style>
