<template>
  <section
    v-if="canRender"
    class="knowledge-feedback"
    :aria-label="isGapReport ? '知识缺口报告' : '知识反馈'"
  >
    <p v-if="existingLoading" class="knowledge-feedback__loading" role="status">正在加载已提交反馈…</p>
    <template v-else-if="activeSignal">
      <div class="knowledge-feedback__submitted" role="status">
        <CheckCircle2 :size="16" aria-hidden="true" />
        <span>{{ isGapReport ? '缺口报告已提交' : '反馈已提交' }}</span>
        <select
          v-if="!isGapReport && entryOptions.length > 1"
          v-model="entryId"
          aria-label="关联知识条目"
        >
          <option v-for="entry in entryOptions" :key="entry.id" :value="entry.id">{{ entry.title }}</option>
        </select>
        <span class="knowledge-feedback__retention">保留 {{ activeSignal.retention_days }} 天</span>
        <button type="button" :disabled="loading" @click="deleteFeedback">
          <Trash2 :size="15" aria-hidden="true" />
          <span>{{ loading ? '删除中' : '删除反馈' }}</span>
        </button>
      </div>
      <dl v-if="submittedConfirmation?.entryId === activeSignal.entry_id" class="knowledge-feedback__confirmation" aria-label="已共享反馈">
        <div><dt>反馈类型</dt><dd>{{ feedbackLabel(submittedConfirmation.label) }}</dd></div>
        <div v-if="submittedConfirmation.note"><dt>已共享说明</dt><dd>{{ submittedConfirmation.note }}</dd></div>
      </dl>
      <p v-if="error" class="knowledge-feedback__error" role="alert">{{ error }}</p>
    </template>
    <template v-else-if="isGapReport">
      <button
        v-if="gapStage === 'idle'"
        type="button"
        class="knowledge-feedback__launch"
        @click="gapStage = 'edit'"
      >
        <CircleHelp :size="15" aria-hidden="true" />
        <span>报告知识缺口</span>
      </button>
      <form v-else-if="gapStage === 'edit'" @submit.prevent="previewGapReport">
        <div class="knowledge-feedback__heading">
          <span>报告知识缺口</span>
          <strong>{{ gapReasonLabel }}</strong>
        </div>
        <div class="knowledge-feedback__labels" role="radiogroup" aria-label="反馈类型">
          <button
            v-for="option in labelOptions"
            :key="option.value"
            type="button"
            role="radio"
            :aria-checked="label === option.value"
            :class="{ active: label === option.value }"
            :disabled="loading"
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
        <button
          v-if="sharedQuestion"
          type="button"
          class="knowledge-feedback__copy-question"
          @click="copyQuestionIntoSharedDescription"
        >
          <Copy :size="15" aria-hidden="true" />
          <span>复制当前问题到共享说明</span>
        </button>
        <div class="knowledge-feedback__actions">
          <span v-if="error" role="alert">{{ error }}</span>
          <button type="button" @click="cancelGapReport">取消</button>
          <button type="submit">预览缺口报告</button>
        </div>
      </form>
      <section v-else class="knowledge-feedback__preview" aria-label="缺口报告预览">
        <div class="knowledge-feedback__heading">
          <span>缺口报告预览</span>
          <strong>{{ gapReasonLabel }}</strong>
        </div>
        <dl>
          <div><dt>反馈类型</dt><dd>{{ feedbackLabel(label || 'insufficient_evidence') }}</dd></div>
          <div><dt>Answer ID</dt><dd>{{ message.id }}</dd></div>
          <div><dt>Outcome</dt><dd>Insufficient Evidence Reply</dd></div>
          <div><dt>Query Condition Set</dt><dd>{{ gapContext?.query_condition_set_identity }}</dd></div>
          <div v-if="note.trim()"><dt>补充说明</dt><dd>{{ note.trim() }}</dd></div>
        </dl>
        <div class="knowledge-feedback__actions">
          <span v-if="error" role="alert">{{ error }}</span>
          <button type="button" :disabled="loading" @click="cancelGapReport">取消</button>
          <button type="button" @click="gapStage = 'edit'">返回编辑</button>
          <button type="button" :disabled="loading" @click="confirmGapReport">
            <Send :size="15" aria-hidden="true" />
            <span>{{ loading ? '提交中' : '确认提交' }}</span>
          </button>
        </div>
      </section>
    </template>
    <form v-else-if="evidenceStage === 'edit'" @submit.prevent="previewEvidenceFeedback">
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
          :disabled="loading || !entryId"
          @click="selectEvidenceLabel(option.value)"
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
      <button
        v-if="sharedQuestion"
        type="button"
        class="knowledge-feedback__copy-question"
        @click="copyQuestionIntoSharedDescription"
      >
        <Copy :size="15" aria-hidden="true" />
        <span>复制当前问题到共享说明</span>
      </button>
      <div class="knowledge-feedback__actions">
        <span v-if="error" role="alert">{{ error }}</span>
        <button type="button" :disabled="loading" @click="cancelEvidenceFeedback">取消</button>
        <button type="submit" :disabled="!label || loading">
          <Send :size="15" aria-hidden="true" />
          <span>{{ loading ? '提交中' : '预览反馈' }}</span>
        </button>
      </div>
    </form>
    <section v-else class="knowledge-feedback__preview" aria-label="反馈预览">
      <div class="knowledge-feedback__heading">
        <span>反馈预览</span>
        <strong>{{ feedbackLabel(label) }}</strong>
      </div>
      <dl>
        <div><dt>Answer ID</dt><dd>{{ message.id }}</dd></div>
        <div><dt>关联知识条目</dt><dd>{{ selectedEntry?.title || entryId }}</dd></div>
        <div><dt>反馈类型</dt><dd>{{ feedbackLabel(label) }}</dd></div>
        <div v-if="note.trim()"><dt>补充说明</dt><dd>{{ note.trim() }}</dd></div>
      </dl>
      <div class="knowledge-feedback__actions">
        <span v-if="error" role="alert">{{ error }}</span>
        <button type="button" :disabled="loading" @click="cancelEvidenceFeedback">取消</button>
        <button type="button" @click="evidenceStage = 'edit'">返回编辑</button>
        <button type="button" :disabled="loading" @click="confirmEvidenceFeedback">
          <Send :size="15" aria-hidden="true" />
          <span>{{ loading ? '提交中' : '确认提交' }}</span>
        </button>
      </div>
    </section>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { Ban, CheckCircle2, CircleHelp, Clock3, Copy, Send, Trash2 } from 'lucide-vue-next';
import { apiAdapter } from '../api/adapters';

const props = defineProps({
  message: {
    type: Object,
    required: true
  },
  mode: {
    type: String,
    default: 'evidence'
  }
});

const labelOptions = [
  { value: 'helpful', text: '有帮助', icon: CheckCircle2 },
  { value: 'insufficient_evidence', text: '证据不足', icon: CircleHelp },
  { value: 'outdated', text: '已过时', icon: Clock3 },
  { value: 'out_of_scope', text: '超出范围', icon: Ban }
];
const feedbackLabels = new Map(labelOptions.map((option) => [option.value, option.text]));

const entryOptions = computed(() => {
  const entries = new Map();
  for (const source of props.message?.evidence_summary?.sources || []) {
    const id = source?.entry_id || source?.metadata?.entry_id;
    if (!id) continue;
    entries.set(id, { id, title: source?.entry_title || source?.metadata?.entry_title || id });
  }
  return [...entries.values()];
});

const isGapReport = computed(() => props.mode === 'gap');
const gapContext = computed(() => {
  const execution = props.message?.answer_execution;
  const reply = execution?.insufficient_evidence_reply;
  if (
    execution?.state !== 'completed' ||
    execution?.outcome !== 'insufficient_evidence_reply' ||
    execution?.assistant_message_id !== props.message?.id ||
    reply?.outcome !== 'insufficient_evidence_reply' ||
    typeof reply.reason !== 'string' ||
    !reply.reason ||
    typeof reply.query_condition_set_identity !== 'string' ||
    !reply.query_condition_set_identity
  ) {
    return null;
  }
  return {
    outcome: reply.outcome,
    reason: reply.reason,
    query_condition_set_identity: reply.query_condition_set_identity
  };
});
const canRender = computed(() =>
  isGapReport.value
    ? Boolean(props.message?.id && gapContext.value)
    : Boolean(entryOptions.value.length && props.message?.id)
);
const gapReasonLabel = computed(() => gapContext.value?.reason || '');
const selectedEntry = computed(() => entryOptions.value.find((entry) => entry.id === entryId.value) || null);
const sharedQuestion = computed(() => {
  const question = props.message?.answer_execution?.question;
  return typeof question === 'string' && question.trim() ? question.trim() : '';
});

const entryId = ref('');
const label = ref('');
const note = ref('');
const error = ref('');
const loading = ref(false);
const existingLoading = ref(false);
const feedbackSignals = ref([]);
const submittedConfirmation = ref(null);
let feedbackLoadVersion = 0;
const gapStage = ref('idle');
const evidenceStage = ref('edit');

const activeSignal = computed(() => {
  const answerId = props.message?.id;
  if (typeof answerId !== 'string' || !answerId) return null;
  return (
    feedbackSignals.value.find(
      (signal) =>
        signal.answer_id === answerId &&
        (isGapReport.value ? signal.entry_id === null : signal.entry_id === entryId.value)
    ) || null
  );
});

const feedbackLabel = (value) => feedbackLabels.get(value) || '';

const normalizeSignals = (items, answerId) => {
  if (!Array.isArray(items)) return [];
  return items.filter(
    (signal) =>
      signal &&
      typeof signal.id === 'string' &&
      signal.id &&
      signal.answer_id === answerId &&
      (typeof signal.entry_id === 'string' || signal.entry_id === null) &&
      feedbackLabels.has(signal.label) &&
      Number.isInteger(signal.retention_days) &&
      signal.retention_days > 0
  );
};

const resetFeedbackState = () => {
  entryId.value = entryOptions.value[0]?.id || '';
  label.value = '';
  note.value = '';
  error.value = '';
  loading.value = false;
  existingLoading.value = false;
  feedbackSignals.value = [];
  submittedConfirmation.value = null;
  gapStage.value = 'idle';
  evidenceStage.value = 'edit';
};

const isCurrentAnswer = (answerId, mode) =>
  props.message?.id === answerId && props.mode === mode;

const loadFeedbackSignals = async () => {
  const answerId = props.message?.id;
  const mode = props.mode;
  if (typeof answerId !== 'string' || !answerId) return;
  const loadVersion = ++feedbackLoadVersion;
  existingLoading.value = true;
  try {
    const response = await apiAdapter.listKnowledgeFeedback(answerId);
    if (loadVersion === feedbackLoadVersion && isCurrentAnswer(answerId, mode)) {
      feedbackSignals.value = normalizeSignals(response?.items, answerId);
    }
  } catch (requestError) {
    if (loadVersion === feedbackLoadVersion && isCurrentAnswer(answerId, mode)) {
      error.value = requestError.message || '已提交反馈加载失败。';
    }
  } finally {
    if (loadVersion === feedbackLoadVersion && isCurrentAnswer(answerId, mode)) existingLoading.value = false;
  }
};

watch(
  () => [props.message?.id || '', props.mode],
  () => {
    resetFeedbackState();
    void loadFeedbackSignals();
  },
  { immediate: true }
);

watch(entryOptions, (entries) => {
  if (!entries.some((entry) => entry.id === entryId.value)) entryId.value = entries[0]?.id || '';
}, { immediate: true });

const recordSubmittedSignal = (response, answerId, mode, scopeEntryId, confirmation) => {
  if (!isCurrentAnswer(answerId, mode)) return;
  const [signal] = normalizeSignals([response], answerId);
  if (!signal || signal.entry_id !== scopeEntryId) {
    error.value = '反馈提交未返回可验证的保留记录。';
    return;
  }
  feedbackSignals.value = [
    ...feedbackSignals.value.filter((item) => item.entry_id !== scopeEntryId),
    signal
  ];
  submittedConfirmation.value = confirmation;
  label.value = '';
  note.value = '';
  gapStage.value = 'idle';
  evidenceStage.value = 'edit';
  window.dispatchEvent(new Event('knowledge-feedback-changed'));
};

const previewEvidenceFeedback = () => {
  if (!label.value || !entryId.value) return;
  error.value = '';
  evidenceStage.value = 'preview';
};

const submitFeedback = async ({ answerId, mode, scopeEntryId, selectedLabel, selectedNote }) => {
  if (loading.value || typeof answerId !== 'string' || !answerId) return;
  feedbackLoadVersion += 1;
  loading.value = true;
  error.value = '';
  try {
    const response = await apiAdapter.submitKnowledgeFeedback({
      answer_id: answerId,
      ...(scopeEntryId === null ? {} : { entry_id: scopeEntryId }),
      label: selectedLabel,
      note: selectedNote
    });
    recordSubmittedSignal(response, answerId, mode, scopeEntryId, {
      label: selectedLabel,
      note: selectedNote,
      entryId: scopeEntryId
    });
  } catch (requestError) {
    if (isCurrentAnswer(answerId, mode)) {
      error.value = isGapReport.value
        ? requestError.message || '缺口报告提交失败。'
        : requestError.message || '反馈提交失败。';
    }
  } finally {
    if (isCurrentAnswer(answerId, mode)) loading.value = false;
  }
};

const selectEvidenceLabel = (value) => {
  if (!entryId.value || loading.value) return;
  label.value = value;
};

const cancelEvidenceFeedback = () => {
  evidenceStage.value = 'edit';
  label.value = '';
  note.value = '';
  error.value = '';
};

const copyQuestionIntoSharedDescription = () => {
  if (!sharedQuestion.value) return;
  note.value = sharedQuestion.value;
  error.value = '';
};

const confirmEvidenceFeedback = async () => {
  if (!label.value || !entryId.value || loading.value) return;
  const answerId = props.message?.id;
  const mode = props.mode;
  const scopeEntryId = entryId.value;
  const selectedLabel = label.value;
  const selectedNote = note.value.trim() || null;
  await submitFeedback({ answerId, mode, scopeEntryId, selectedLabel, selectedNote });
};

const cancelGapReport = () => {
  gapStage.value = 'idle';
  label.value = '';
  note.value = '';
  error.value = '';
};

const previewGapReport = () => {
  if (!gapContext.value) return;
  if (!label.value) label.value = 'insufficient_evidence';
  error.value = '';
  gapStage.value = 'preview';
};

const confirmGapReport = async () => {
  if (!gapContext.value || loading.value) return;
  const answerId = props.message?.id;
  const mode = props.mode;
  await submitFeedback({
    answerId,
    mode,
    scopeEntryId: null,
    selectedLabel: label.value || 'insufficient_evidence',
    selectedNote: note.value.trim() || null
  });
};

const deleteFeedback = async () => {
  const signal = activeSignal.value;
  const answerId = props.message?.id;
  const mode = props.mode;
  if (!signal || typeof answerId !== 'string' || !answerId || loading.value) return;
  loading.value = true;
  feedbackLoadVersion += 1;
  error.value = '';
  try {
    const response = await apiAdapter.deleteKnowledgeFeedback(signal.id);
    if (response?.deleted !== true) {
      throw new Error('反馈删除未确认。');
    }
    if (isCurrentAnswer(answerId, mode)) {
      feedbackSignals.value = feedbackSignals.value.filter((item) => item.id !== signal.id);
      submittedConfirmation.value = null;
      label.value = '';
      note.value = '';
      gapStage.value = 'idle';
      evidenceStage.value = 'edit';
      window.dispatchEvent(new Event('knowledge-feedback-changed'));
    }
  } catch (requestError) {
    if (isCurrentAnswer(answerId, mode)) {
      error.value = requestError.message || '反馈删除失败。';
    }
  } finally {
    if (isCurrentAnswer(answerId, mode)) loading.value = false;
  }
};

const refreshAfterFeedbackChange = () => {
  void loadFeedbackSignals();
};

onMounted(() => {
  window.addEventListener('knowledge-feedback-changed', refreshAfterFeedbackChange);
});

onBeforeUnmount(() => {
  window.removeEventListener('knowledge-feedback-changed', refreshAfterFeedbackChange);
});
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

.knowledge-feedback__launch,
.knowledge-feedback__heading,
.knowledge-feedback__actions,
.knowledge-feedback__submitted {
  display: flex;
  align-items: center;
  gap: 10px;
}

.knowledge-feedback__confirmation {
  display: grid;
  gap: 4px;
  margin: 8px 0 0;
  color: var(--color-ink-soft);
  font-size: 12px;
  line-height: 1.5;
}

.knowledge-feedback__confirmation div {
  display: flex;
  gap: 8px;
}

.knowledge-feedback__confirmation dt {
  color: var(--color-ink-faint);
}

.knowledge-feedback__confirmation dd {
  margin: 0;
  overflow-wrap: anywhere;
}

.knowledge-feedback__heading {
  justify-content: space-between;
  color: var(--color-ink-soft);
  font-size: 12px;
  font-weight: 600;
}

.knowledge-feedback__launch,
.knowledge-feedback__copy-question,
.knowledge-feedback__heading select,
.knowledge-feedback__submitted select,
.knowledge-feedback__submitted button,
.knowledge-feedback__actions button,
.knowledge-feedback textarea {
  border: 1px solid var(--color-rule);
  border-radius: var(--radius-control);
  background: var(--color-paper-raised);
  color: var(--color-ink);
  font: inherit;
}

.knowledge-feedback__launch {
  min-height: 32px;
  display: inline-flex;
  align-items: center;
  justify-self: start;
  gap: 6px;
  padding: 4px 10px;
  cursor: pointer;
  font: inherit;
  font-size: 12px;
}

.knowledge-feedback__copy-question {
  min-height: 32px;
  display: inline-flex;
  align-items: center;
  justify-self: start;
  gap: 6px;
  padding: 4px 10px;
  cursor: pointer;
  font: inherit;
  font-size: 12px;
}

.knowledge-feedback__heading select,
.knowledge-feedback__submitted select {
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
  flex-wrap: wrap;
  color: var(--color-moss);
  font-size: 12px;
}

.knowledge-feedback__submitted button {
  min-height: 32px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  cursor: pointer;
  font-size: 12px;
}

.knowledge-feedback__submitted button:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.knowledge-feedback__retention {
  margin-left: auto;
  color: var(--color-ink-soft);
}

.knowledge-feedback__loading,
.knowledge-feedback__error {
  margin: 0;
  color: var(--color-ink-soft);
  font-size: 12px;
}

.knowledge-feedback__error {
  color: var(--color-danger);
}

.knowledge-feedback__preview {
  display: grid;
  gap: 10px;
}

.knowledge-feedback__preview dl {
  display: grid;
  gap: 6px;
  margin: 0;
}

.knowledge-feedback__preview dl div {
  display: grid;
  grid-template-columns: minmax(120px, 0.6fr) minmax(0, 1.4fr);
  gap: 10px;
  font-size: 12px;
  line-height: 1.5;
}

.knowledge-feedback__preview dt {
  color: var(--color-ink-soft);
}

.knowledge-feedback__preview dd {
  min-width: 0;
  margin: 0;
  overflow-wrap: anywhere;
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

  .knowledge-feedback__preview dl div {
    grid-template-columns: minmax(0, 1fr);
    gap: 2px;
  }

  .knowledge-feedback__retention {
    margin-left: 0;
  }
}
</style>
