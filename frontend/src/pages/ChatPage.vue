<template>
  <section ref="chatSectionRef" class="conversation-workspace" :class="{ 'conversation-workspace--excerpt-open': selectedSource }">
    <SessionDrawer
      class="conversation-workspace__session-rail"
      :loading="sessionsLoading"
      :sessions="chatStore.sessions"
      :active-id="chatStore.activeSessionId"
      @select="openSession"
      @remove="removeSession"
      @refresh="loadSessions"
      @start="startNewSession"
    />

    <div class="conversation-workspace__content">
      <div class="chat-page">
        <div class="top-bar">
        <div>
          <h1>对话工作区</h1>
          <p class="subtitle" v-if="streamSubtitle">{{ streamSubtitle }}</p>
        </div>
        <div class="top-actions">
          <el-button class="btn-ghost" @click="startNewSession">新建会话</el-button>
          <el-button class="btn-ghost session-toggle" @click="toggleSessions">会话</el-button>
        </div>
        </div>

        <el-drawer v-model="sessionVisible" title="最近会话" direction="ltr" size="min(86vw, 320px)">
        <SessionDrawer
          :loading="sessionsLoading"
          :sessions="chatStore.sessions"
          :active-id="chatStore.activeSessionId"
          @select="openSession"
          @remove="removeSession"
          @refresh="loadSessions"
          @start="startNewSession"
        />
        </el-drawer>

        <ChatMessageList
          :messages="chatStore.messages"
          :coverage="publishedCoverage"
          :coverage-error="coverageError"
          :coverage-loading="coverageLoading"
          :retry-disabled="chatStore.loading"
          :show-diagnostics="authStore.isAdmin"
          @ask-coverage="prefillCoverageQuestion"
          @refine="prefillCoverageQuestion"
          @retry="retryAssistantMessage"
          @open-source="openSourceExcerpt"
        />
        <RetainedFeedback />

        <div class="composer card">
        <el-input
          ref="composerInputRef"
          v-model="input"
          type="textarea"
          :rows="3"
          placeholder="请输入需要检索的问题"
          aria-describedby="composer-feedback"
        />
        <section class="query-conditions" aria-label="查询条件">
          <div class="query-conditions__header">
            <h2>查询条件</h2>
            <div class="query-conditions__actions">
              <el-checkbox
                v-model="inheritConditions"
                :disabled="chatStore.loading || !canInheritConditions"
              >沿用上一轮条件</el-checkbox>
              <el-tooltip content="重置查询条件" placement="top">
                <el-button
                  circle
                  :disabled="chatStore.loading"
                  aria-label="重置查询条件"
                  @click="resetQueryConditions"
                >
                  <RotateCcw :size="15" aria-hidden="true" />
                </el-button>
              </el-tooltip>
              <el-tooltip content="添加查询条件" placement="top">
                <el-button
                  circle
                  :disabled="chatStore.loading || inheritConditions"
                  aria-label="添加查询条件"
                  @click="addQueryCondition"
                >
                  <Plus :size="16" aria-hidden="true" />
                </el-button>
              </el-tooltip>
            </div>
          </div>
          <div v-if="!inheritConditions && queryConditions.length" class="query-conditions__rows">
            <div v-for="(condition, index) in queryConditions" :key="condition.condition_id" class="query-conditions__row">
              <el-input
                v-model="condition.field"
                :aria-label="`条件 ${index + 1} 字段`"
                placeholder="字段"
                :disabled="chatStore.loading"
              />
              <el-input
                v-model="condition.operator"
                :aria-label="`条件 ${index + 1} 运算符`"
                placeholder="运算符"
                :disabled="chatStore.loading"
              />
              <el-input
                v-model="condition.value"
                :aria-label="`条件 ${index + 1} 值`"
                placeholder="值"
                :disabled="chatStore.loading"
              />
              <el-tooltip content="删除查询条件" placement="top">
                <el-button
                  circle
                  :disabled="chatStore.loading"
                  :aria-label="`删除条件 ${index + 1}`"
                  @click="removeQueryCondition(index)"
                >
                  <Trash2 :size="15" aria-hidden="true" />
                </el-button>
              </el-tooltip>
            </div>
          </div>
        </section>
        <p v-if="composerError" id="composer-feedback" class="composer-feedback" role="alert">{{ composerError }}</p>
        <div class="composer-actions">
          <el-button class="btn-ghost" :disabled="!chatStore.loading" @click="chatStore.stopStreaming">停止</el-button>
          <el-button type="primary" :loading="chatStore.loading" :disabled="!authStore.isLoggedIn || chatStore.loading" @click="onSend">
            发送
          </el-button>
        </div>
        </div>
      </div>

      <aside v-if="selectedSource" class="evidence-excerpt" role="complementary" aria-label="来源摘录" @keydown.esc.prevent="closeSourceExcerpt">
        <header class="evidence-excerpt__header">
          <div>
            <p>{{ evidenceExcerptHeading }}</p>
            <h2>{{ selectedSource.entry_title || sourceLabel }}</h2>
          </div>
          <button ref="excerptCloseRef" type="button" class="evidence-excerpt__close" aria-label="关闭来源摘录" @click="closeSourceExcerpt">关闭</button>
        </header>
        <p v-if="selectedSource.citation_id || selectedSource.source_id" class="evidence-excerpt__id">
          {{ selectedSource.citation_id || selectedSource.source_id }}
        </p>
        <dl v-if="selectedSource.citation_id" class="evidence-excerpt__metadata">
          <div v-if="selectedSource.entry_id">
            <dt>Entry ID</dt>
            <dd>{{ selectedSource.entry_id }}</dd>
          </div>
          <div v-if="selectedSource.section_id">
            <dt>Section</dt>
            <dd>{{ selectedSource.section_id }}</dd>
          </div>
          <div v-if="selectedSource.snapshot_id">
            <dt>Snapshot</dt>
            <dd>{{ selectedSource.snapshot_id }}</dd>
          </div>
          <div v-if="selectedSource.citation_identity">
            <dt>Citation Identity</dt>
            <dd>{{ selectedSource.citation_identity }}</dd>
          </div>
          <div v-if="controlledSourceLocator">
            <dt>受控来源定位符</dt>
            <dd>{{ controlledSourceLocator }}</dd>
          </div>
          <div v-if="selectedSource.source_authority">
            <dt>来源机构</dt>
            <dd>{{ selectedSource.source_authority }}</dd>
          </div>
          <div v-if="selectedSource.source_version">
            <dt>适用版本</dt>
            <dd>{{ selectedSource.source_version }}</dd>
          </div>
          <div v-if="selectedSource.publication_version">
            <dt>知识版本</dt>
            <dd>{{ selectedSource.publication_version }}</dd>
          </div>
          <div v-if="selectedSource.review_date">
            <dt>复核日期</dt>
            <dd>{{ selectedSource.review_date }}</dd>
          </div>
          <div v-if="selectedSource.review_status">
            <dt>审查状态</dt>
            <dd>{{ selectedSource.review_status }}</dd>
          </div>
          <div v-if="selectedSource.assurance_level">
            <dt>保证级别</dt>
            <dd>{{ selectedSource.assurance_level }}</dd>
          </div>
          <div v-if="formatConditions(selectedSource.applicability_conditions)">
            <dt>适用条件</dt>
            <dd>{{ formatConditions(selectedSource.applicability_conditions) }}</dd>
          </div>
          <div v-if="formatConditions(selectedSource.non_applicability_conditions)">
            <dt>不适用条件</dt>
            <dd>{{ formatConditions(selectedSource.non_applicability_conditions) }}</dd>
          </div>
        </dl>
        <a
          v-if="sourceUrl"
          class="evidence-excerpt__link"
          :href="sourceUrl"
          target="_blank"
          rel="noopener noreferrer"
        >打开公开来源</a>
        <div class="evidence-excerpt__content">{{ selectedSource.withdrawal_notice || selectedSource.excerpt || '未返回可展示的来源摘录。' }}</div>
      </aside>
    </div>
  </section>
</template>

<script setup>
import { computed, nextTick, onMounted, ref, watch } from 'vue';
import { ElMessage, ElMessageBox } from 'element-plus';
import { Plus, RotateCcw, Trash2 } from 'lucide-vue-next';
import ChatMessageList from '../components/ChatMessageList.vue';
import RetainedFeedback from '../components/RetainedFeedback.vue';
import SessionDrawer from '../components/SessionDrawer.vue';
import { useChatStore } from '../store/chat';
import { useAuthStore } from '../store/auth';
import { resolveRetryTurn } from '../store/chat-state';
import {
  getControlledEvidenceSourceLocator,
  getEvidenceSourceLabel,
  getEvidenceSourceUrl
} from '../app/evidence-summary';
import { apiAdapter } from '../api/adapters';
import { useRoute } from 'vue-router';

const chatStore = useChatStore();
const authStore = useAuthStore();
const route = useRoute();
const input = ref('');
const queryConditions = ref([]);
const inheritConditions = ref(false);
const nextConditionNumber = ref(1);
const chatSectionRef = ref(null);
const sessionVisible = ref(false);
const composerError = ref('');
const sessionsLoading = ref(false);
const publishedCoverage = ref(null);
const coverageLoading = ref(false);
const coverageError = ref('');
const selectedSource = ref(null);
const sourceTrigger = ref(null);
const excerptCloseRef = ref(null);
const composerInputRef = ref(null);

const sourceLabel = computed(() => getEvidenceSourceLabel(selectedSource.value));
const sourceUrl = computed(() =>
  selectedSource.value?.withdrawal_notice ? '' : getEvidenceSourceUrl(selectedSource.value)
);
const controlledSourceLocator = computed(() =>
  selectedSource.value?.withdrawal_notice ? '' : getControlledEvidenceSourceLocator(selectedSource.value)
);
const evidenceExcerptHeading = computed(() =>
  selectedSource.value?.section_id === 'recommendation_or_reviewed_branches'
    ? 'Governing Engineering Decision Entry'
    : 'Published Engineering Decision Entry'
);
const canInheritConditions = computed(() =>
  [...chatStore.messages]
    .reverse()
    .some(
      (message) =>
        message.role === 'assistant' &&
        !message.contract_error &&
        message.answer_execution?.state === 'completed'
    )
);

const streamSubtitle = computed(() => {
  return chatStore.loading ? '正在等待闭合执行记录…' : '';
});

const scrollToBottom = async () => {
  await nextTick();
  if (!chatSectionRef.value) return;
  chatSectionRef.value.scrollIntoView({ behavior: 'smooth', block: 'end' });
};

const formatConditions = (conditions) => {
  if (!Array.isArray(conditions)) return '';
  return conditions
    .map((condition) => {
      if (!condition || typeof condition !== 'object') return '';
      const { field, operator, value } = condition;
      return [field, operator, value].every((item) => typeof item === 'string' && item.trim())
        ? `${field.trim()} ${operator.trim()} ${value.trim()}`
        : '';
    })
    .filter(Boolean)
    .join('；');
};

const loadSessions = async () => {
  if (!authStore.isLoggedIn) return;
  sessionsLoading.value = true;
  try {
    await chatStore.loadSessions();
  } catch (error) {
    ElMessage.error(error.message || '加载会话失败');
  } finally {
    sessionsLoading.value = false;
  }
};

const loadPublishedCoverage = async () => {
  if (!authStore.isLoggedIn) return;
  coverageLoading.value = true;
  coverageError.value = '';
  publishedCoverage.value = null;
  try {
    const payload = await apiAdapter.getKnowledgeMap();
    publishedCoverage.value = {
      total_entries: Number.isInteger(payload?.total_entries) ? payload.total_entries : 0,
      themes: Array.isArray(payload?.themes) ? payload.themes : []
    };
  } catch {
    coverageError.value = '无法加载已发布知识覆盖。';
  } finally {
    coverageLoading.value = false;
  }
};

const openSession = async (sessionId) => {
  sessionsLoading.value = true;
  try {
    await chatStore.loadSessionMessages(sessionId);
    resetQueryConditions();
    sessionVisible.value = false;
  } catch (error) {
    ElMessage.error(error.message || '加载会话消息失败');
  } finally {
    sessionsLoading.value = false;
  }
};

const removeSession = async (sessionId) => {
  const removedActiveSession = chatStore.activeSessionId === sessionId;
  try {
    await ElMessageBox.confirm('确认删除该会话？', '删除会话', {
      type: 'warning',
      confirmButtonText: '删除',
      cancelButtonText: '取消'
    });
    sessionsLoading.value = true;
    await chatStore.deleteSession(sessionId);
    if (removedActiveSession) resetQueryConditions();
    ElMessage.success('会话已删除');
  } catch (error) {
    if (error !== 'cancel') ElMessage.error(error.message || '删除会话失败');
  } finally {
    sessionsLoading.value = false;
  }
};

const toggleSessions = async () => {
  if (!authStore.isLoggedIn) {
    ElMessage.warning('请先登录');
    return;
  }
  sessionVisible.value = !sessionVisible.value;
  if (sessionVisible.value) {
    await loadSessions();
  }
};

const startNewSession = () => {
  chatStore.activeSessionId = '';
  chatStore.messages = [];
  resetQueryConditions();
  sessionVisible.value = false;
};

const openSourceExcerpt = async ({ source, trigger }) => {
  if (!source) return;
  selectedSource.value = source;
  sourceTrigger.value = trigger;
  await nextTick();
  excerptCloseRef.value?.focus();
};

const closeSourceExcerpt = async () => {
  const trigger = sourceTrigger.value;
  selectedSource.value = null;
  sourceTrigger.value = null;
  await nextTick();
  trigger?.focus();
};

const prefillCoverageQuestion = async (request) => {
  const question = typeof request === 'string' ? request : request?.question;
  if (typeof question !== 'string' || !question.trim()) return;
  if (request && typeof request === 'object' && Array.isArray(request.query_conditions)) {
    inheritConditions.value = false;
    queryConditions.value = request.query_conditions.map((condition) => ({
      condition_id:
        typeof condition?.condition_id === 'string' && condition.condition_id
          ? condition.condition_id
          : `condition-${Date.now().toString(36)}-${nextConditionNumber.value++}`,
      field: typeof condition?.field === 'string' ? condition.field : '',
      operator: typeof condition?.operator === 'string' ? condition.operator : '',
      value: typeof condition?.value === 'string' ? condition.value : ''
    }));
  }
  input.value = question.trim();
  composerError.value = '';
  await nextTick();
  composerInputRef.value?.focus();
};

const addQueryCondition = () => {
  queryConditions.value.push({
    condition_id: `condition-${Date.now().toString(36)}-${nextConditionNumber.value++}`,
    field: '',
    operator: 'equals',
    value: ''
  });
};

const removeQueryCondition = (index) => {
  queryConditions.value.splice(index, 1);
};

const resetQueryConditions = () => {
  queryConditions.value = [];
  inheritConditions.value = false;
};

const preparedQueryConditions = () => {
  const normalized = queryConditions.value.map((condition) => ({
    condition_id: condition.condition_id,
    field: condition.field.trim(),
    operator: condition.operator.trim(),
    value: condition.value.trim()
  }));
  if (normalized.some((condition) => !condition.field || !condition.operator || !condition.value)) {
    throw new Error('请完整填写每个查询条件。');
  }
  const keys = new Set();
  for (const condition of normalized) {
    const key = `${condition.field}\u0000${condition.operator}`;
    if (keys.has(key)) {
      throw new Error('同一字段与运算符只能保留一个查询条件。');
    }
    keys.add(key);
  }
  return normalized;
};

const onSend = async () => {
  if (!authStore.isLoggedIn) {
    ElMessage.warning('请先登录');
    return;
  }
  const question = input.value.trim();
  if (!question) {
    composerError.value = '请输入问题后再发送。';
    return;
  }
  let conditions;
  try {
    if (inheritConditions.value && !canInheritConditions.value) {
      throw new Error('当前会话没有可沿用的已完成查询条件。');
    }
    conditions = inheritConditions.value ? undefined : preparedQueryConditions();
  } catch (error) {
    composerError.value = error.message || '查询条件无效。';
    return;
  }
  composerError.value = '';
  input.value = '';
  await chatStore.sendMessage(question, {
    token: authStore.token,
    query_conditions: conditions?.length ? conditions : undefined,
    inherit_conditions: inheritConditions.value
  });
};

const retryAssistantMessage = async (messageIndex) => {
  if (chatStore.loading) return;
  try {
    const retry = resolveRetryTurn(chatStore.messages, messageIndex);
    if (retry === null) return;
    composerError.value = '';
    await chatStore.sendMessage(retry.question, {
      token: authStore.token,
      query_conditions: retry.query_conditions,
      inherit_conditions: retry.inherit_conditions
    });
  } catch (error) {
    composerError.value = error.message || '无法恢复已冻结的查询条件。';
  }
};

watch(
  () => chatStore.streamTick,
  () => {
    scrollToBottom();
  }
);

watch(
  () => chatStore.messages,
  () => {
    const selected = selectedSource.value;
    if (!selected) return;
    const identityKeys = ['citation_id', 'citation_identity', 'snapshot_id'];
    if (
      identityKeys.some(
        (key) => typeof selected[key] !== 'string' || !selected[key]
      )
    ) {
      selectedSource.value = null;
      sourceTrigger.value = null;
      return;
    }
    const currentSource = chatStore.messages
      .filter((message) => message?.role === 'assistant')
      .flatMap((message) =>
        Array.isArray(message?.evidence_summary?.sources) ? message.evidence_summary.sources : []
      )
      .find(
        (source) =>
          source &&
          identityKeys.every((key) => source[key] === selected[key])
      );
    if (!currentSource) {
      selectedSource.value = null;
      sourceTrigger.value = null;
      return;
    }
    selectedSource.value = currentSource;
  },
  { deep: true }
);

watch(canInheritConditions, (canInherit) => {
  if (!canInherit) inheritConditions.value = false;
});

onMounted(() => {
  if (typeof route.query.q === 'string' && route.query.q.trim()) {
    input.value = route.query.q.trim();
  }
  loadSessions();
  loadPublishedCoverage();
});
</script>

<style scoped>
.conversation-workspace {
  display: grid;
  grid-template-columns: 264px minmax(0, 820px);
  justify-content: center;
  width: min(100%, 1084px);
  min-height: calc(100vh - 92px);
  border: 1px solid var(--color-rule);
  background: var(--color-paper-raised);
}

.conversation-workspace__session-rail {
  min-height: 100%;
}

.conversation-workspace--excerpt-open {
  grid-template-columns: 264px minmax(0, 932px);
  width: min(100%, 1196px);
}

.conversation-workspace__content {
  min-width: 0;
  display: grid;
  grid-template-columns: minmax(0, 1fr);
}

.conversation-workspace--excerpt-open .conversation-workspace__content {
  grid-template-columns: minmax(0, 1fr) minmax(280px, 320px);
}

.chat-page {
  min-width: 0;
  padding: 32px clamp(24px, 4vw, 48px) 40px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.top-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.session-toggle {
  display: none;
}

.composer {
  margin-top: 12px;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.composer-actions {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

.query-conditions {
  display: grid;
  gap: 10px;
  padding-top: 12px;
  border-top: 1px solid var(--color-rule);
}

.query-conditions__header,
.query-conditions__actions {
  display: flex;
  align-items: center;
}

.query-conditions__header {
  justify-content: space-between;
  gap: 12px;
}

.query-conditions__header h2 {
  margin: 0;
  color: var(--color-ink);
  font-size: 13px;
  font-weight: 600;
  line-height: 1.5;
}

.query-conditions__actions {
  gap: 8px;
}

.query-conditions__actions :deep(.el-checkbox) {
  min-width: 0;
  margin-right: 4px;
}

.query-conditions__actions :deep(.el-checkbox__label) {
  padding-left: 6px;
  color: var(--color-ink-soft);
  font-size: 12px;
}

.query-conditions__rows {
  display: grid;
  gap: 8px;
}

.query-conditions__row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 0.78fr) minmax(0, 1fr) 32px;
  align-items: center;
  gap: 8px;
}

.query-conditions__row :deep(.el-button) {
  width: 32px;
  height: 32px;
  margin: 0;
}

.composer-feedback {
  margin: 0;
  color: var(--color-danger);
  font-size: 13px;
  line-height: 1.5;
}

.evidence-excerpt {
  min-width: 0;
  padding: 24px;
  border-left: 1px solid var(--color-rule);
  background: var(--color-paper-muted);
  animation: evidence-excerpt-enter 200ms ease-out both;
}

.evidence-excerpt__header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--color-rule);
}

.evidence-excerpt__header p,
.evidence-excerpt__header h2,
.evidence-excerpt__id,
.evidence-excerpt__metadata,
.evidence-excerpt__content {
  margin: 0;
}

.evidence-excerpt__metadata {
  display: grid;
  gap: 8px;
  margin-top: 16px;
  padding: 12px 0;
  border-top: 1px solid var(--color-rule);
  border-bottom: 1px solid var(--color-rule);
}

.evidence-excerpt__metadata div {
  display: grid;
  grid-template-columns: 72px minmax(0, 1fr);
  gap: 12px;
}

.evidence-excerpt__metadata dt,
.evidence-excerpt__metadata dd {
  margin: 0;
  overflow-wrap: anywhere;
  font-size: 12px;
  line-height: 1.6;
}

.evidence-excerpt__metadata dt {
  color: var(--color-ink-soft);
}

.evidence-excerpt__link {
  display: inline-flex;
  margin-top: 16px;
  color: var(--color-copper-strong);
  font-size: 13px;
  font-weight: 600;
  text-decoration: underline;
  text-underline-offset: 3px;
}

.evidence-excerpt__header p {
  color: var(--color-moss);
  font-size: 12px;
}

.evidence-excerpt__header h2 {
  margin-top: 4px;
  overflow-wrap: anywhere;
  font-family: var(--font-display);
  font-size: 18px;
  font-weight: 600;
  line-height: 1.4;
}

.evidence-excerpt__close {
  flex: 0 0 auto;
  min-height: 32px;
  padding: 4px 8px;
  border: 1px solid var(--line-strong);
  border-radius: var(--radius-control);
  background: transparent;
  color: var(--color-ink);
  cursor: pointer;
  font: inherit;
  font-size: 12px;
  transition: transform 180ms ease-out;
}

.evidence-excerpt__close:hover {
  border-color: var(--color-copper);
  background: var(--color-paper-raised);
  color: var(--color-copper-strong);
}

.evidence-excerpt__close:active {
  transform: translateY(1px);
}

.evidence-excerpt__id {
  margin-top: 16px;
  color: var(--color-ink-soft);
  font-family: var(--font-mono);
  font-size: 11px;
  line-height: 1.5;
  overflow-wrap: anywhere;
}

.evidence-excerpt__content {
  margin-top: 20px;
  color: var(--color-ink);
  font-size: 14px;
  line-height: 1.8;
  overflow-wrap: anywhere;
  white-space: pre-wrap;
}

@keyframes evidence-excerpt-enter {
  from {
    opacity: 0;
    transform: translateX(12px);
  }

  to {
    opacity: 1;
    transform: translateX(0);
  }
}

@media (prefers-reduced-motion: reduce) {
  .evidence-excerpt {
    animation: none;
  }
}

@media (max-width: 1180px) {
  .conversation-workspace {
    display: block;
    max-width: 820px;
    margin: 0 auto;
  }

  .conversation-workspace__session-rail {
    display: none;
  }

  .session-toggle {
    display: inline-flex;
  }

  .conversation-workspace--excerpt-open .conversation-workspace__content {
    grid-template-columns: minmax(0, 1fr);
  }
}

@media (max-width: 760px) {
  .evidence-excerpt {
    border-top: 1px solid var(--color-rule);
    border-left: 0;
  }
}

@media (max-width: 640px) {
  .conversation-workspace {
    min-height: auto;
    border-right: 0;
    border-left: 0;
  }

  .chat-page {
    padding: 20px 16px 28px;
  }

  .top-bar {
    align-items: flex-start;
  }

  .top-actions {
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 8px;
  }

  .composer {
    padding: 16px;
  }

  .query-conditions__header {
    align-items: flex-start;
  }

  .query-conditions__actions {
    flex-wrap: wrap;
    justify-content: flex-end;
  }

  .query-conditions__row {
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) 32px;
  }

  .query-conditions__row :deep(.el-input:nth-child(3)) {
    grid-column: 1 / span 2;
  }
}
</style>
