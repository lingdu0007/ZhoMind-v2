<template>
  <section ref="listRef" class="message-list" aria-label="对话内容">
    <div v-if="messages.length === 0" class="empty">
      <p class="empty__eyebrow">Published Knowledge</p>
      <h2>从团队知识开始提问</h2>
      <p v-if="coverageLoading" class="empty__state" role="status">正在加载已发布知识覆盖…</p>
      <p v-else-if="coverageError" class="empty__state empty__state--error" role="alert">{{ coverageError }}</p>
      <section
        v-else-if="coverageThemes.length"
        class="published-coverage"
        aria-label="已发布知识覆盖"
      >
        <p class="published-coverage__summary">{{ coverageTotal }} 条已发布工程决策可供查询。</p>
        <section v-for="theme in coverageThemes" :key="theme.domain" class="published-coverage__theme">
          <h3>{{ theme.label }}</h3>
          <ul>
            <li v-for="entry in theme.entries" :key="entry.entry_id">
              <p class="published-coverage__entry-title">{{ entry.title }}</p>
              <p class="published-coverage__entry-summary">{{ entry.approved_summary }}</p>
              <dl class="published-coverage__cues">
                <div><dt>知识版本</dt><dd>{{ entry.publication_version }}</dd></div>
                <div><dt>复核日期</dt><dd>{{ entry.review_date }}</dd></div>
                <div v-if="assuranceLabel(entry.assurance_level)"><dt>Assurance</dt><dd>{{ assuranceLabel(entry.assurance_level) }}</dd></div>
                <div><dt>来源</dt><dd>{{ coverageSourceCount(entry) }} 个来源</dd></div>
              </dl>
              <button
                type="button"
                class="published-coverage__question"
                :aria-label="entry.suggested_query"
                @click="$emit('ask-coverage', entry.suggested_query)"
              >{{ entry.suggested_query }}</button>
            </li>
          </ul>
        </section>
        <RouterLink class="published-coverage__map-link" :to="{ name: 'knowledge-map' }">浏览知识地图</RouterLink>
      </section>
      <p v-else class="empty__state">当前没有可浏览的已发布决策。</p>
    </div>
    <article
      v-for="(msg, idx) in messages"
      :key="messageKey(msg, idx)"
      class="msg"
      :class="`msg--${msg.role}`"
      :aria-label="msg.role === 'user' ? '用户消息' : '助手消息'"
    >
      <header class="msg-head">
        <span class="role" :class="msg.role">{{ msg.role === 'user' ? '你' : '助手' }}</span>
        <span v-if="msg.isThinking" class="status-dot" role="status">{{ msg.status || '正在检索与生成回答' }}</span>
        <span v-else-if="msg.streaming" class="status-dot" role="status">正在生成回答</span>
        <span
          v-else-if="msg.role === 'assistant' && (localTerminalFor(msg) || presentationFor(msg).kind !== 'unverified')"
          class="status-text"
          :class="`status-${localTerminalFor(msg)?.kind || presentationFor(msg).kind}`"
          role="status"
        >{{ localTerminalFor(msg)?.label || presentationFor(msg).label }}</span>
        <span v-else-if="msg.status" class="status-text" role="status">{{ msg.status }}</span>
      </header>
      <template v-if="msg.role === 'user'">
        <div class="content">{{ msg.content }}</div>
        <section
          v-if="msg.answer_execution?.query_condition_set"
          class="query-condition-set"
          aria-label="查询条件"
        >
          <div class="query-condition-set__heading">
            <span>查询条件</span>
            <strong>{{ conditionProvenanceLabel(msg.answer_execution.condition_provenance) }}</strong>
          </div>
          <ul v-if="queryConditionsFor(msg).length" class="query-condition-set__items">
            <li v-for="condition in queryConditionsFor(msg)" :key="condition.condition_id">
              <span>{{ condition.field }}</span>
              <span>{{ condition.operator }}</span>
              <span>{{ condition.value }}</span>
            </li>
          </ul>
          <p v-else class="query-condition-set__empty">无显式条件</p>
        </section>
        <section
          v-if="persistenceFailureFor(msg)"
          class="answer-outcome answer-outcome--terminal answer-outcome--failed"
          aria-label="持久化失败"
          role="alert"
        >
          <p class="answer-outcome__label">Failed</p>
          <p class="answer-outcome__detail">无法持久化回答；未形成回答或引用。</p>
        </section>
      </template>

      <template v-else-if="msg.isThinking || msg.streaming">
        <section class="answer-outcome answer-outcome--progress" aria-live="polite">
          <p class="answer-outcome__label">正在形成闭合执行记录</p>
          <p class="answer-outcome__detail">回答与引用会在冻结结果一致后显示。</p>
        </section>
      </template>

      <template v-else-if="localTerminalFor(msg)">
        <section
          class="answer-outcome answer-outcome--terminal"
          :class="`answer-outcome--${localTerminalFor(msg).kind}`"
          :aria-label="localTerminalFor(msg).ariaLabel"
        >
          <p class="answer-outcome__label">{{ localTerminalFor(msg).label }}</p>
          <p class="answer-outcome__detail">{{ localTerminalFor(msg).detail }}</p>
          <button
            v-if="localTerminalFor(msg).retryable"
            type="button"
            class="retry-button"
            :disabled="retryDisabled"
            @click="$emit('retry', idx)"
          >重试</button>
        </section>
      </template>

      <template v-else-if="msg.contract_error">
        <section class="answer-outcome answer-outcome--contract-failure" role="alert">
          <p class="answer-outcome__label">Closed result unavailable</p>
          <p class="answer-outcome__detail">{{ msg.contract_error }}</p>
        </section>
      </template>

      <template
        v-else-if="presentationFor(msg).kind === 'contract-failure' || presentationFor(msg).kind === 'unverified'"
      >
        <section class="answer-outcome answer-outcome--contract-failure" role="alert">
          <p class="answer-outcome__label">Closed result unavailable</p>
          <p class="answer-outcome__detail">该回答没有满足可展示的闭合执行与证据记录。</p>
        </section>
      </template>

      <template v-else-if="presentationFor(msg).kind === 'supported'">
        <section class="answer-outcome answer-outcome--supported" aria-label="已支持的知识回答">
          <p class="answer-outcome__label">Supported by published knowledge</p>
          <div v-if="frozenDecisionSections(msg).length" class="decision-sections">
            <section
              v-for="section in frozenDecisionSections(msg)"
              :key="section.id"
              class="decision-section"
              :aria-label="section.label || undefined"
            >
              <h3 v-if="section.label">{{ section.label }}</h3>
              <p v-for="(block, blockIndex) in section.blocks" :key="blockIndex">
                <template v-for="(fragment, fragmentIndex) in block" :key="fragmentIndex">
                  <span v-if="fragment.type === 'text'">{{ fragment.value }}</span>
                  <button
                    v-else
                    type="button"
                    class="citation-marker"
                    :aria-label="`打开引用 ${fragment.citationId}`"
                    @click="openCitation(msg, fragment.citationId, $event)"
                  >[{{ fragment.citationId }}]</button>
                </template>
              </p>
            </section>
          </div>

          <section class="evidence-summary evidence-summary--supported" aria-label="证据摘要">
            <div class="evidence-summary__heading">
              <span>Governing Engineering Decision Entry</span>
              <strong>{{ getEvidenceCoverageLabel(msg.evidence_summary?.coverage) }}</strong>
            </div>
            <button
              v-if="governingSourceFor(msg) && !governingSourceFor(msg).withdrawal_notice"
              type="button"
              class="evidence-summary__source evidence-summary__source--governing"
              :aria-label="`打开治理条目 ${governingSourceFor(msg).citation_id}`"
              @click="openSource(msg, governingSourceFor(msg), $event)"
            >
              <span>{{ governingSourceFor(msg).entry_title || getEvidenceSourceLabel(governingSourceFor(msg)) }}</span>
              <span>{{ sectionLabel(governingSourceFor(msg).section_id) }}</span>
            </button>
            <p v-else-if="governingSourceFor(msg)?.withdrawal_notice" class="evidence-summary__withdrawn">
              Governing Engineering Decision Entry：{{ governingSourceFor(msg).withdrawal_notice }}
            </p>
            <ul v-if="citationSourcesFor(msg).length" class="evidence-summary__sources">
              <li v-for="source in citationSourcesFor(msg)" :key="source.citation_id">
                <p v-if="source.withdrawal_notice" class="evidence-summary__withdrawn">
                  {{ getEvidenceSourceLabel(source) }}：{{ source.withdrawal_notice }}
                </p>
                <button
                  v-else
                  type="button"
                  class="evidence-summary__source"
                  :aria-label="`打开引用 ${source.citation_id}`"
                  @click="openSource(msg, source, $event)"
                >
                  <span>[{{ source.citation_id }}] {{ getEvidenceSourceLabel(source) }}</span>
                  <span>{{ sectionLabel(source.section_id) }}</span>
                </button>
              </li>
            </ul>
            <details class="frozen-answer-record">
              <summary>冻结回答记录</summary>
              <dl>
                <div>
                  <dt>Evidence Set</dt>
                  <dd>{{ msg.answer_execution?.evidence_set_identity }}</dd>
                </div>
                <div>
                  <dt>Knowledge Versions</dt>
                  <dd>{{ identityList(msg.answer_execution?.knowledge_version_identities) }}</dd>
                </div>
                <div>
                  <dt>Snapshots</dt>
                  <dd>{{ identityList(msg.answer_execution?.snapshot_ids) }}</dd>
                </div>
              </dl>
            </details>
          </section>
          <KnowledgeFeedback v-if="!msg.streaming && msg.id" :message="msg" />
        </section>
      </template>

      <template v-else-if="presentationFor(msg).kind === 'insufficient'">
        <section class="answer-outcome answer-outcome--insufficient" aria-label="证据不足回复">
          <p class="answer-outcome__label">Insufficient Evidence Reply</p>
          <div class="content">{{ displayedAnswerText(msg) }}</div>
          <section v-if="insufficiencyFor(msg)" class="insufficient-detail">
            <h3>{{ insufficiencyFor(msg).title }}</h3>
            <p>{{ insufficiencyFor(msg).detail }}</p>
            <div class="insufficient-detail__actions">
              <button type="button" @click="refineQuestion(msg)">细化问题</button>
            </div>
          </section>
          <section v-if="coverageEntriesFor(msg).length" class="non-supporting-coverage" aria-label="非支持性已发布覆盖">
            <h3>已发布覆盖不构成此回答的支持</h3>
            <ul>
              <li v-for="entry in coverageEntriesFor(msg)" :key="entry.entry_id">
                <span>{{ entry.title }}</span>
                <button type="button" @click="$emit('ask-coverage', entry.suggested_query)">{{ entry.suggested_query }}</button>
              </li>
            </ul>
          </section>
          <KnowledgeFeedback v-if="isGapReportFor(msg)" mode="gap" :message="msg" />
        </section>
      </template>

      <template v-else-if="presentationFor(msg).kind === 'non-knowledge-base'">
        <section class="answer-outcome answer-outcome--non-knowledge-base" aria-label="非知识库回复">
          <p class="answer-outcome__label">Non-Knowledge-Base Reply</p>
          <div class="content">{{ displayedAnswerText(msg) }}</div>
        </section>
      </template>

      <template v-else-if="presentationFor(msg).kind === 'generation-unavailable'">
        <section class="answer-outcome answer-outcome--generation-unavailable" aria-label="生成不可用">
          <p class="answer-outcome__label">Generation Unavailable</p>
          <p class="answer-outcome__detail">已关闭该请求，但没有形成可展示的知识回答、引用或证据预览。</p>
          <button type="button" class="retry-button" :disabled="retryDisabled" @click="$emit('retry', idx)">重试</button>
        </section>
      </template>

      <template v-else-if="terminalDetailFor(msg)">
        <section class="answer-outcome answer-outcome--terminal" :class="`answer-outcome--${presentationFor(msg).kind}`">
          <p class="answer-outcome__label">{{ presentationFor(msg).label }}</p>
          <p class="answer-outcome__detail">{{ terminalDetailFor(msg) }}</p>
          <button
            v-if="presentationFor(msg).retryable"
            type="button"
            class="retry-button"
            :disabled="retryDisabled"
            @click="$emit('retry', idx)"
          >重试</button>
        </section>
      </template>

      <template v-else>
        <div class="content">{{ msg.content }}</div>
      </template>

      <RetrievalDiagnostics
        v-if="showDiagnostics && msg.role === 'assistant' && msg.retrieval_diagnostics && !msg.streaming && !msg.contract_error"
        :diagnostics="msg.retrieval_diagnostics"
      />
    </article>
  </section>
</template>

<script setup>
import { computed, nextTick, ref, watch } from 'vue';
import {
  getEvidenceCoverageLabel,
  getEvidenceSourceLabel,
  getKnowledgeAssuranceLabel
} from '../app/evidence-summary';
import {
  getAnswerExecutionPresentation,
  getInsufficientEvidencePresentation,
  isRenderableFrozenSupportedAnswer,
  parseFrozenDecisionAnswer
} from '../app/answer-execution-presentation';
import KnowledgeFeedback from './KnowledgeFeedback.vue';
import RetrievalDiagnostics from './RetrievalDiagnostics.vue';

const listRef = ref(null);

const messageKey = (message, index) => {
  const identity =
    typeof message?.id === 'string' && message.id
      ? message.id
      : typeof message?.local_stream_id === 'string' && message.local_stream_id
        ? message.local_stream_id
        : String(index);
  return `${message?.role || 'message'}:${identity}`;
};

const queryConditionsFor = (message) => {
  const conditions = message?.answer_execution?.query_condition_set?.conditions;
  return Array.isArray(conditions) ? conditions : [];
};

const conditionProvenanceLabel = (provenance) => (provenance?.mode === 'inherited' ? '继承' : '本轮');
const coverageSourceCount = (entry) =>
  Number.isInteger(entry?.source_count) ? entry.source_count : entry?.public_source_count || 0;

const terminalStateDetails = {
  stopped: '该请求已停止，未形成完整回答。',
  failed: '执行失败，未形成可支持或可引用的回答。',
  throttled: '请求当前受限，尚未形成回答。',
  rejected: '请求未获接纳，尚未形成回答。'
};

const sectionLabels = {
  recommendation_or_reviewed_branches: 'Recommendation or reviewed branches',
  applicability: 'Applicability',
  alternatives: 'Alternatives',
  minimum_implementation_guidance: 'Minimum implementation guidance',
  minimum_acceptance_guidance: 'Minimum acceptance guidance'
};

const props = defineProps({
  messages: {
    type: Array,
    default: () => []
  },
  coverage: {
    type: Object,
    default: null
  },
  coverageLoading: {
    type: Boolean,
    default: false
  },
  coverageError: {
    type: String,
    default: ''
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

const emit = defineEmits(['ask-coverage', 'refine', 'retry', 'open-source']);

const coverageThemes = computed(() => {
  const themes = props.coverage?.themes;
  if (!Array.isArray(themes)) return [];
  return themes.filter(
    (theme) =>
      theme &&
      typeof theme.domain === 'string' &&
      typeof theme.label === 'string' &&
      Array.isArray(theme.entries) &&
      theme.entries.length
  );
});

const coverageTotal = computed(() =>
  Number.isInteger(props.coverage?.total_entries) ? props.coverage.total_entries : 0
);

const coverageEntries = computed(() =>
  coverageThemes.value
    .flatMap((theme) => theme.entries)
    .filter(
      (entry) =>
        entry &&
        typeof entry.entry_id === 'string' &&
        typeof entry.title === 'string' &&
        typeof entry.suggested_query === 'string' &&
        entry.suggested_query
    )
);

const coverageTerms = (value) =>
  typeof value === 'string'
    ? (() => {
        const normalized = value.toLocaleLowerCase();
        const words = normalized
          .split(/[^\p{L}\p{N}_-]+/u)
          .filter((term) => term.length >= 2 && !/^\p{Script=Han}+$/u.test(term));
        const hanBigrams = Array.from(normalized.matchAll(/\p{Script=Han}+/gu)).flatMap((match) => {
          const characters = Array.from(match[0]);
          return characters.slice(0, -1).map((character, index) => character + characters[index + 1]);
        });
        return Array.from(new Set([...words, ...hanBigrams]));
      })()
    : [];

const coverageEntriesFor = (message) => {
  const entries = coverageEntries.value;
  const terms = coverageTerms(message?.answer_execution?.question);
  if (!terms.length) return entries.slice(0, 3);
  const related = entries
    .map((entry, index) => {
      const searchable = `${entry.title} ${entry.suggested_query} ${entry.approved_summary || ''}`.toLocaleLowerCase();
      return {
        entry,
        index,
        score: terms.reduce((total, term) => total + Number(searchable.includes(term)), 0)
      };
    })
    .filter(({ score }) => score > 0)
    .sort((left, right) => right.score - left.score || left.index - right.index)
    .map(({ entry }) => entry);
  return (related.length ? related : entries).slice(0, 3);
};

const presentationFor = (message) => {
  const presentation = getAnswerExecutionPresentation(message?.answer_execution);
  if (
    presentation.kind === 'supported' &&
    !isRenderableFrozenSupportedAnswer(message?.content, citationSourcesFor(message))
  ) {
    return {
      kind: 'contract-failure',
      label: 'Closed result unavailable',
      retryable: false
    };
  }
  return presentation;
};

const citationSourcesFor = (message) => {
  const sources = message?.evidence_summary?.sources;
  return Array.isArray(sources)
    ? sources.filter((source) => source && typeof source.citation_id === 'string' && source.citation_id)
    : [];
};

const sourceForCitation = (message, citationId) =>
  citationSourcesFor(message).find((source) => source.citation_id === citationId) || null;

const governingSourceFor = (message) =>
  citationSourcesFor(message).find(
    (source) => source.section_id === 'recommendation_or_reviewed_branches'
  ) || null;

const frozenDecisionSections = (message) =>
  parseFrozenDecisionAnswer(
    message?.content,
    citationSourcesFor(message).map((source) => source.citation_id)
  );

const insufficiencyFor = (message) =>
  getInsufficientEvidencePresentation(message?.insufficient_evidence_reply?.reason);

const terminalDetailFor = (message) => terminalStateDetails[presentationFor(message).kind] || '';

const localTerminalFor = (message) => {
  const state = message?.local_terminal_state;
  if (!['stopped', 'failed', 'throttled', 'rejected'].includes(state)) return null;
  return {
    kind: state,
    label: {
      stopped: 'Stopped',
      failed: 'Failed',
      throttled: 'Throttled',
      rejected: 'Rejected'
    }[state],
    detail:
      state === 'stopped'
        ? '本地已停止请求，尚未确认服务端是否接纳该轮执行。'
        : state === 'throttled'
          ? '请求在服务端接纳前已被限流，未形成回答或引用。'
          : state === 'rejected'
            ? '请求在服务端接纳前被拒绝，未形成回答或引用。'
            : '请求在服务端确认接纳前失败，未形成回答或引用。',
    ariaLabel: {
      stopped: '本地停止',
      failed: '本地执行失败',
      throttled: '本地限流',
      rejected: '本地拒绝'
    }[state],
    retryable: message.local_retryable === true
  };
};

const sectionLabel = (sectionId) => sectionLabels[sectionId] || sectionId || 'Published knowledge section';

const assuranceLabel = (value) => getKnowledgeAssuranceLabel(value);

const identityList = (identities) =>
  Array.isArray(identities) && identities.length ? identities.join('\n') : 'None';

const displayedAnswerText = (message) => {
  const text = typeof message?.content === 'string' ? message.content : '';
  const presentation = presentationFor(message);
  const heading = `【${presentation.label}】`;
  return text.startsWith(heading) ? text.slice(heading.length).trimStart() : text;
};

const openSource = (message, source, event) => {
  if (!source) return;
  emit('open-source', { source, trigger: event.currentTarget, message });
};

const openCitation = (message, citationId, event) => {
  openSource(message, sourceForCitation(message, citationId), event);
};

const refineQuestion = (message) => {
  const question = message?.answer_execution?.question;
  if (typeof question !== 'string' || !question.trim()) return;
  emit('refine', {
    question,
    query_conditions: queryConditionsFor(message)
  });
};

const isGapReportFor = (message) =>
  presentationFor(message).kind === 'insufficient' &&
  message?.answer_execution?.state === 'completed' &&
  message?.answer_execution?.outcome === 'insufficient_evidence_reply' &&
  message?.answer_execution?.assistant_message_id === message?.id;

const persistenceFailureFor = (message) => {
  const execution = message?.answer_execution;
  return (
    execution?.state === 'failed' &&
    execution?.failure_code === 'ANSWER_EXECUTION_PERSISTENCE_FAILED' &&
    (execution.assistant_message_id === undefined || execution.assistant_message_id === null)
  );
};

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

.empty__state {
  max-width: 520px;
  margin: 14px auto 0;
  color: var(--color-ink-soft);
  font-size: 14px;
  line-height: 1.75;
}

.empty__state--error {
  color: var(--color-danger);
}

.published-coverage {
  width: min(100%, 680px);
  margin: 20px auto 0;
  border-top: 1px solid var(--color-rule);
  text-align: left;
}

.published-coverage__summary {
  margin: 14px 0 0;
  color: var(--color-ink-soft);
  font-size: 13px;
  line-height: 1.65;
}

.published-coverage__theme {
  padding-top: 16px;
}

.published-coverage__theme h3 {
  margin: 0;
  color: var(--color-moss);
  font-family: var(--font-display);
  font-size: 16px;
  font-weight: 600;
  line-height: 1.45;
}

.published-coverage__theme ul {
  display: grid;
  gap: 14px;
  margin: 10px 0 0;
  padding: 0;
  list-style: none;
}

.published-coverage__theme li {
  padding-top: 14px;
  border-top: 1px solid var(--color-rule);
}

.published-coverage__entry-title,
.published-coverage__entry-summary {
  margin: 0;
}

.published-coverage__entry-title {
  color: var(--color-ink);
  font-size: 14px;
  font-weight: 600;
  line-height: 1.55;
}

.published-coverage__entry-summary {
  margin-top: 4px;
  color: var(--color-ink-soft);
  font-size: 13px;
  line-height: 1.65;
}

.published-coverage__cues {
  display: flex;
  flex-wrap: wrap;
  gap: 5px 12px;
  margin: 8px 0 0;
}

.published-coverage__cues div {
  display: inline-flex;
  gap: 4px;
  min-width: 0;
}

.published-coverage__cues dt,
.published-coverage__cues dd {
  margin: 0;
  color: var(--color-ink-soft);
  font-size: 11px;
  line-height: 1.5;
}

.published-coverage__cues dd {
  color: var(--color-ink);
}

.published-coverage__question,
.published-coverage__map-link {
  display: inline-flex;
  max-width: 100%;
  margin-top: 10px;
  padding: 0;
  border: 0;
  background: transparent;
  color: var(--color-copper-strong);
  cursor: pointer;
  font: inherit;
  font-size: 13px;
  font-weight: 600;
  line-height: 1.55;
  overflow-wrap: anywhere;
  text-align: left;
  text-decoration: underline;
  text-underline-offset: 3px;
}

.published-coverage__map-link {
  margin-top: 18px;
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

.status-supported {
  color: var(--color-moss);
}

.status-insufficient,
.status-throttled,
.status-rejected {
  color: var(--color-warning);
}

.status-generation-unavailable,
.status-contract-failure,
.status-failed {
  color: var(--color-danger);
}

.status-stopped {
  color: var(--color-ink-soft);
}

.answer-outcome {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  min-width: 0;
  gap: 10px;
  padding: 2px 0 2px 16px;
  border-left: 3px solid var(--color-rule);
}

.answer-outcome--supported {
  border-left-color: var(--color-moss);
}

.answer-outcome--insufficient,
.answer-outcome--throttled,
.answer-outcome--rejected {
  border-left-color: var(--color-warning);
}

.answer-outcome--generation-unavailable,
.answer-outcome--failed,
.answer-outcome--contract-failure {
  border-left-color: var(--color-danger);
}

.answer-outcome__label,
.answer-outcome__detail {
  margin: 0;
}

.answer-outcome__label {
  color: var(--color-ink);
  font-size: 13px;
  font-weight: 600;
  line-height: 1.5;
}

.answer-outcome__detail {
  color: var(--color-ink-soft);
  font-size: 13px;
  line-height: 1.65;
}

.decision-sections {
  display: grid;
  gap: 16px;
}

.decision-section {
  padding-top: 14px;
  border-top: 1px solid var(--color-rule);
}

.decision-section h3,
.decision-section p {
  margin: 0;
}

.decision-section h3 {
  color: var(--color-ink);
  font-family: var(--font-display);
  font-size: 16px;
  font-weight: 600;
  line-height: 1.45;
}

.decision-section p {
  margin-top: 7px;
  color: var(--color-ink);
  font-size: 15px;
  line-height: 1.8;
  overflow-wrap: anywhere;
}

.citation-marker {
  display: inline;
  min-width: 0;
  padding: 0;
  border: 0;
  background: transparent;
  color: var(--color-copper-strong);
  cursor: pointer;
  font: inherit;
  font-family: var(--font-mono);
  font-size: 0.88em;
  font-weight: 600;
  line-height: inherit;
  text-decoration: underline;
  text-underline-offset: 2px;
}

.citation-marker:hover {
  color: var(--color-moss);
}

.insufficient-detail,
.non-supporting-coverage {
  padding-top: 12px;
  border-top: 1px solid var(--color-rule);
}

.insufficient-detail h3,
.non-supporting-coverage h3,
.insufficient-detail p,
.non-supporting-coverage ul {
  margin: 0;
}

.insufficient-detail h3,
.non-supporting-coverage h3 {
  color: var(--color-ink);
  font-size: 14px;
  font-weight: 600;
  line-height: 1.55;
}

.insufficient-detail p {
  margin-top: 6px;
  color: var(--color-ink-soft);
  font-size: 13px;
  line-height: 1.7;
}

.insufficient-detail__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}

.insufficient-detail__actions button,
.non-supporting-coverage button {
  min-height: 32px;
  padding: 4px 10px;
  border: 1px solid var(--color-rule);
  border-radius: var(--radius-control);
  background: var(--color-paper-raised);
  color: var(--color-ink);
  cursor: pointer;
  font: inherit;
  font-size: 12px;
  line-height: 1.45;
}

.insufficient-detail__actions button:hover,
.non-supporting-coverage button:hover {
  border-color: var(--color-copper);
  color: var(--color-copper-strong);
}

.non-supporting-coverage ul {
  display: grid;
  gap: 8px;
  margin-top: 8px;
  padding: 0;
  list-style: none;
}

.non-supporting-coverage li {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding-top: 8px;
  border-top: 1px solid var(--color-rule);
  color: var(--color-ink-soft);
  font-size: 13px;
  line-height: 1.55;
}

.non-supporting-coverage li span {
  min-width: 0;
  overflow-wrap: anywhere;
}

.non-supporting-coverage button {
  flex: 0 0 auto;
  max-width: 55%;
  overflow-wrap: anywhere;
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

.query-condition-set {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--color-rule);
}

.query-condition-set__heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  color: var(--color-ink-soft);
  font-size: 12px;
}

.query-condition-set__heading strong {
  color: var(--color-moss);
  font-size: 12px;
  font-weight: 600;
}

.query-condition-set__items {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin: 8px 0 0;
  padding: 0;
  list-style: none;
}

.query-condition-set__items li {
  display: inline-flex;
  min-width: 0;
  align-items: center;
  gap: 4px;
  padding: 4px 7px;
  border: 1px solid var(--color-rule);
  background: var(--color-paper-raised);
  color: var(--color-ink);
  font-family: var(--font-mono);
  font-size: 11px;
  line-height: 1.4;
  overflow-wrap: anywhere;
}

.query-condition-set__items li span:nth-child(2) {
  color: var(--color-ink-soft);
}

.query-condition-set__empty {
  margin: 8px 0 0;
  color: var(--color-ink-soft);
  font-size: 12px;
  line-height: 1.5;
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

.evidence-summary__withdrawn {
  margin: 0;
  padding: 8px 0;
  border-top: 1px solid color-mix(in srgb, var(--color-moss) 28%, transparent);
  color: var(--color-ink-soft);
  font-size: 13px;
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

.evidence-summary__source--governing {
  border-top: 0;
  font-weight: 600;
}

.frozen-answer-record {
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid color-mix(in srgb, var(--color-moss) 28%, transparent);
}

.frozen-answer-record summary {
  color: var(--color-ink-soft);
  cursor: pointer;
  font-size: 12px;
  line-height: 1.5;
}

.frozen-answer-record dl {
  display: grid;
  gap: 8px;
  margin: 10px 0 0;
}

.frozen-answer-record dl div {
  display: grid;
  grid-template-columns: 120px minmax(0, 1fr);
  gap: 8px;
}

.frozen-answer-record dt,
.frozen-answer-record dd {
  margin: 0;
  overflow-wrap: anywhere;
  font-family: var(--font-mono);
  font-size: 11px;
  line-height: 1.55;
}

.frozen-answer-record dt {
  color: var(--color-ink-soft);
}

.frozen-answer-record dd {
  color: var(--color-ink);
  white-space: pre-line;
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

  .answer-outcome {
    padding-left: 12px;
  }

  .non-supporting-coverage li {
    align-items: flex-start;
    flex-direction: column;
  }

  .non-supporting-coverage button {
    max-width: 100%;
  }

  .frozen-answer-record dl div {
    grid-template-columns: 1fr;
    gap: 2px;
  }
}
</style>
