<template>
  <section class="maintenance-evidence" aria-label="维护证据" :aria-busy="loading || busy">
    <p v-if="loading" role="status">正在加载维护证据…</p>
    <p v-if="error" role="alert" class="maintenance-evidence__error">{{ error }}</p>
    <p v-else-if="notice" role="status">{{ notice }}</p>

    <MaintenanceProviderVerification v-if="highSeverity && openWork && item.classification === 'product-privacy-operations'
      && (isWorker || context.is_administrator)" :item="item" :context="context" :disabled="unavailable"
      @authorization="providerAuthorization = $event" @busy="emit('busy', $event)" />

    <MaintenanceFreshness v-if="isWorker && openWork && item.classification === 'source-freshness'
      && item.severity === 'p2' && inputs.targets.some((target) => target.entry_id)"
      :key="item.id" :targets="inputs.targets" :disabled="unavailable" @busy="emit('busy', $event)" />
    <MaintenanceIntegrity v-if="isWorker && openWork && item.classification === 'content-integrity'
      && inputs.targets.some((target) => target.entry_id)"
      :key="item.id" :targets="inputs.targets" :disabled="unavailable" @busy="emit('busy', $event)" />

    <form v-if="isWorker && openWork" aria-label="独立复现" @submit.prevent="reproduce">
      <h3>独立复现</h3>
      <label>已保存的本人执行<select v-model="draft.answer_id" aria-label="已保存的本人执行" required :disabled="unavailable">
        <option value="">选择执行</option>
        <option v-for="answer in inputs.answers" :key="answer.answer_id" :value="answer.answer_id">
          {{ answer.session_id }} · {{ outcomeLabel(answer.outcome) }}
        </option>
      </select></label>
      <label>反馈目标<select v-model="draft.signal_id" aria-label="反馈目标" required :disabled="unavailable">
        <option value="">选择反馈目标</option>
        <option v-for="target in inputs.targets" :key="target.signal_id" :value="target.signal_id">
          {{ target.entry_id || '知识缺口' }} · {{ labelNames[target.label] || target.label }} · {{ target.signal_id.slice(0, 8) }}
        </option>
      </select></label>
      <label>预期结果<select v-model="draft.expected_outcome" aria-label="预期结果" required :disabled="unavailable">
        <option v-for="[value, label] in outcomes" :key="value" :value="value">{{ label }}</option>
      </select></label>
      <label>验证观察<select v-model="draft.verified_observation" aria-label="验证观察" :disabled="unavailable">
        <option v-for="[value, label] in observations" :key="value" :value="value">{{ label }}</option>
      </select></label>
      <label v-if="needsReference">参照执行<select v-model="draft.reference_answer_id" aria-label="参照执行" required :disabled="unavailable">
        <option value="">选择已获证据支持的本人执行</option>
        <option v-for="answer in supportedAnswers" :key="answer.answer_id" :value="answer.answer_id">{{ answer.session_id }}</option>
      </select></label>
      <label v-if="needsEntry">条目标识<input v-model.trim="draft.entry_identity" required maxlength="166" :disabled="unavailable" /></label>
      <label class="maintenance-evidence__consent">
        <input v-model="draft.confirmed_synthetic_fixture" type="checkbox" :disabled="unavailable" />
        确认这是独立编写、可共享的非个人场景
      </label>
      <div class="maintenance-evidence__actions">
        <RouterLink to="/chat">我的对话</RouterLink>
        <button type="submit" :disabled="unavailable || !draft.confirmed_synthetic_fixture || !draft.answer_id || !draft.signal_id
          || (requiresReproductionAuthorization && !reproductionAuthorizationReady)">
          <FlaskConical :size="16" aria-hidden="true" />登记并运行复现
        </button>
      </div>
    </form>

    <section v-if="fixture" aria-label="认证复现结果">
      <h3>认证复现</h3>
      <dl>
        <dt>场景</dt><dd>{{ fixture.id }}</dd>
        <dt>查询条件集</dt><dd>{{ fixture.query_condition_set_identity }}</dd>
        <dt>预期结果</dt><dd>{{ outcomeLabel(fixture.expected_outcome) }}</dd>
        <dt>观察结果</dt><dd>{{ outcomeLabel(fixture.observed_outcome) }}</dd>
        <dt>执行状态</dt><dd>{{ fixture.observed_state === 'completed' ? '已完成' : '执行失败' }}</dd>
        <dt>验证观察</dt><dd>{{ observationLabel(fixture.verified_observation) }}</dd>
        <dt>检索配置</dt><dd>{{ fixture.retrieval_profile_identity }}</dd>
        <dt>复现时发布版本</dt><dd>{{ fixture.active_publication_identities.join(' · ') || '无可用发布版本' }}</dd>
        <template v-if="fixture.publication_review">
          <dt>涉及发布版本</dt><dd>{{ fixture.publication_review.publication_identity }}</dd>
          <dt>复现时来源</dt><dd>
            <ul><li v-for="source in fixture.publication_review.source_facts" :key="source.source_identity">
              <span>{{ source.source_identity }}</span>
              <span>{{ sourceStates[source.availability] }}</span>
            </li></ul>
          </dd>
        </template>
      </dl>
      <div v-if="isMaintainer && openWork" class="maintenance-evidence__actions">
        <button v-if="item.diagnosis_fixture_identity !== fixture.id" type="button" :disabled="unavailable" @click="diagnose">
          <ListChecks :size="16" aria-hidden="true" />确认诊断
        </button>
        <button v-if="item.diagnosis_fixture_identity === fixture.id && !approved && !isWorker" type="button" :disabled="unavailable" @click="approveFinding">
          <ShieldCheck :size="16" aria-hidden="true" />批准非个人发现
        </button>
        <button v-if="item.state === 'triaged' && item.diagnosis_fixture_identity === fixture.id" type="button" :disabled="unavailable" @click="transition('in_progress')">
          <Play :size="16" aria-hidden="true" />开始处理
        </button>
      </div>
    </section>

    <section v-if="findings.length" aria-label="非个人发现">
      <h3>非个人发现</h3>
      <ul>
        <li v-for="finding in findings" :key="finding.id">
          <strong>{{ observationLabel(finding.pattern.observation) }}</strong>
          <span>{{ finding.id }}</span>
          <span>{{ outcomeLabel(finding.observed_outcome) }}</span>
        </li>
      </ul>
    </section>

    <section v-if="replay || (approvedFixtures.length && item.classification !== 'confirmation')" aria-label="修复后验证">
      <h3>修复后验证</h3>
      <fieldset v-if="isMaintainer && openWork && approvedFixtures.length > 1">
        <legend>本次关闭验证场景</legend>
        <label v-for="scenario in approvedFixtures" :key="scenario.id" class="maintenance-evidence__consent">
          <input v-model="resolutionFixtureIdentities" type="checkbox" :value="scenario.id"
            :disabled="unavailable || !scenarioReplays[scenario.id]?.passed" />
          <span>{{ scenario.id }}<small>{{ scenarioReplays[scenario.id]?.passed ? '重放通过' : '尚无通过的最新重放' }}</small>
            <small v-for="target in scenario.affected_scope.entry_versions" :key="target.publication_identity">{{ target.entry_identity }}</small>
            <small v-for="target in scenario.affected_scope.gap_contexts" :key="target.query_condition_set_identity">{{ target.query_condition_set_identity }}</small>
          </span>
        </label>
      </fieldset>
      <label v-if="approvedFixtures.length">已批准场景<select v-model="repairFixtureIdentity" aria-label="已批准场景" :disabled="unavailable">
        <option v-for="scenario in approvedFixtures" :key="scenario.id" :value="scenario.id">
          {{ scenario.id.slice(-8) }} · {{ outcomeLabel(scenario.expected_outcome) }}
        </option>
      </select></label>
      <dl v-if="repairFixture">
        <dt>场景标识</dt><dd>{{ repairFixture.id }}</dd>
        <dt>查询条件集</dt><dd>{{ repairFixture.query_condition_set_identity }}</dd>
        <dt>声明预期</dt><dd>{{ outcomeLabel(repairFixture.expected_outcome) }}</dd>
      </dl>
      <label v-if="isWorker && repairFixture && item.state === 'in_progress'">重放使用的本人执行
        <select v-model="replayAnswerIdentity" aria-label="重放使用的本人执行" :disabled="unavailable">
          <option value="">请选择</option>
          <option v-for="answer in replayAnswers" :key="answer.answer_id" :value="answer.answer_id">
            {{ answer.session_id }}
          </option>
        </select>
      </label>
      <dl v-if="replay">
        <dt>重放场景</dt><dd>{{ replay.fixture_identity }}</dd>
        <dt>重放结果</dt><dd>{{ replay.passed ? '重放通过' : '重放未通过' }}</dd>
        <dt>预期结果</dt><dd>{{ outcomeLabel(replay.expected_outcome) }}</dd>
        <dt>观察结果</dt><dd>{{ outcomeLabel(replay.observed_outcome) }}</dd>
        <dt>执行状态</dt><dd>{{ replay.observed_state === 'completed' ? '已完成' : '执行失败' }}</dd>
        <dt>重放记录</dt><dd>{{ replay.id }}</dd>
        <template v-if="replay.publication_review">
          <dt>验证修订</dt><dd>{{ replay.publication_review.revision_identity }}</dd>
          <dt>验证发布版本</dt><dd>{{ replay.publication_review.publication_identity }}</dd>
        </template>
        <template v-else>
          <template v-for="publication in replay.evidence_publications" :key="publication.publication_identity">
            <dt>验证修订</dt><dd>{{ publication.revision_identity }}</dd>
            <dt>验证发布版本</dt><dd>{{ publication.publication_identity }}</dd>
          </template>
        </template>
        <template v-if="replay.generation_context">
          <dt>重放路由</dt><dd>{{ replay.generation_context.route_identity }}</dd>
          <dt>{{ replay.generation_context.authorization_identity ? '调用准入' : '路由验收' }}</dt>
          <dd>{{ replay.generation_context.acceptance_record_identity }}</dd>
          <template v-if="replay.generation_context.authorization_identity">
            <dt>验证授权</dt><dd>{{ replay.generation_context.authorization_identity }}</dd>
          </template>
        </template>
      </dl>
      <label v-if="isMaintainer && requiresReacceptance"
        class="maintenance-evidence__reacceptance">重新验收记录标识
        <input v-model.trim="reacceptanceIdentity" aria-label="重新验收记录标识"
          maxlength="190" :disabled="unavailable" />
      </label>
      <div class="maintenance-evidence__actions">
        <button v-if="isWorker && repairFixture && item.state === 'in_progress'" type="button"
          :disabled="unavailable || !replayInputReady || (requiresReplayAuthorization && !replayAuthorizationReady)" @click="runReplay">
          <RotateCw :size="16" aria-hidden="true" />重放已声明场景
        </button>
        <button v-if="isMaintainer && canResolveBoundary" type="button" :disabled="unavailable" @click="resolveBoundary">
          <CheckCheck :size="16" aria-hidden="true" />以边界查询解决
        </button>
        <button v-if="isMaintainer && canResolveSource" type="button" :disabled="unavailable" @click="resolveSource">
          <CheckCheck :size="16" aria-hidden="true" />确认来源修复
        </button>
        <button v-if="isMaintainer && canResolveProvider" type="button" :disabled="unavailable" @click="resolveProvider">
          <CheckCheck :size="16" aria-hidden="true" />确认 Provider 恢复
        </button>
        <button v-if="isMaintainer && canResolveRetrieval" type="button" :disabled="unavailable" @click="resolveRetrieval">
          <FlaskConical :size="16" aria-hidden="true" />确认检索实验结果
        </button>
        <button v-if="isMaintainer && canResolveProduct" type="button" :disabled="unavailable" @click="resolveProduct">
          <CheckCheck :size="16" aria-hidden="true" />确认产品修复
        </button>
        <button v-if="isMaintainer && canResolveContent" type="button" :disabled="unavailable" @click="resolveContent">
          <CheckCheck :size="16" aria-hidden="true" />确认内容修复
        </button>
        <button v-if="isMaintainer && canResolveCoverage" type="button" :disabled="unavailable" @click="resolveCoverage">
          <CheckCheck :size="16" aria-hidden="true" />确认覆盖补齐
        </button>
      </div>
    </section>
    <button v-if="isMaintainer && (item.state === 'resolved' || canCloseConfirmation)" type="button" :disabled="unavailable" @click="transition('closed_confirmation')">
      <CheckCheck :size="16" aria-hidden="true" />关闭确认
    </button>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, reactive, ref, watch } from 'vue';
import { RouterLink } from 'vue-router';
import { CheckCheck, FlaskConical, ListChecks, Play, RotateCw, ShieldCheck } from 'lucide-vue-next';
import { maintenanceApi } from '../api/maintenance';
import MaintenanceFreshness from './MaintenanceFreshness.vue';
import MaintenanceIntegrity from './MaintenanceIntegrity.vue';
import MaintenanceProviderVerification from './MaintenanceProviderVerification.vue';

const props = defineProps({ item: { type: Object, required: true }, context: { type: Object, required: true }, disabled: Boolean });
const emit = defineEmits(['refresh', 'busy']);
const loading = ref(false);
const busy = ref(false);
const error = ref('');
const notice = ref('');
const inputs = ref({ targets: [], answers: [] });
const fixture = ref(null);
const scenarioReplays = ref({});
const resolutionFixtureIdentities = ref([]);
const replay = computed(() => scenarioReplays.value[repairFixtureIdentity.value] || null);
const findings = ref([]);
const approvedFixtures = ref([]);
const repairFixtureIdentity = ref('');
const replayAnswerIdentity = ref('');
const reacceptanceIdentity = ref('');
const providerAuthorization = ref(null);
let requestSequence = 0;
const outcomes = [
  ['insufficient_evidence_reply', '证据不足'], ['evidence_gated_answer', '证据支持的回答'], ['generation_unavailable', '生成不可用']
];
const observations = [
  ['coverage_gap', '覆盖缺口'], ['retrieval_miss', '检索遗漏'], ['condition_loss', '条件丢失'],
  ['citation_drift', '引用偏移'], ['provider_failure', 'Provider 失败'], ['product_failure', '产品失败'],
  ['stale_source', '来源过时'], ['wrong_content', '内容错误'], ['confirmation', '确认']
];
const labelNames = { helpful: '有帮助', insufficient_evidence: '证据不足', outdated: '已过时', out_of_scope: '超出范围' };
const sourceStates = {
  verified_usable: '已验证可用',
  changed_or_unreachable_awaiting_review: '已变化或不可达，待评审',
  unavailable_for_new_evidence: '不可用于新证据'
};
const outcomeLabel = (value) => outcomes.find(([key]) => key === value)?.[1] || '无已完成结果';
const observationLabel = (value) => observations.find(([key]) => key === value)?.[1] || value;
const draft = reactive({
  answer_id: '', signal_id: '', expected_outcome: 'insufficient_evidence_reply', verified_observation: 'coverage_gap',
  confirmed_synthetic_fixture: false, reference_answer_id: '', entry_identity: ''
});
const isWorker = computed(() => props.item.work_owner === props.context.member_identity);
const isMaintainer = computed(() => props.context.is_maintainer && props.item.accountable_maintainer === props.context.member_identity);
const openWork = computed(() => ['triaged', 'in_progress'].includes(props.item.state));
const unavailable = computed(() => props.disabled || loading.value || busy.value);
const needsReference = computed(() => ['retrieval_miss', 'condition_loss', 'citation_drift', 'product_failure'].includes(draft.verified_observation));
const needsEntry = computed(() => ['stale_source', 'wrong_content'].includes(draft.verified_observation));
const supportedAnswers = computed(() => inputs.value.answers.filter((answer) => answer.outcome === 'evidence_gated_answer'));
const approved = computed(() => fixture.value && findings.value.some((finding) => finding.fixture_identity === fixture.value.id));
const repairFixture = computed(() => approvedFixtures.value.find((scenario) => scenario.id === repairFixtureIdentity.value));
const replayAnswers = computed(() => inputs.value.answers.filter((answer) =>
  answer.query_condition_set_identity === repairFixture.value?.query_condition_set_identity));
const replayInputReady = computed(() => replayAnswers.value.some((answer) => answer.answer_id === replayAnswerIdentity.value));
const resolutionPairs = computed(() => approvedFixtures.value
  .filter((scenario) => resolutionFixtureIdentities.value.includes(scenario.id))
  .map((scenario) => ({ fixture: scenario, replay: scenarioReplays.value[scenario.id] })));
const coversScope = (scenarios) => scenarios.length > 0
  && Object.values(props.item.affected_scope || {}).some((targets) => targets.length)
  && Object.entries(props.item.affected_scope || {}).every(([kind, targets]) => targets.every((target) =>
    scenarios.some((scenario) => scenario.affected_scope[kind].some((covered) =>
      Object.entries(target).every(([key, value]) => covered[key] === value)))));
const resolutionReady = computed(() => resolutionPairs.value.every((pair) => pair.replay?.passed)
  && coversScope(resolutionPairs.value.map((pair) => pair.fixture)));
const canCloseConfirmation = computed(() => openWork.value
  && props.item.classification === 'confirmation' && props.item.disposition === 'confirmation'
  && !['p0', 'p1'].includes(props.item.severity)
  && coversScope(approvedFixtures.value.filter((scenario) =>
    scenario.verified_observation === 'confirmation'
    && scenario.expected_outcome === 'evidence_gated_answer'
    && scenario.observed_outcome === 'evidence_gated_answer'
    && scenario.affected_scope.entry_versions.every((target) =>
      scenario.evidence_publication_identities.includes(target.publication_identity)))));
const verificationArtifacts = computed(() => resolutionPairs.value.flatMap((pair) => [pair.fixture.id, pair.replay.id]));
const canResolveBoundary = computed(() => props.item.state === 'in_progress' && resolutionReady.value
  && props.item.classification === 'coverage-gap' && props.item.disposition === 'coverage-work'
  && resolutionPairs.value.every((pair) => pair.fixture.expected_outcome === 'insufficient_evidence_reply'
    && pair.fixture.observed_outcome === 'insufficient_evidence_reply')
  && !['p0', 'p1'].includes(props.item.severity));
const changedSources = computed(() => resolutionPairs.value.flatMap((pair) => (pair.fixture.publication_review?.source_facts || [])
  .filter((source) => source.availability !== 'verified_usable').map((source) => source.source_identity)));
const highSeverity = computed(() => ['p0', 'p1'].includes(props.item.severity));
const authorizationMatches = (conditions) => providerAuthorization.value?.item_identity === props.item.id
  && providerAuthorization.value.item_revision === props.item.revision
  && providerAuthorization.value.work_owner === props.context.member_identity
  && providerAuthorization.value.query_condition_set_identity === conditions;
const requiresReproductionAuthorization = computed(() => highSeverity.value
  && props.item.classification === 'product-privacy-operations' && draft.verified_observation === 'provider_failure');
const reproductionAuthorizationReady = computed(() => authorizationMatches(
  inputs.value.answers.find((answer) => answer.answer_id === draft.answer_id)?.query_condition_set_identity
));
const requiresReplayAuthorization = computed(() => highSeverity.value
  && props.item.classification === 'product-privacy-operations' && repairFixture.value?.verified_observation === 'provider_failure');
const replayAuthorizationReady = computed(() => authorizationMatches(repairFixture.value?.query_condition_set_identity));
const requiresReacceptance = computed(() => highSeverity.value && props.item.state === 'in_progress' && resolutionReady.value
  && ({
    'source-freshness': ['source-change'],
    'content-integrity': ['entry-revision'],
    'coverage-gap': ['coverage-work'],
    'retrieval-answer-behavior': ['retrieval-experiment'],
    'product-privacy-operations': ['provider-work', 'product-repair']
  }[props.item.classification] || []).includes(props.item.disposition));
const reacceptanceReady = computed(() => /^delivery_acceptance_record:[a-z0-9][a-z0-9._:-]{2,159}$/.test(reacceptanceIdentity.value));
const canResolveSource = computed(() => props.item.state === 'in_progress' && resolutionReady.value
  && props.item.classification === 'source-freshness' && props.item.disposition === 'source-change'
  && changedSources.value.length > 0 && resolutionPairs.value.every((pair) =>
    pair.fixture.verified_observation === 'stale_source' && pair.fixture.expected_outcome === 'evidence_gated_answer')
  && (!highSeverity.value || reacceptanceReady.value));
const canResolveProvider = computed(() => props.item.state === 'in_progress' && resolutionReady.value
  && props.item.classification === 'product-privacy-operations' && props.item.disposition === 'provider-work'
  && resolutionPairs.value.every((pair) => pair.fixture.verified_observation === 'provider_failure'
    && pair.replay.generation_context && pair.replay.observed_outcome === 'evidence_gated_answer'
    && (!highSeverity.value || pair.replay.generation_context.authorization_identity))
  && (!highSeverity.value || reacceptanceReady.value));
const canResolveRetrieval = computed(() => props.item.state === 'in_progress' && resolutionReady.value
  && props.item.classification === 'retrieval-answer-behavior' && props.item.disposition === 'retrieval-experiment'
  && resolutionPairs.value.every((pair) => ['citation_drift', 'retrieval_miss', 'condition_loss'].includes(pair.fixture.verified_observation)
    && pair.replay.observed_outcome === 'evidence_gated_answer') && (!highSeverity.value || reacceptanceReady.value));
const canResolveProduct = computed(() => props.item.state === 'in_progress' && resolutionReady.value
  && props.item.classification === 'product-privacy-operations' && props.item.disposition === 'product-repair'
  && resolutionPairs.value.every((pair) => pair.fixture.verified_observation === 'product_failure'
    && pair.replay.observed_outcome === 'evidence_gated_answer') && (!highSeverity.value || reacceptanceReady.value));
const canResolveContent = computed(() => props.item.state === 'in_progress' && resolutionReady.value
  && props.item.classification === 'content-integrity' && props.item.disposition === 'entry-revision'
  && resolutionPairs.value.every((pair) => pair.fixture.verified_observation === 'wrong_content' && pair.replay.publication_review
    && pair.replay.observed_outcome === 'evidence_gated_answer') && (!highSeverity.value || reacceptanceReady.value));
const canResolveCoverage = computed(() => props.item.state === 'in_progress' && resolutionReady.value
  && props.item.classification === 'coverage-gap' && props.item.disposition === 'coverage-work'
  && resolutionPairs.value.every((pair) => pair.fixture.verified_observation === 'coverage_gap'
    && pair.fixture.expected_outcome === 'evidence_gated_answer' && pair.replay.observed_outcome === 'evidence_gated_answer'
    && pair.replay.evidence_publications?.length)
  && (!highSeverity.value || reacceptanceReady.value));

const load = async () => {
  const sequence = ++requestSequence;
  const item = props.item;
  replayAnswerIdentity.value = '';
  loading.value = true;
  error.value = '';
  try {
    const [choices, registered, verified, replayed] = await Promise.all([
      isWorker.value && openWork.value ? maintenanceApi.reproductionInputs(item.id) : { targets: [], answers: [] },
      item.fixture_identity ? maintenanceApi.fixture(item.fixture_identity) : null,
      Promise.all((item.finding_identities || []).map((identity) => maintenanceApi.finding(identity))),
      item.replay_identity ? maintenanceApi.replayResult(item.replay_identity) : null
    ]);
    if (sequence !== requestSequence) return;
    const approvedIdentities = [...new Set(verified.map((finding) => finding.fixture_identity))];
    const scenarios = await Promise.all(approvedIdentities.map((identity) =>
      identity === registered?.id ? registered : maintenanceApi.fixture(identity)));
    const results = await Promise.all(scenarios.map(async (scenario) => [
      scenario.id, scenario.latest_replay_identity
        ? (scenario.latest_replay_identity === replayed?.id ? replayed : await maintenanceApi.replayResult(scenario.latest_replay_identity))
        : null
    ]));
    if (sequence !== requestSequence) return;
    inputs.value = choices;
    fixture.value = registered;
    findings.value = verified;
    scenarioReplays.value = Object.fromEntries(results);
    resolutionFixtureIdentities.value = results.filter(([, result]) => result?.passed).map(([identity]) => identity);
    approvedFixtures.value = scenarios;
    if (!approvedIdentities.includes(repairFixtureIdentity.value)) {
      repairFixtureIdentity.value = [replayed?.fixture_identity, registered?.id, ...approvedIdentities]
        .find((identity) => approvedIdentities.includes(identity)) || '';
    }
  } catch (failure) {
    if (sequence !== requestSequence) return;
    inputs.value = { targets: [], answers: [] };
    fixture.value = null;
    findings.value = [];
    scenarioReplays.value = {};
    resolutionFixtureIdentities.value = [];
    approvedFixtures.value = [];
    repairFixtureIdentity.value = '';
    error.value = failure.message || '维护证据加载失败。';
  } finally {
    if (sequence === requestSequence) loading.value = false;
  }
};
const act = async (operation, message) => {
  if (unavailable.value) return;
  busy.value = true;
  emit('busy', true);
  error.value = '';
  notice.value = '';
  try {
    await operation();
    notice.value = message;
    emit('refresh');
  } catch (failure) {
    error.value = failure.code === 'MAINTENANCE_STALE' ? '该事项已更新，请刷新后重试。' : (failure.message || '证据操作失败。');
  } finally {
    busy.value = false;
    emit('busy', false);
  }
};
const reproduce = () => {
  if (requiresReproductionAuthorization.value && !reproductionAuthorizationReady.value) return;
  return act(async () => {
    await maintenanceApi.reproduce(props.item.id, {
      expected_revision: props.item.revision, answer_id: draft.answer_id, signal_id: draft.signal_id,
      expected_outcome: draft.expected_outcome, verified_observation: draft.verified_observation,
      confirmed_synthetic_fixture: draft.confirmed_synthetic_fixture,
      ...(needsReference.value ? { reference_answer_id: draft.reference_answer_id } : {}),
      ...(needsEntry.value ? { entry_identity: draft.entry_identity } : {}),
      ...(requiresReproductionAuthorization.value ? { provider_verification_authorization_identity: providerAuthorization.value.id } : {})
    });
    draft.confirmed_synthetic_fixture = false;
  }, '复现场景已登记');
};
const diagnose = () => act(() => maintenanceApi.diagnose(props.item.id, {
  expected_revision: props.item.revision, fixture_identity: fixture.value.id, observation: fixture.value.verified_observation
}), '诊断已记录');
const approveFinding = () => act(() => maintenanceApi.approveFinding(props.item.id, {
  expected_revision: props.item.revision, fixture_identity: fixture.value.id
}), '非个人发现已批准');
const transition = (state) => act(() => maintenanceApi.transition(props.item.id, props.item.revision, state), '维护状态已更新');
const runReplay = () => {
  if (!replayInputReady.value) return;
  if (requiresReplayAuthorization.value && !replayAuthorizationReady.value) return;
  return act(() => maintenanceApi.replay(props.item.id, {
    expected_revision: props.item.revision, fixture_identity: repairFixture.value.id, answer_id: replayAnswerIdentity.value,
    ...(requiresReplayAuthorization.value ? { provider_verification_authorization_identity: providerAuthorization.value.id } : {})
  }), '重放已完成');
};
const resolveBoundary = () => act(() => maintenanceApi.resolve(props.item.id, {
  expected_revision: props.item.revision, disposition: 'boundary-query', artifact_identities: verificationArtifacts.value
}), '边界查询已确认');
const resolveSource = () => act(() => maintenanceApi.resolve(props.item.id, {
  expected_revision: props.item.revision, disposition: 'source-change',
  artifact_identities: [...new Set([
    ...verificationArtifacts.value, ...changedSources.value,
    ...(highSeverity.value ? [reacceptanceIdentity.value] : [])
  ])]
}), '来源修复已确认');
const resolveProvider = () => act(() => maintenanceApi.resolve(props.item.id, {
  expected_revision: props.item.revision, disposition: 'provider-work',
  artifact_identities: [...new Set([
    ...verificationArtifacts.value,
    ...resolutionPairs.value.flatMap((pair) => [pair.replay.generation_context.route_identity,
      highSeverity.value ? reacceptanceIdentity.value : pair.replay.generation_context.acceptance_record_identity])
  ])]
}), 'Provider 恢复已确认');
const resolveRetrieval = () => act(() => maintenanceApi.resolve(props.item.id, {
  expected_revision: props.item.revision, disposition: 'retrieval-experiment',
  artifact_identities: [...verificationArtifacts.value, ...(highSeverity.value ? [reacceptanceIdentity.value] : [])]
}), '检索实验结果已确认');
const resolveProduct = () => act(() => maintenanceApi.resolve(props.item.id, {
  expected_revision: props.item.revision, disposition: 'product-repair',
  artifact_identities: [...verificationArtifacts.value, ...(highSeverity.value ? [reacceptanceIdentity.value] : [])]
}), '产品修复已确认');
const resolveContent = () => act(() => maintenanceApi.resolve(props.item.id, {
  expected_revision: props.item.revision, disposition: 'entry-revision',
  artifact_identities: [...new Set([
    ...verificationArtifacts.value,
    ...resolutionPairs.value.flatMap((pair) => [pair.replay.publication_review.revision_identity, pair.replay.publication_review.publication_identity]),
    ...(highSeverity.value ? [reacceptanceIdentity.value] : [])
  ])]
}), '内容修复已确认');
const resolveCoverage = () => act(() => maintenanceApi.resolve(props.item.id, {
  expected_revision: props.item.revision, disposition: 'entry-revision',
  artifact_identities: [...new Set([
    ...verificationArtifacts.value,
    ...resolutionPairs.value.flatMap((pair) => pair.replay.evidence_publications.flatMap((publication) =>
      [publication.publication_identity, publication.revision_identity])),
    ...(highSeverity.value ? [reacceptanceIdentity.value] : [])
  ])]
}), '覆盖补齐已确认');
watch(() => [props.item.id, props.item.revision, props.context.member_identity], load, { immediate: true });
watch(() => [props.item.id, props.item.replay_identity], () => { reacceptanceIdentity.value = ''; });
watch(repairFixtureIdentity, () => { replayAnswerIdentity.value = ''; });
watch(() => [props.item.id, props.item.revision, props.context.member_identity], () => { providerAuthorization.value = null; });
onBeforeUnmount(() => { requestSequence += 1; });
</script>

<style scoped>
.maintenance-evidence { min-width: 0; padding-top: 16px; }
.maintenance-evidence section, .maintenance-evidence form { border-top: 1px solid var(--color-rule); padding: 18px 0; }
.maintenance-evidence h3 { font-size: 15px; font-weight: 600; margin: 0 0 14px; }
.maintenance-evidence form { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.maintenance-evidence form h3, .maintenance-evidence__consent, .maintenance-evidence__actions { grid-column: 1 / -1; }
.maintenance-evidence label { display: grid; gap: 6px; min-width: 0; }
.maintenance-evidence input, .maintenance-evidence select, .maintenance-evidence button {
  font: inherit; color: var(--color-ink); border: 1px solid var(--color-rule); border-radius: 4px;
  background: var(--color-paper-raised); min-height: 36px; min-width: 0; padding: 7px 9px;
}
.maintenance-evidence input, .maintenance-evidence select { width: 100%; }
.maintenance-evidence button { display: inline-flex; align-items: center; justify-content: center; gap: 7px; cursor: pointer; }
.maintenance-evidence button:disabled { opacity: .55; cursor: default; }
.maintenance-evidence__consent { display: flex !important; align-items: start; line-height: 1.6; }
.maintenance-evidence__consent input { width: 18px; height: 18px; min-height: 18px; flex-shrink: 0; }
.maintenance-evidence fieldset { min-width: 0; border: 0; padding: 0 0 16px; margin: 0; }
.maintenance-evidence fieldset label { padding: 8px 0; }
.maintenance-evidence fieldset span { min-width: 0; overflow-wrap: anywhere; }
.maintenance-evidence fieldset small { display: block; color: var(--color-ink-soft); }
.maintenance-evidence__actions { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.maintenance-evidence__actions a { color: var(--color-moss); padding: 9px 0; }
.maintenance-evidence__reacceptance { margin-bottom: 12px; }
.maintenance-evidence dl { display: grid; grid-template-columns: 96px minmax(0, 1fr); gap: 8px 12px; line-height: 1.6; }
.maintenance-evidence dt { color: var(--color-ink-soft); }
.maintenance-evidence dd { margin: 0; overflow-wrap: anywhere; }
.maintenance-evidence p { margin: 0; padding: 10px 0; line-height: 1.6; }
.maintenance-evidence__error { color: var(--color-warning); overflow-wrap: anywhere; }
.maintenance-evidence ul { list-style: none; padding: 0; margin: 0; }
.maintenance-evidence li { display: grid; gap: 6px; padding: 10px 0; overflow-wrap: anywhere; }
.maintenance-evidence li span { color: var(--color-ink-soft); }
@media (max-width: 760px) {
  .maintenance-evidence form { grid-template-columns: minmax(0, 1fr); }
  .maintenance-evidence dl { grid-template-columns: 78px minmax(0, 1fr); gap: 8px; }
}
</style>
