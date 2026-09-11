<template>
  <section class="maintenance-cadence" :aria-busy="loading || busy">
    <header class="maintenance-cadence__heading">
      <h2>例行评审</h2>
      <button type="button" title="刷新例行评审" aria-label="刷新例行评审" :disabled="unavailable" @click="load">
        <RefreshCw :size="17" aria-hidden="true" />
      </button>
    </header>
    <p v-if="loading" role="status">正在加载评审快照…</p>
    <p v-if="error" role="alert" class="maintenance-cadence__error">{{ error }}</p>
    <p v-else-if="notice" role="status">{{ notice }}</p>
    <template v-if="reviewContext && dashboard">
      <dl class="maintenance-cadence__summary">
        <dt>逾期未分诊</dt><dd>{{ dashboard.overdue_triage_count }}</dd>
        <template v-for="[value, label] in periods" :key="value">
          <dt>{{ label }}</dt><dd :class="{ 'maintenance-cadence__due': dashboard.cadence_due[value] }">{{ dashboard.cadence_due[value] ? '待评审' : '已记录' }}</dd>
        </template>
      </dl>
      <section aria-label="本次工作快照">
        <h3>未结工作</h3>
        <p v-if="!Object.keys(reviewContext.item_revisions).length">暂无本人负责的未结维护项。</p>
        <ul class="maintenance-cadence__work">
          <li v-for="[identity, revision] in Object.entries(reviewContext.item_revisions)" :key="identity">
            <span>{{ identity }}</span><span>修订 {{ revision }}</span>
          </li>
        </ul>
      </section>
      <section aria-label="知识健康">
        <h3>知识健康</h3>
        <div class="maintenance-cadence__table-scroll">
          <table aria-label="知识健康快照">
            <thead><tr><th>条目</th><th>发布版本</th><th>当前编辑版本</th><th>来源状态</th></tr></thead>
            <tbody>
              <tr v-for="entry in reviewContext.knowledge_health.published_entries" :key="entry.entry_identity">
                <td>{{ entry.entry_identity }}</td>
                <td>{{ entry.publication_identity }}<small>{{ entry.published_revision_identity }}</small></td>
                <td>{{ entry.current_revision_identity }}<small>{{ entry.current_lifecycle_state }} · {{ entry.current_revision_answer_eligible ? '可用于回答' : '不可用于回答' }}</small>
                  <small v-if="entry.published_revision_identity !== entry.current_revision_identity">存在后续编辑版本</small>
                </td>
                <td>{{ entry.source_states.map(sourceLabel).join(' · ') }}</td>
              </tr>
              <tr v-if="!reviewContext.knowledge_health.published_entries.length"><td colspan="4">暂无当前发布版本。</td></tr>
            </tbody>
          </table>
        </div>
      </section>
      <section aria-label="延期快照">
        <h3>延期工作</h3>
        <p v-if="!reviewContext.deferrals.length">暂无待评审延期。</p>
        <ul class="maintenance-cadence__deferrals">
          <li v-for="candidate in reviewContext.deferrals" :key="candidate.id">
            <span>{{ candidate.id }}</span><span>修订 {{ candidate.revision }} · {{ candidate.owner_identity }}</span>
            <span :class="{ 'maintenance-cadence__due': candidate.overdue_review }">{{ candidate.review_date }}{{ candidate.overdue_review ? ' · 已逾期' : '' }}</span>
          </li>
        </ul>
      </section>
      <form aria-label="记录例行评审" class="maintenance-cadence__form" @submit.prevent="record">
        <label>评审周期<select v-model="period" aria-label="评审周期" :disabled="unavailable">
          <option v-for="[value, label] in periods" :key="value" :value="value">{{ label }}</option>
        </select></label>
        <fieldset v-if="period === 'quarterly'">
          <legend>本季度抽样验收</legend>
          <p v-if="!reviewContext.sample_options.length">暂无本季度已核验的有效验收记录。</p>
          <div v-for="sample in reviewContext.sample_options" :key="sample.record_identity" class="maintenance-cadence__sample-details">
            <label class="maintenance-cadence__sample">
              <input v-model="samples" type="checkbox" :value="sample.record_identity" :aria-label="`抽样验收 ${sample.record_identity}`" :disabled="unavailable" />
              <span>{{ sample.record_identity }}<small>{{ sample.accepted_scope.collection_identities.join(' · ') || sample.accepted_scope.deployment_identity }}</small></span>
            </label>
            <dl><dt>核验时间</dt><dd>{{ sample.verified_at }}</dd><dt>核验人</dt><dd>{{ sample.verified_by }}</dd></dl>
            <ul aria-label="抽样核验结果">
              <li v-for="check in sample.verified_checks" :key="check.check_id">
                <span>{{ check.check_id }}</span><span>{{ check.result === 'passed' ? '通过' : '沿用已验证结果' }}</span>
                <small v-for="link in check.evidence_links" :key="link">{{ link }}</small>
              </li>
            </ul>
          </div>
        </fieldset>
        <label class="maintenance-cadence__consent"><input v-model="confirmed" type="checkbox" :disabled="unavailable" />确认本次评审</label>
        <button type="submit" :disabled="unavailable || !confirmed || (period === 'quarterly' && !samples.length)">
          <CalendarCheck :size="16" aria-hidden="true" />记录例行评审
        </button>
      </form>
      <section aria-label="评审历史">
        <h3>评审记录</h3>
        <div class="maintenance-cadence__table-scroll">
          <table aria-label="例行评审记录">
            <thead><tr><th>日期</th><th>周期</th><th>工作项</th><th>发布条目</th><th>延期</th><th>抽样</th><th></th></tr></thead>
            <tbody>
              <tr v-for="review in dashboard.cadence_records" :key="review.id">
                <td>{{ review.review_date }}</td><td>{{ periodLabel(review.period) }}</td>
                <td>{{ Object.keys(review.item_revisions).length }}</td><td>{{ review.review_snapshot.knowledge_health.published_entries.length }}</td>
                <td>{{ review.review_snapshot.deferrals.length }}</td><td>{{ review.sampled_acceptances.length }}</td>
                <td><button type="button" :aria-label="`查看评审 ${review.id}`" title="查看评审快照" @click="selectedReviewIdentity = review.id"><Eye :size="16" aria-hidden="true" /></button></td>
              </tr>
              <tr v-if="!dashboard.cadence_records.length"><td colspan="7">尚无已记录评审。</td></tr>
            </tbody>
          </table>
        </div>
      </section>
      <section v-if="selectedReview" aria-label="已记录评审快照">
        <h3>{{ periodLabel(selectedReview.period) }} · {{ selectedReview.review_date }}</h3>
        <dl><dt>评审记录</dt><dd>{{ selectedReview.id }}</dd><dt>快照摘要</dt><dd>{{ selectedReview.context_sha256 }}</dd>
          <dt>负责维护人</dt><dd>{{ selectedReview.review_snapshot.maintainer_identity }}</dd>
          <dt>发布条目</dt><dd>{{ selectedReview.review_snapshot.knowledge_health.published_entries.length }}</dd>
          <dt>延期事项</dt><dd>{{ selectedReview.review_snapshot.deferrals.length }}</dd>
        </dl>
        <h4 v-if="selectedReview.sampled_acceptances.length">已绑定抽样验收</h4>
        <ul class="maintenance-cadence__samples">
          <li v-for="sample in selectedReview.sampled_acceptances" :key="sample.record_identity">
            <strong>{{ sample.record_identity }}</strong><span>{{ sample.status_event_id }}</span>
            <span>{{ sample.accepted_scope.entry_identities.join(' · ') }}</span>
            <span>{{ sample.accepted_scope.collection_identities.join(' · ') }}</span>
            <dl><dt>核验时间</dt><dd>{{ sample.verified_at }}</dd><dt>核验人</dt><dd>{{ sample.verified_by }}</dd></dl>
            <div v-for="check in sample.verified_checks" :key="check.check_id">
              <span>{{ check.check_id }}</span><small>{{ check.result === 'passed' ? '通过' : '沿用已验证结果' }}</small>
              <small v-for="link in check.evidence_links" :key="link">{{ link }}</small>
            </div>
          </li>
        </ul>
      </section>
    </template>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue';
import { CalendarCheck, Eye, RefreshCw } from 'lucide-vue-next';
import { maintenanceApi } from '../api/maintenance';

const props = defineProps({ context: { type: Object, required: true }, disabled: Boolean });
const emit = defineEmits(['busy']);
const reviewContext = ref(null);
const dashboard = ref(null);
const loading = ref(false);
const busy = ref(false);
const error = ref('');
const notice = ref('');
const period = ref('weekly');
const confirmed = ref(false);
const samples = ref([]);
const selectedReviewIdentity = ref('');
let requestSequence = 0;
const periods = [['weekly', '每周'], ['monthly', '每月'], ['quarterly', '每季度']];
const periodLabel = (value) => periods.find(([key]) => key === value)?.[1] || value;
const sourceLabel = (value) => ({
  verified_usable: '已验证可用', changed_or_unreachable_awaiting_review: '变更或不可达，待复核', unavailable_for_new_evidence: '不可用于新证据'
})[value] || value;
const unavailable = computed(() => props.disabled || loading.value || busy.value);
const selectedReview = computed(() => dashboard.value?.cadence_records.find((review) => review.id === selectedReviewIdentity.value));
const load = async () => {
  const sequence = ++requestSequence;
  loading.value = true;
  error.value = '';
  confirmed.value = false;
  try {
    const [context, state] = await Promise.all([maintenanceApi.reviewContext(), maintenanceApi.dashboard()]);
    if (sequence !== requestSequence) return;
    reviewContext.value = context;
    dashboard.value = state;
    samples.value = samples.value.filter((identity) => context.sample_options.some((sample) => sample.record_identity === identity));
  } catch (failure) {
    if (sequence !== requestSequence) return;
    reviewContext.value = null;
    dashboard.value = null;
    error.value = failure.message || '评审快照加载失败。';
  } finally {
    if (sequence === requestSequence) loading.value = false;
  }
};
const record = async () => {
  if (unavailable.value || !confirmed.value) return;
  busy.value = true;
  emit('busy', true);
  error.value = '';
  notice.value = '';
  try {
    const result = await maintenanceApi.recordCadence({
      period: period.value, item_revisions: reviewContext.value.item_revisions, context_sha256: reviewContext.value.context_sha256,
      sample_acceptance_identities: period.value === 'quarterly' ? samples.value : [],
      evidence_links: [`evidence://maintenance/${period.value}-review`]
    });
    selectedReviewIdentity.value = result.id;
    await load();
    if (!error.value) notice.value = '例行评审已记录';
  } catch (failure) {
    error.value = failure.code === 'MAINTENANCE_STALE' ? '评审依据已更新，请刷新快照后重新核对。' : (failure.message || '例行评审记录失败。');
    confirmed.value = false;
  } finally {
    busy.value = false;
    emit('busy', false);
  }
};
watch(period, () => { confirmed.value = false; });
watch(() => props.context.member_identity, load, { immediate: true });
onBeforeUnmount(() => { requestSequence += 1; });
</script>

<style scoped>
.maintenance-cadence { padding: 20px 0; min-width: 0; }
.maintenance-cadence h2, .maintenance-cadence h3, .maintenance-cadence h4, .maintenance-cadence p { margin: 0; }
.maintenance-cadence h2 { font-size: 18px; }
.maintenance-cadence h3 { font-size: 15px; margin-bottom: 14px; }
.maintenance-cadence h4 { font-size: 13px; margin: 16px 0 10px; }
.maintenance-cadence p { padding: 12px 0; line-height: 1.6; }
.maintenance-cadence__heading { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.maintenance-cadence section, .maintenance-cadence__form { border-top: 1px solid var(--color-rule); padding: 20px 0; }
.maintenance-cadence button, .maintenance-cadence select, .maintenance-cadence input {
  min-width: 0; min-height: 36px; padding: 7px 10px; font: inherit; color: var(--color-ink);
  border: 1px solid var(--color-rule); border-radius: 4px; background: var(--color-paper-raised);
}
.maintenance-cadence button { display: inline-flex; align-items: center; justify-content: center; gap: 7px; cursor: pointer; }
.maintenance-cadence button:disabled { opacity: .55; cursor: default; }
.maintenance-cadence label { display: grid; gap: 6px; min-width: 0; }
.maintenance-cadence dl { display: grid; grid-template-columns: 100px minmax(0, 1fr); gap: 8px 14px; line-height: 1.6; }
.maintenance-cadence dd { margin: 0; overflow-wrap: anywhere; }
.maintenance-cadence dt { color: var(--color-ink-soft); }
.maintenance-cadence__summary { grid-template-columns: repeat(4, minmax(0, 1fr)) !important; padding: 16px 0; }
.maintenance-cadence__due, .maintenance-cadence__error { color: var(--color-warning); }
.maintenance-cadence__table-scroll { width: 100%; max-width: 100%; overflow-x: auto; }
.maintenance-cadence table { border-collapse: collapse; table-layout: fixed; width: 100%; min-width: 680px; font-size: 12px; line-height: 1.6; }
.maintenance-cadence th, .maintenance-cadence td { padding: 10px; border-bottom: 1px solid var(--color-rule); text-align: left; vertical-align: top; overflow-wrap: anywhere; }
.maintenance-cadence th { color: var(--color-ink-soft); font-weight: 500; }
.maintenance-cadence small { display: block; font: inherit; color: var(--color-ink-soft); margin-top: 5px; }
.maintenance-cadence__form { display: grid; gap: 16px; justify-items: start; }
.maintenance-cadence__form select { min-width: 160px; }
.maintenance-cadence fieldset { padding: 0; border: 0; margin: 0; min-width: 0; max-width: 100%; }
.maintenance-cadence legend { margin-bottom: 12px; }
.maintenance-cadence__consent, .maintenance-cadence__sample { display: flex !important; align-items: start; gap: 10px; line-height: 1.6; }
.maintenance-cadence__sample { padding: 8px 0; }
.maintenance-cadence__sample-details { min-width: 0; overflow-wrap: anywhere; border-bottom: 1px solid var(--color-rule); padding: 12px 0; }
.maintenance-cadence__sample span { overflow-wrap: anywhere; min-width: 0; }
.maintenance-cadence input[type="checkbox"] { width: 18px; height: 18px; min-height: 18px; flex-shrink: 0; }
.maintenance-cadence ul { padding: 0; margin: 0; list-style: none; }
.maintenance-cadence li { display: grid; gap: 6px; padding: 8px 0; overflow-wrap: anywhere; }
@media (max-width: 760px) {
  .maintenance-cadence__summary { grid-template-columns: 92px minmax(0, 1fr) !important; }
  .maintenance-cadence dl { grid-template-columns: 92px minmax(0, 1fr); gap: 8px; }
}
</style>
