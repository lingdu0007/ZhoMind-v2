<template>
  <section class="maintenance-roadmap" :aria-busy="loading || busy">
    <header class="maintenance-roadmap__heading">
      <h2>路线图候选</h2>
      <button type="button" title="刷新路线图" aria-label="刷新路线图" :disabled="unavailable" @click="load">
        <RefreshCw :size="17" aria-hidden="true" />
      </button>
    </header>
    <p v-if="loading" role="status">正在加载路线图候选…</p>
    <p v-if="error" role="alert" class="maintenance-roadmap__error">{{ error }}</p>
    <p v-else-if="notice" role="status">{{ notice }}</p>

    <form v-if="context.is_maintainer && eligibleItems.length" class="maintenance-roadmap__qualification"
      aria-label="路线图资格确认" @submit.prevent="qualify">
      <label>来源维护项<select v-model="draft.item_identity" aria-label="来源维护项" required :disabled="unavailable">
        <option value="">选择维护项</option>
        <option v-for="item in eligibleItems" :key="item.id" :value="item.id">{{ item.severity.toUpperCase() }} · {{ item.classification }} · {{ item.id.slice(-8) }}</option>
      </select></label>
      <label>候选负责人<input v-model.trim="draft.owner_username" required maxlength="64" :disabled="unavailable" /></label>
      <label>期望目标<select v-model="draft.desired_outcome" aria-label="期望目标" :disabled="unavailable">
        <option v-for="[value, label] in outcomes" :key="value" :value="value">{{ label }}</option>
      </select></label>
      <label>无法以有界工作闭合的原因<select v-model="draft.bounded_work_reason" aria-label="无法以有界工作闭合的原因" :disabled="unavailable">
        <option v-for="[value, label] in reasons" :key="value" :value="value">{{ label }}</option>
      </select></label>
      <label>下次评审日期<input v-model="draft.review_date" type="date" required :disabled="unavailable" /></label>
      <button type="submit" :disabled="unavailable || !draft.item_identity"><GitBranch :size="16" aria-hidden="true" />确认路线图资格</button>
    </form>

    <p v-if="!loading && !candidates.length && !error">当前没有可见的路线图候选。</p>
    <div v-if="candidates.length" class="maintenance-roadmap__work">
      <ul aria-label="路线图候选列表" class="maintenance-roadmap__list">
        <li v-for="candidate in candidates" :key="candidate.id">
          <button type="button" :aria-pressed="selected?.id === candidate.id" :disabled="busy" @click="selection = candidate.id">
            <strong>{{ outcomeLabel(candidate.desired_outcome) }}</strong>
            <span>{{ states[candidate.state] }} · {{ candidate.id.slice(-8) }}</span>
            <span v-if="candidate.overdue_review" class="maintenance-roadmap__overdue">评审已逾期</span>
          </button>
        </li>
      </ul>
      <section v-if="selected" class="maintenance-roadmap__detail" aria-label="路线图候选详情">
        <header class="maintenance-roadmap__heading"><h3>{{ outcomeLabel(selected.desired_outcome) }}</h3><strong>{{ states[selected.state] }}</strong></header>
        <dl>
          <dt>候选</dt><dd>{{ selected.id }}</dd>
          <dt>修订</dt><dd>{{ selected.revision }}</dd>
          <dt>负责人</dt><dd>{{ selected.owner_identity }}</dd>
          <dt>评审日期</dt><dd>{{ selected.review_date || '已终结' }}<span v-if="selected.overdue_review" class="maintenance-roadmap__overdue"> · 已逾期</span></dd>
          <dt>期望目标</dt><dd>{{ outcomeLabel(selected.desired_outcome) }}</dd>
          <dt>有界工作限制</dt><dd>{{ reasonLabel(selected.bounded_work_reason) }}</dd>
          <dt>当前范围边界</dt><dd>{{ selected.scope_boundary }}</dd>
          <dt>影响范围</dt><dd>{{ selected.pattern.coverage_position }}</dd>
          <dt>分类</dt><dd>{{ selected.pattern.classification }}</dd>
          <dt>30 天执行数</dt><dd>{{ selected.qualification.distinct_executions_30_days }}</dd>
          <dt>独立发现数</dt><dd>{{ selected.qualification.independent_findings }}</dd>
          <dt>直接范围延期</dt><dd>{{ selected.qualification.direct_scope_deferral ? '是' : '否' }}</dd>
          <dt>维护决策</dt><dd>{{ selected.maintenance_decision_links.join(' · ') }}</dd>
          <template v-if="selected.rationale"><dt>处置理由</dt><dd>{{ rationaleLabel(selected.rationale) }}</dd></template>
        </dl>
        <MaintenanceAffectedScope :scope="selected.affected_scope" />
        <form v-if="selected.owner_identity === context.member_identity && selected.state === 'deferred'"
          aria-label="月度路线图评审" class="maintenance-roadmap__review" @submit.prevent="review">
          <h3>月度评审</h3>
          <label>评审处置<select v-model="reviewDraft.action" aria-label="评审处置" :disabled="unavailable">
            <option value="renew">续期</option><option value="close">关闭候选</option><option value="start_wayfinder">创建独立探索图</option>
          </select></label>
          <label>处置理由<select v-model="reviewDraft.rationale" aria-label="处置理由" :disabled="unavailable">
            <option v-for="[value, label] in rationales" :key="value" :value="value">{{ label }}</option>
          </select></label>
          <label>后续负责人<input v-model.trim="reviewDraft.owner_username" required maxlength="64" :disabled="unavailable" /></label>
          <label v-if="reviewDraft.action === 'renew'">下次评审日期<input v-model="reviewDraft.review_date" type="date" required :disabled="unavailable" /></label>
          <button type="submit" :disabled="unavailable"><CalendarCheck :size="16" aria-hidden="true" />记录月度评审</button>
        </form>
        <section v-if="selected.map_identity" class="maintenance-roadmap__map" aria-label="Wayfinder map">
          <header class="maintenance-roadmap__heading">
            <h3>Wayfinder map</h3>
            <button type="button" title="下载探索图" aria-label="下载探索图" :disabled="!mapArtifact || mapLoading" @click="downloadMap">
              <Download :size="17" aria-hidden="true" />
            </button>
          </header>
          <p v-if="mapLoading" role="status">正在加载探索图…</p>
          <p v-if="mapError" role="alert" class="maintenance-roadmap__error">{{ mapError }}</p>
          <pre v-if="mapArtifact">{{ mapArtifact.markdown }}</pre>
        </section>
      </section>
    </div>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, reactive, ref, watch } from 'vue';
import { CalendarCheck, Download, GitBranch, RefreshCw } from 'lucide-vue-next';
import { maintenanceApi } from '../api/maintenance';
import MaintenanceAffectedScope from './MaintenanceAffectedScope.vue';

const props = defineProps({ context: { type: Object, required: true }, items: { type: Array, required: true }, disabled: Boolean });
const emit = defineEmits(['refresh', 'busy']);
const candidates = ref([]);
const selection = ref('');
const loading = ref(false);
const busy = ref(false);
const error = ref('');
const notice = ref('');
const mapArtifact = ref(null);
const mapLoading = ref(false);
const mapError = ref('');
let requestSequence = 0;
let mapSequence = 0;
const monthlyDate = () => new Date(Date.now() + 30 * 86400000).toISOString().slice(0, 10);
const draft = reactive({
  item_identity: '', owner_username: '', desired_outcome: 'clarify_scope',
  bounded_work_reason: 'requires_separate_scope', review_date: monthlyDate()
});
const reviewDraft = reactive({ action: 'renew', rationale: 'still_outside_scope', owner_username: '', review_date: monthlyDate() });
const outcomes = [['expand_coverage', '扩充知识覆盖'], ['improve_sources', '改善来源'], ['evaluate_retrieval', '评估检索'], ['repair_product_workflow', '修复产品流程'], ['clarify_scope', '明确范围']];
const reasons = [['requires_separate_scope', '需要独立范围'], ['cross_domain_work', '涉及跨领域工作'], ['requires_new_capability', '需要新能力']];
const rationales = [['still_outside_scope', '仍在当前范围之外'], ['separate_discovery_needed', '需要独立探索'], ['no_longer_needed', '已不再需要'], ['resolved_elsewhere', '已在其他工作中解决']];
const states = { deferred: '已延期', closed: '已关闭', mapped: '已建立探索图' };
const outcomeLabel = (value) => outcomes.find(([key]) => key === value)?.[1] || value;
const reasonLabel = (value) => reasons.find(([key]) => key === value)?.[1] || value;
const rationaleLabel = (value) => rationales.find(([key]) => key === value)?.[1] || value;
const selected = computed(() => candidates.value.find((candidate) => candidate.id === selection.value) || candidates.value[0]);
const eligibleItems = computed(() => props.items.filter((item) => item.accountable_maintainer === props.context.member_identity
  && ['triaged', 'in_progress'].includes(item.state) && !['p0', 'p1'].includes(item.severity)));
const unavailable = computed(() => props.disabled || loading.value || busy.value);
const load = async () => {
  const sequence = ++requestSequence;
  loading.value = true;
  error.value = '';
  try {
    const result = await maintenanceApi.roadmap();
    if (sequence === requestSequence) candidates.value = result.candidates;
  } catch (failure) {
    if (sequence !== requestSequence) return;
    candidates.value = [];
    error.value = failure.message || '路线图加载失败。';
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
    await load();
    if (!error.value) notice.value = message;
    emit('refresh');
  } catch (failure) {
    error.value = failure.code === 'MAINTENANCE_STALE' ? '该候选或维护项已更新，请刷新后重试。' : (failure.message || '路线图操作失败。');
  } finally {
    busy.value = false;
    emit('busy', false);
  }
};
const qualify = () => act(async () => {
  const item = eligibleItems.value.find((value) => value.id === draft.item_identity);
  if (!item) throw new Error('来源维护项已更新，请重新选择。');
  const candidate = await maintenanceApi.qualifyRoadmap(item.id, {
    expected_revision: item.revision, owner_username: draft.owner_username,
    desired_outcome: draft.desired_outcome, bounded_work_reason: draft.bounded_work_reason, review_date: draft.review_date
  });
  selection.value = candidate.id;
  draft.item_identity = '';
}, '路线图候选已建立');
const review = () => act(() => maintenanceApi.reviewRoadmap(selected.value.id, {
  expected_revision: selected.value.revision, action: reviewDraft.action, rationale: reviewDraft.rationale,
  owner_username: reviewDraft.owner_username, review_date: reviewDraft.action === 'renew' ? reviewDraft.review_date : null
}), '路线图评审已记录');
const downloadMap = () => {
  if (!mapArtifact.value) return;
  const url = URL.createObjectURL(new Blob([mapArtifact.value.markdown], { type: 'text/markdown;charset=utf-8' }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `wayfinder-${mapArtifact.value.id.split(':')[1]}.md`;
  anchor.click();
  URL.revokeObjectURL(url);
};
watch(() => selected.value?.id, () => {
  reviewDraft.action = 'renew';
  reviewDraft.rationale = 'still_outside_scope';
  reviewDraft.owner_username = props.context.username;
  reviewDraft.review_date = monthlyDate();
});
watch(() => selected.value?.map_identity, async (identity) => {
  const sequence = ++mapSequence;
  mapArtifact.value = null;
  mapError.value = '';
  mapLoading.value = Boolean(identity);
  if (!identity) return;
  try {
    const artifact = await maintenanceApi.map(identity);
    if (sequence === mapSequence) mapArtifact.value = artifact;
  } catch (failure) {
    if (sequence === mapSequence) mapError.value = failure.message || '探索图加载失败。';
  } finally {
    if (sequence === mapSequence) mapLoading.value = false;
  }
});
watch(() => props.context.member_identity, load, { immediate: true });
onBeforeUnmount(() => { requestSequence += 1; mapSequence += 1; });
</script>

<style scoped>
.maintenance-roadmap { padding: 20px 0; min-width: 0; }
.maintenance-roadmap h2, .maintenance-roadmap h3, .maintenance-roadmap p { margin: 0; }
.maintenance-roadmap h2 { font-size: 18px; }
.maintenance-roadmap h3 { font-size: 15px; }
.maintenance-roadmap p { padding: 14px 0; line-height: 1.6; }
.maintenance-roadmap__heading { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
.maintenance-roadmap input, .maintenance-roadmap select, .maintenance-roadmap button {
  min-width: 0; min-height: 36px; padding: 7px 10px; font: inherit; color: var(--color-ink);
  background: var(--color-paper-raised); border: 1px solid var(--color-rule); border-radius: 4px;
}
.maintenance-roadmap input, .maintenance-roadmap select { width: 100%; }
.maintenance-roadmap button { display: inline-flex; align-items: center; justify-content: center; gap: 7px; cursor: pointer; }
.maintenance-roadmap button:disabled { opacity: .55; cursor: default; }
.maintenance-roadmap label { display: grid; gap: 6px; min-width: 0; }
.maintenance-roadmap__qualification, .maintenance-roadmap__review { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; padding: 20px 0; border-bottom: 1px solid var(--color-rule); }
.maintenance-roadmap__qualification button { align-self: end; }
.maintenance-roadmap__review { margin-top: 20px; border-top: 1px solid var(--color-rule); }
.maintenance-roadmap__review h3 { grid-column: 1 / -1; }
.maintenance-roadmap__work { display: grid; grid-template-columns: minmax(210px, 260px) minmax(0, 1fr); gap: 26px; margin-top: 20px; }
.maintenance-roadmap__list { list-style: none; padding: 0; margin: 0; }
.maintenance-roadmap__list button { display: grid; width: 100%; justify-content: stretch; text-align: left; padding: 14px 10px; border: 0; border-bottom: 1px solid var(--color-rule); border-radius: 0; background: transparent; }
.maintenance-roadmap__list span { font-size: 12px; color: var(--color-ink-soft); }
.maintenance-roadmap__list [aria-pressed="true"] { background: var(--color-paper-muted); box-shadow: inset 2px 0 var(--color-moss); }
.maintenance-roadmap__detail { min-width: 0; }
.maintenance-roadmap dl { display: grid; grid-template-columns: 110px minmax(0, 1fr); gap: 8px 14px; line-height: 1.6; }
.maintenance-roadmap dt { color: var(--color-ink-soft); }
.maintenance-roadmap dd { margin: 0; overflow-wrap: anywhere; }
.maintenance-roadmap__error, .maintenance-roadmap__overdue { color: var(--color-warning) !important; overflow-wrap: anywhere; }
.maintenance-roadmap__map { margin-top: 20px; padding-top: 20px; border-top: 1px solid var(--color-rule); }
.maintenance-roadmap pre { padding: 16px 0; margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; line-height: 1.65; }
@media (max-width: 760px) {
  .maintenance-roadmap__work, .maintenance-roadmap__qualification, .maintenance-roadmap__review { grid-template-columns: minmax(0, 1fr); }
  .maintenance-roadmap dl { grid-template-columns: 92px minmax(0, 1fr); gap: 8px; }
}
</style>
