<template>
  <section class="maintenance" :aria-busy="loading || busy">
    <header class="maintenance__header">
      <div><h1>知识维护</h1><span v-if="context">{{ context.username }}</span></div>
      <button type="button" title="刷新维护工作" aria-label="刷新维护工作" :disabled="loading || busy" @click="load">
        <RefreshCw :size="17" aria-hidden="true" />
      </button>
    </header>
    <p v-if="error" role="alert" class="maintenance__error">{{ error }}</p>
    <p v-if="loading" role="status">正在加载维护工作…</p>
    <p v-else-if="notice" role="status">{{ notice }}</p>

    <template v-if="context">
      <div v-if="context.assignment?.state === 'assigned'" class="maintenance__responsibility">
        <strong>维护责任待接任</strong>
        <button type="button" :disabled="busy || loading" @click="acceptResponsibility">
          <UserCheck :size="16" aria-hidden="true" />接受维护责任
        </button>
      </div>
      <nav class="maintenance__tabs" role="tablist" aria-label="维护视图">
        <button type="button" role="tab" :aria-selected="tab === 'work'" @click="tab = 'work'">维护工作</button>
        <button type="button" role="tab" :aria-selected="tab === 'roadmap'" @click="tab = 'roadmap'">路线图候选</button>
        <button v-if="context.is_maintainer" type="button" role="tab" :aria-selected="tab === 'inbox'" @click="tab = 'inbox'">反馈收件箱</button>
        <button v-if="context.is_maintainer" type="button" role="tab" :aria-selected="tab === 'cadence'" @click="tab = 'cadence'">例行评审</button>
        <button v-if="context.is_administrator" type="button" role="tab" :aria-selected="tab === 'assignment'" @click="tab = 'assignment'">责任指派</button>
      </nav>
      <MaintenanceRoadmap v-if="tab === 'roadmap'" :context="context" :items="items" :disabled="loading || busy"
        @refresh="load" @busy="busy = $event" />
      <MaintenanceCadence v-if="tab === 'cadence' && context.is_maintainer" :context="context" :disabled="loading || busy" @busy="busy = $event" />

      <form v-if="context.is_administrator" class="maintenance__assignment" aria-label="维护责任指派" @submit.prevent="assign">
        <label>维护人用户名<input v-model.trim="assignee" required maxlength="64" :disabled="busy" /></label>
        <button type="submit" :disabled="busy || loading"><UserPlus :size="16" aria-hidden="true" />指派维护责任</button>
      </form>

      <section v-if="tab === 'inbox' && context.is_maintainer" aria-label="反馈收件箱">
        <p v-if="!loading && !signals.length">当前没有保留的反馈。</p>
        <ul class="maintenance__signals">
          <li v-for="signal in signals" :key="signal.id">
            <label class="maintenance__selection">
              <input v-model="selectedSignals" type="checkbox" :value="signal.id" aria-label="选择反馈" :disabled="busy" />
              <strong>{{ feedbackLabels[signal.label] || signal.label }}</strong>
              <span>{{ signal.entry_id || '知识缺口' }}</span>
            </label>
            <dl><dt>提交成员</dt><dd>{{ signal.submitted_by }}</dd><dt>提交时间</dt><dd>{{ signal.submitted_at }}</dd></dl>
            <p v-if="signal.description" class="maintenance__description">{{ signal.description }}</p>
          </li>
        </ul>
        <form class="maintenance__create" aria-label="创建维护项" @submit.prevent="createItem">
          <label>维护分类<select v-model="draft.classification" aria-label="维护分类" :disabled="busy">
            <option v-for="[value, label] in classifications" :key="value" :value="value">{{ label }}</option>
          </select></label>
          <label>优先级<select v-model="draft.severity" aria-label="优先级" :disabled="busy">
            <option v-for="severity in ['p0', 'p1', 'p2', 'p3']" :key="severity" :value="severity">{{ severity.toUpperCase() }}</option>
          </select></label>
          <label>影响范围<select v-model="draft.coverage_position" aria-label="影响范围" :disabled="busy">
            <option v-for="[value, label] in coverage" :key="value" :value="value">{{ label }}</option>
          </select></label>
          <label>Work Owner 用户名<input v-model.trim="draft.work_owner_username" required maxlength="64" :disabled="busy" /></label>
          <label v-if="['p0', 'p1'].includes(draft.severity)">停用验收记录<input v-model.trim="containment" required maxlength="192" :disabled="busy" /></label>
          <button type="submit" :disabled="busy || loading || !selectedSignals.length"><Plus :size="16" aria-hidden="true" />创建维护项</button>
        </form>
      </section>

      <div v-if="tab === 'work'" class="maintenance__work">
        <section aria-label="维护工作列表">
          <p v-if="!loading && !items.length && !error">当前没有分配给你的维护工作。</p>
          <ul class="maintenance__items">
            <li v-for="item in items" :key="item.id">
              <button type="button" :aria-pressed="selectedItem?.id === item.id" @click="selectItem(item)">
                <span>{{ item.severity.toUpperCase() }} · {{ classificationLabel(item.classification) }}</span>
                <small>{{ stateLabels[item.state] || item.state }}</small>
              </button>
            </li>
          </ul>
        </section>
        <section v-if="selectedItem" class="maintenance__detail" aria-label="维护项详情">
          <header><h2>{{ classificationLabel(selectedItem.classification) }}</h2><strong>{{ stateLabels[selectedItem.state] || selectedItem.state }}</strong></header>
          <dl>
            <dt>维护项</dt><dd>{{ selectedItem.id }}</dd>
            <dt>修订</dt><dd>{{ selectedItem.revision }}</dd>
            <dt>影响范围</dt><dd>{{ coverageLabel(selectedItem.coverage_position) }}</dd>
            <dt>处置</dt><dd>{{ selectedItem.disposition }}</dd>
            <dt>维护人</dt><dd>{{ selectedItem.accountable_maintainer }}</dd>
            <dt>Work Owner</dt><dd>{{ selectedItem.work_owner }}</dd>
            <dt>保留反馈</dt><dd>{{ selectedItem.signal_count }}</dd>
            <template v-if="selectedItem.administrator_identity"><dt>参与管理员</dt><dd>{{ selectedItem.administrator_identity }}</dd></template>
            <template v-if="selectedItem.blocking_scope"><dt>停用范围</dt><dd>{{ selectedItem.blocking_scope.scope }} · {{ selectedItem.blocking_scope.identity }}</dd></template>
          </dl>
          <MaintenanceAffectedScope :scope="selectedItem.affected_scope" />
          <div class="maintenance__actions">
            <button v-if="canTriage" type="button" :disabled="busy || loading" @click="transition('triaged')"><ListChecks :size="16" aria-hidden="true" />完成分诊</button>
            <button v-if="context.is_administrator && ['open', 'triaged', 'in_progress'].includes(selectedItem.state)" type="button" :disabled="busy || loading" @click="joinAdministrator">
              <ShieldCheck :size="16" aria-hidden="true" />参与维护
            </button>
          </div>
          <ul v-if="selectedItem.result_links.length" class="maintenance__results">
            <li v-for="link in selectedItem.result_links" :key="link">{{ link }}</li>
          </ul>
          <MaintenanceEvidence :key="selectedItem.id" :item="selectedItem" :context="context" :disabled="loading || busy"
            @refresh="load" @busy="busy = $event" />
        </section>
      </div>
    </template>
  </section>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { ListChecks, Plus, RefreshCw, ShieldCheck, UserCheck, UserPlus } from 'lucide-vue-next';
import { maintenanceApi } from '../api/maintenance';
import MaintenanceEvidence from '../components/MaintenanceEvidence.vue';
import MaintenanceRoadmap from '../components/MaintenanceRoadmap.vue';
import MaintenanceCadence from '../components/MaintenanceCadence.vue';
import MaintenanceAffectedScope from '../components/MaintenanceAffectedScope.vue';

const route = useRoute();
const router = useRouter();
const loading = ref(true);
const busy = ref(false);
const error = ref('');
const notice = ref('');
const context = ref(null);
const tab = ref('work');
const items = ref([]);
const signals = ref([]);
const selectedSignals = ref([]);
const selectedIdentity = ref(typeof route.query.item === 'string' ? route.query.item : '');
const assignee = ref('');
const containment = ref('');
const classifications = [
  ['confirmation', '确认'], ['content-integrity', '内容完整性'], ['source-freshness', '来源时效'],
  ['coverage-gap', '覆盖缺口'], ['retrieval-answer-behavior', '检索与回答行为'],
  ['product-privacy-operations', '产品、隐私与运维'], ['scope-roadmap', '范围与路线图']
];
const coverage = [
  ['rag_source_admission_and_chunking', 'RAG 来源准入与分块'],
  ['sparse_dense_hybrid_and_reranking_choices', '稀疏、稠密、混合检索与重排'],
  ['evidence_sufficiency_refusal_and_acceptance', '证据充分性、拒答与验收'],
  ['tools_and_mcp_permissions_and_failure_behavior', '工具与 MCP 权限及失败行为'],
  ['agent_context_state_and_memory', 'Agent 上下文、状态与记忆'],
  ['orchestration_retry_human_intervention_and_side_effects', '编排、重试、人工介入与副作用'],
  ['provider_failure_and_observability', 'Provider 失败与可观测性'],
  ['prompt_injection_isolation_and_security', '提示注入隔离与安全']
];
const stateLabels = { open: '待分诊', triaged: '已分诊', in_progress: '处理中', resolved: '已解决', deferred: '已延期', closed_confirmation: '已关闭确认' };
const feedbackLabels = { helpful: '有帮助', insufficient_evidence: '证据不足', outdated: '已过时', out_of_scope: '超出范围' };
const draft = reactive({
  classification: 'coverage-gap', severity: 'p3', disposition: 'needs-reproduction',
  coverage_position: 'evidence_sufficiency_refusal_and_acceptance', work_owner_username: ''
});
const classificationLabel = (value) => classifications.find(([key]) => key === value)?.[1] || value;
const coverageLabel = (value) => coverage.find(([key]) => key === value)?.[1] || value;
const selectedItem = computed(() => items.value.find((item) => item.id === selectedIdentity.value) || items.value[0]);
const canTriage = computed(() => context.value?.is_maintainer && selectedItem.value?.state === 'open'
  && selectedItem.value.accountable_maintainer === context.value.member_identity);

const load = async () => {
  loading.value = true;
  error.value = '';
  try {
    context.value = await maintenanceApi.context();
    const [work, inbox] = await Promise.all([
      maintenanceApi.items(),
      context.value.is_maintainer ? maintenanceApi.inbox() : Promise.resolve({ signals: [] })
    ]);
    items.value = work.items;
    signals.value = inbox.signals;
    selectedSignals.value = selectedSignals.value.filter((identity) => signals.value.some((signal) => signal.id === identity));
    if (tab.value === 'inbox' && !context.value.is_maintainer) tab.value = 'work';
  } catch (failure) {
    error.value = failure.message || '维护工作加载失败。';
    items.value = [];
    signals.value = [];
  } finally {
    loading.value = false;
  }
};
const act = async (operation, message) => {
  if (busy.value) return;
  busy.value = true;
  error.value = '';
  notice.value = '';
  try {
    await operation();
    await load();
    if (!error.value) notice.value = message;
  } catch (failure) {
    error.value = failure.code === 'MAINTENANCE_STALE' ? '该事项已更新，请刷新后重试。' : (failure.message || '维护操作失败。');
  } finally {
    busy.value = false;
  }
};
const selectItem = (item) => {
  selectedIdentity.value = item.id;
  void router.replace({ query: { ...route.query, item: item.id } });
};
const assign = () => act(() => maintenanceApi.assign(assignee.value), '维护责任已指派');
const acceptResponsibility = () => act(() => maintenanceApi.accept(context.value.assignment.id), '维护责任已接任');
const createItem = () => act(async () => {
  const created = await maintenanceApi.create({
    ...draft, signal_ids: selectedSignals.value,
    ...(['p0', 'p1'].includes(draft.severity) ? { containment_record_identity: containment.value } : {})
  });
  selectItem(created);
  selectedSignals.value = [];
  tab.value = 'work';
}, '维护项已创建');
const transition = (state) => act(() => maintenanceApi.transition(selectedItem.value.id, selectedItem.value.revision, state), '维护状态已更新');
const joinAdministrator = () => act(() => maintenanceApi.joinAdministrator(selectedItem.value.id, selectedItem.value.revision), '已记录管理员参与');
onMounted(load);
</script>

<style scoped>
.maintenance { width: min(100%, 1120px); margin: 0 auto; color: var(--color-ink); font-size: 13px; }
.maintenance h1, .maintenance h2, .maintenance p, .maintenance ul, .maintenance dd { margin: 0; }
.maintenance h1 { font-size: 24px; font-weight: 600; }
.maintenance h2 { font-size: 17px; font-weight: 600; }
.maintenance p { padding: 12px 0; line-height: 1.6; }
.maintenance__header, .maintenance__detail header, .maintenance__responsibility, .maintenance__actions {
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
}
.maintenance__header { padding-bottom: 18px; border-bottom: 1px solid var(--color-rule); }
.maintenance__header span { display: block; margin-top: 6px; color: var(--color-ink-soft); }
.maintenance button, .maintenance input, .maintenance select {
  min-height: 36px; min-width: 0; border: 1px solid var(--color-rule); border-radius: 4px;
  background: var(--color-paper-raised); color: var(--color-ink); font: inherit;
}
.maintenance button { display: inline-flex; align-items: center; justify-content: center; gap: 7px; padding: 6px 12px; cursor: pointer; }
.maintenance button:disabled { cursor: default; opacity: .55; }
.maintenance input, .maintenance select { width: 100%; padding: 6px 9px; }
.maintenance label { display: grid; gap: 6px; min-width: 0; }
.maintenance__error { color: var(--color-warning); }
.maintenance__tabs { display: flex; gap: 8px; padding: 18px 0; border-bottom: 1px solid var(--color-rule); overflow-x: auto; }
.maintenance__tabs button { flex-shrink: 0; border-color: transparent; background: transparent; }
.maintenance__tabs [aria-selected="true"] { color: var(--color-moss); border-bottom-color: var(--color-moss); }
.maintenance__responsibility { padding: 18px 0; }
.maintenance__assignment { display: flex; align-items: end; gap: 12px; padding: 20px 0; border-bottom: 1px solid var(--color-rule); }
.maintenance__assignment label { width: min(100%, 280px); }
.maintenance__create { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; padding: 22px 0; }
.maintenance__create button { align-self: end; }
.maintenance__signals, .maintenance__items, .maintenance__results { padding: 0; list-style: none; }
.maintenance__signals li { padding: 18px 0; border-bottom: 1px solid var(--color-rule); }
.maintenance .maintenance__selection { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.maintenance__selection input { width: 18px; height: 18px; min-height: 18px; }
.maintenance dl { display: grid; grid-template-columns: minmax(76px, max-content) minmax(0, 1fr); gap: 8px 16px; line-height: 1.6; }
.maintenance dt { color: var(--color-ink-soft); }
.maintenance dd, .maintenance__description, .maintenance__results li { overflow-wrap: anywhere; }
.maintenance__work { display: grid; grid-template-columns: minmax(210px, 280px) minmax(0, 1fr); gap: 28px; padding-top: 20px; }
.maintenance__items li { border-bottom: 1px solid var(--color-rule); }
.maintenance__items button { width: 100%; display: grid; justify-content: stretch; text-align: left; border: 0; border-radius: 0; background: transparent; padding: 14px 10px; }
.maintenance__items small { color: var(--color-ink-soft); }
.maintenance__items [aria-pressed="true"] { background: var(--color-paper-muted); box-shadow: inset 2px 0 var(--color-moss); }
.maintenance__detail { min-width: 0; }
.maintenance__detail header { align-items: start; flex-wrap: wrap; padding: 12px 0; }
.maintenance__detail header strong { color: var(--color-moss); font-size: 12px; }
.maintenance__actions { justify-content: flex-start; flex-wrap: wrap; padding-top: 14px; }
.maintenance__results { margin-top: 18px !important; color: var(--color-ink-soft); }
@media (max-width: 760px) {
  .maintenance__work, .maintenance__create { grid-template-columns: minmax(0, 1fr); }
  .maintenance__assignment { align-items: stretch; flex-direction: column; }
  .maintenance__assignment label { width: 100%; }
  .maintenance__responsibility { align-items: start; flex-direction: column; }
  .maintenance dl { grid-template-columns: 78px minmax(0, 1fr); gap: 8px; }
}
</style>
