<template>
  <section v-if="publication || error" class="publication-withdrawal" aria-label="发布撤回">
    <h3>发布撤回</h3>
    <p v-if="error" role="alert">{{ error }}</p>
    <button v-if="error" type="button" :disabled="busy" @click="load">
      <RefreshCw :size="16" aria-hidden="true" />刷新撤回记录
    </button>
    <template v-if="publication">
      <code>{{ publication.identity }}</code>
      <template v-if="record">
        <p role="status">已撤回</p>
        <dl>
          <dt>原因</dt><dd>{{ record.reason_code }}</dd>
          <dt>操作者</dt><dd>{{ record.actor_identity }}</dd>
          <dt>时间</dt><dd>{{ record.occurred_at }}</dd>
          <dt>审计</dt><dd>{{ record.event_identity }}</dd>
        </dl>
        <p role="status">{{ reconciliation?.state === 'completed' ? '清理完成' : '清理待完成' }}</p>
        <button v-if="reconciliation?.state !== 'completed'" type="button" :disabled="busy" @click="reconcile">
          <RotateCcw :size="16" aria-hidden="true" />重试清理
        </button>
      </template>
      <form v-else-if="isCurrent && !error" @submit.prevent="withdraw">
        <label>撤回原因
          <select v-model="reason" :disabled="busy" aria-label="撤回原因">
            <option value="integrity_defect">正确性缺陷</option>
            <option value="privacy_defect">隐私缺陷</option>
            <option value="source_unavailable">来源不可用</option>
            <option value="editorial_withdrawal">编辑撤回</option>
          </select>
        </label>
        <label class="publication-withdrawal__confirmation">
          <input v-model="confirmed" type="checkbox" :disabled="busy" />
          确认撤回此精确发布版本
        </label>
        <button type="submit" :disabled="busy || !confirmed">
          <ShieldX :size="16" aria-hidden="true" />撤回发布版本
        </button>
      </form>
      <p v-else-if="!isCurrent">历史发布版本</p>
    </template>
  </section>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref } from 'vue';
import { RefreshCw, RotateCcw, ShieldX } from 'lucide-vue-next';
import { apiAdapter } from '../api/adapters';

const props = defineProps({ candidateId: { type: String, required: true } });
const publication = ref(null);
const isCurrent = ref(false);
const record = ref(null);
const reconciliation = ref(null);
const reason = ref('integrity_defect');
const confirmed = ref(false);
const busy = ref(false);
const error = ref('');
let active = true;
let epoch = 0;

async function load() {
  const request = ++epoch;
  busy.value = true;
  error.value = '';
  confirmed.value = false;
  try {
    const result = await apiAdapter.getReviewedCandidatePublication(props.candidateId);
    if (!active || request !== epoch) return;
    publication.value = result.published_knowledge_version;
    isCurrent.value = result.is_current_for_entry === true;
    if (!publication.value) return;
    try {
      const withdrawal = await apiAdapter.getPublicationWithdrawal(publication.value.identity);
      if (!active || request !== epoch) return;
      record.value = withdrawal;
      const status = await apiAdapter.getPublicationReconciliation(publication.value.identity);
      if (active && request === epoch) reconciliation.value = status;
    } catch (failure) {
      if (failure.status !== 404) throw failure;
    }
  } catch {
    if (active && request === epoch) error.value = '撤回记录加载失败。';
  } finally {
    if (active && request === epoch) busy.value = false;
  }
}

async function withdraw() {
  if (busy.value || !confirmed.value || !isCurrent.value || record.value || !publication.value) return;
  busy.value = true;
  error.value = '';
  const identity = publication.value.identity;
  try {
    const result = await apiAdapter.withdrawPublication(identity, {
      reason_code: reason.value,
      trigger: reason.value === 'source_unavailable' ? 'source'
        : reason.value === 'editorial_withdrawal' ? 'editorial' : 'integrity'
    });
    if (!active) return;
    record.value = result;
    confirmed.value = false;
    reconciliation.value = await apiAdapter.getPublicationReconciliation(identity);
  } catch {
    if (active) error.value = '撤回结果未确认，请刷新记录。';
  } finally {
    if (active) busy.value = false;
  }
}

async function reconcile() {
  if (busy.value || !record.value) return;
  busy.value = true;
  error.value = '';
  try {
    const status = await apiAdapter.reconcilePublicationWithdrawal(record.value.publication_identity);
    if (active) reconciliation.value = status;
  } catch {
    if (active) error.value = '清理结果未确认，请刷新记录。';
  } finally {
    if (active) busy.value = false;
  }
}

onMounted(load);
onBeforeUnmount(() => { active = false; epoch += 1; });
</script>

<style scoped>
.publication-withdrawal { padding: 16px 0; border-top: 1px solid #d6dce0; overflow-wrap: anywhere; }
h3 { font-size: 16px; margin: 0 0 12px; }
dl { display: grid; grid-template-columns: 80px minmax(0, 1fr); gap: 8px; }
dd { margin: 0; }
form { display: grid; gap: 12px; margin-top: 12px; }
label { display: flex; align-items: center; gap: 8px; }
select { min-height: 36px; max-width: 100%; border: 1px solid #aab5be; background: white; padding: 4px 8px; }
button { display: inline-flex; align-items: center; justify-content: center; gap: 6px; min-height: 36px; width: fit-content; max-width: 100%; border: 1px solid #9e3542; border-radius: 4px; color: #8c2431; background: #fff; padding: 6px 12px; cursor: pointer; }
button:disabled { opacity: .5; cursor: default; }
code { display: block; white-space: normal; }
</style>
