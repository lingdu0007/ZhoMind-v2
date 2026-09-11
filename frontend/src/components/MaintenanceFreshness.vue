<template>
  <div class="maintenance-freshness">
  <form aria-label="来源时效复核" @submit.prevent="submit">
    <h3>来源时效复核</h3>
    <label>反馈条目<select v-model="entryId" :disabled="busy || disabled" aria-label="复核条目">
      <option v-for="entry in entries" :key="entry" :value="entry">{{ entry }}</option>
    </select></label>
    <p v-if="loading" role="status">正在核对条目权限…</p>
    <p v-if="error" role="alert">{{ error }}</p>
    <p v-else-if="notice" role="status">{{ notice }}</p>
    <template v-if="review">
      <dl>
        <dt>发布版本</dt><dd>{{ review.publication_identity || '当前修订尚未发布' }}</dd>
        <dt>首次复核</dt><dd>{{ review.started_at || '尚未登记' }}</dd>
        <dt>证据资格</dt><dd>{{ eligible ? '可用于回答' : '不可用于新回答' }}</dd>
      </dl>
      <label>时效触发条件<select v-model="triggerId" :disabled="loading || busy || disabled" aria-label="时效触发条件">
        <option v-for="trigger in review.trigger_ids" :key="trigger" :value="trigger">{{ trigger }}</option>
      </select></label>
      <button v-if="review.can_request" type="submit" :disabled="loading || busy || disabled || !triggerId">
        <Clock3 :size="16" aria-hidden="true" />登记来源复核
      </button>
    </template>
  </form>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue';
import { Clock3 } from 'lucide-vue-next';
import { maintenanceApi } from '../api/maintenance';

const props = defineProps({ targets: { type: Array, required: true }, disabled: Boolean });
const emit = defineEmits(['busy']);
const entries = computed(() => [...new Set(props.targets.map((target) => target.entry_id).filter(Boolean))]);
const entryId = ref('');
const triggerId = ref('');
const review = ref(null);
const eligible = ref(false);
const loading = ref(false);
const busy = ref(false);
const error = ref('');
const notice = ref('');
let sequence = 0;
const load = async () => {
  const current = ++sequence;
  review.value = null;
  error.value = '';
  if (!entryId.value) {
    loading.value = false;
    return;
  }
  loading.value = true;
  try {
    const result = await maintenanceApi.editorialEntry(entryId.value);
    if (current !== sequence) return;
    review.value = result.freshness_review;
    eligible.value = result.answer_eligible;
    if (!review.value.trigger_ids.includes(triggerId.value)) triggerId.value = review.value.trigger_ids[0] || '';
  } catch (failure) {
    if (current === sequence) error.value = failure.message || '条目复核权限核对失败。';
  } finally {
    if (current === sequence) loading.value = false;
  }
};
const submit = async () => {
  if (busy.value || loading.value || props.disabled || !review.value?.can_request || !triggerId.value) return;
  busy.value = true;
  emit('busy', true);
  error.value = '';
  notice.value = '';
  try {
    await maintenanceApi.requestFreshnessReview(entryId.value, {
      publication_identity: review.value.publication_identity,
      revision_identity: review.value.revision_identity, trigger_id: triggerId.value
    });
    notice.value = '来源复核已登记';
    await load();
  } catch (failure) {
    error.value = failure.message || '来源复核登记失败。';
  } finally {
    busy.value = false;
    emit('busy', false);
  }
};
watch(entries, (values) => {
  if (!values.includes(entryId.value)) entryId.value = values[0] || '';
}, { immediate: true });
watch(entryId, () => { notice.value = ''; load(); }, { immediate: true });
onBeforeUnmount(() => { sequence += 1; });
</script>

<style scoped>
.maintenance-freshness { min-width: 0; border-top: 1px solid var(--color-rule); padding: 18px 0; }
.maintenance-freshness form { min-width: 0; display: grid; grid-template-columns: minmax(0, 1fr); gap: 12px; }
.maintenance-freshness h3 { font-size: 15px; margin: 0; }
.maintenance-freshness label { display: grid; gap: 6px; min-width: 0; }
.maintenance-freshness select, .maintenance-freshness button {
  min-width: 0; min-height: 36px; padding: 7px 9px; font: inherit; color: var(--color-ink);
  background: var(--color-paper-raised); border: 1px solid var(--color-rule); border-radius: 4px;
}
.maintenance-freshness button { display: inline-flex; align-items: center; gap: 7px; justify-self: start; cursor: pointer; }
.maintenance-freshness button:disabled { opacity: .55; cursor: default; }
.maintenance-freshness dl { display: grid; grid-template-columns: 80px minmax(0, 1fr); gap: 8px; margin: 0; line-height: 1.6; }
.maintenance-freshness dt { color: var(--color-ink-soft); }
.maintenance-freshness dd { margin: 0; overflow-wrap: anywhere; }
.maintenance-freshness p { margin: 0; line-height: 1.6; overflow-wrap: anywhere; }
.maintenance-freshness [role="alert"] { color: var(--color-warning); }
</style>
