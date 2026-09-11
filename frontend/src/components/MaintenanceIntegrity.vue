<template>
  <div class="maintenance-integrity">
    <form aria-label="内容完整性复核" @submit.prevent="submit">
      <h3>内容完整性复核</h3>
      <label>反馈条目<select v-model="entryId" :disabled="busy || disabled" aria-label="完整性复核条目">
        <option v-for="entry in entries" :key="entry" :value="entry">{{ entry }}</option>
      </select></label>
      <p v-if="loading" role="status">正在核对条目权限…</p>
      <p v-if="error" role="alert">{{ error }}</p>
      <p v-else-if="notice" role="status">{{ notice }}</p>
      <template v-if="review">
        <dl>
          <dt>发布版本</dt><dd>{{ review.publication_identity || '当前修订尚未发布' }}</dd>
          <dt>复核记录</dt><dd>{{ review.event_id || '尚未登记' }}</dd>
          <dt>证据资格</dt><dd>{{ eligible ? '可用于回答' : '不可用于新回答' }}</dd>
        </dl>
        <label>复核来源<select v-model="sourceIdentity" aria-label="复核来源"
          :disabled="loading || busy || disabled || Boolean(review.event_id)">
          <option v-for="source in review.source_identities" :key="source" :value="source">{{ source }}</option>
        </select></label>
        <label>缺陷类别<select v-model="defect" aria-label="缺陷类别"
          :disabled="loading || busy || disabled || Boolean(review.event_id)">
          <option value="integrity_defect">完整性缺陷</option>
          <option value="known_contradiction">已知矛盾</option>
        </select></label>
        <label class="maintenance-integrity__confirmation">
          <input v-model="confirmed" type="checkbox" :disabled="loading || busy || disabled || !review.can_record" />
          确认已独立复核当前发布修订和来源
        </label>
        <button v-if="review.can_record" type="submit" :disabled="loading || busy || disabled || !confirmed || !sourceIdentity">
          <ShieldAlert :size="16" aria-hidden="true" />确认完整性缺陷
        </button>
      </template>
    </form>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue';
import { ShieldAlert } from 'lucide-vue-next';
import { maintenanceApi } from '../api/maintenance';

const props = defineProps({ targets: { type: Array, required: true }, disabled: Boolean });
const emit = defineEmits(['busy']);
const entries = computed(() => [...new Set(props.targets.map((target) => target.entry_id).filter(Boolean))]);
const entryId = ref('');
const sourceIdentity = ref('');
const defect = ref('integrity_defect');
const review = ref(null);
const eligible = ref(false);
const confirmed = ref(false);
const loading = ref(false);
const busy = ref(false);
const error = ref('');
const notice = ref('');
let sequence = 0;
const load = async () => {
  const current = ++sequence;
  review.value = null;
  confirmed.value = false;
  error.value = '';
  if (!entryId.value) {
    loading.value = false;
    return;
  }
  loading.value = true;
  try {
    const result = await maintenanceApi.editorialEntry(entryId.value);
    if (current !== sequence) return;
    review.value = result.integrity_review;
    eligible.value = result.answer_eligible;
    sourceIdentity.value = review.value.source_identity || review.value.source_identities[0] || '';
    defect.value = review.value.defect || 'integrity_defect';
  } catch (failure) {
    if (current === sequence) error.value = failure.message || '内容复核权限核对失败。';
  } finally {
    if (current === sequence) loading.value = false;
  }
};
const submit = async () => {
  if (busy.value || loading.value || props.disabled || !review.value?.can_record || !confirmed.value || !sourceIdentity.value) return;
  busy.value = true;
  emit('busy', true);
  error.value = '';
  notice.value = '';
  try {
    await maintenanceApi.recordIntegrityReview(entryId.value, {
      publication_identity: review.value.publication_identity,
      revision_identity: review.value.revision_identity, source_identity: sourceIdentity.value,
      defect: defect.value, confirmed_independent_review: true
    });
    notice.value = '完整性复核已登记';
    await load();
  } catch (failure) {
    error.value = failure.message || '完整性复核登记失败。';
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
.maintenance-integrity { min-width: 0; border-top: 1px solid var(--color-rule); padding: 18px 0; }
.maintenance-integrity form { min-width: 0; display: grid; grid-template-columns: minmax(0, 1fr); gap: 12px; }
.maintenance-integrity h3 { font-size: 15px; margin: 0; }
.maintenance-integrity label { display: grid; gap: 6px; min-width: 0; }
.maintenance-integrity select, .maintenance-integrity button {
  min-width: 0; min-height: 36px; padding: 7px 9px; font: inherit; color: var(--color-ink);
  background: var(--color-paper-raised); border: 1px solid var(--color-rule); border-radius: 4px;
}
.maintenance-integrity button { display: inline-flex; align-items: center; gap: 7px; justify-self: start; cursor: pointer; }
.maintenance-integrity button:disabled { opacity: .55; cursor: default; }
.maintenance-integrity dl { display: grid; grid-template-columns: 80px minmax(0, 1fr); gap: 8px; margin: 0; line-height: 1.6; }
.maintenance-integrity dt { color: var(--color-ink-soft); }
.maintenance-integrity dd { margin: 0; overflow-wrap: anywhere; }
.maintenance-integrity p { margin: 0; line-height: 1.6; overflow-wrap: anywhere; }
.maintenance-integrity [role="alert"] { color: var(--color-warning); }
.maintenance-integrity .maintenance-integrity__confirmation { display: flex; align-items: start; line-height: 1.6; gap: 8px; }
.maintenance-integrity input[type="checkbox"] { flex: 0 0 16px; width: 16px; height: 16px; margin-top: 4px; }
</style>
