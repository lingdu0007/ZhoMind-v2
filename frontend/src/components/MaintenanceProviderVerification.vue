<template>
  <div class="provider-verification" role="region" aria-label="Provider 验证授权" :aria-busy="busy">
    <form v-if="isAdministrator" aria-label="Provider 验证授权" @submit.prevent="sign">
      <h3>Provider 验证授权</h3>
      <label>验证路由标识<input v-model.trim="draft.route_identity" aria-label="验证路由标识"
        maxlength="79" required :disabled="unavailable" /></label>
      <label>调用准入验收记录<input v-model.trim="draft.admission_acceptance_identity" aria-label="调用准入验收记录"
        maxlength="190" required :disabled="unavailable" /></label>
      <label class="provider-verification__wide">授权查询条件集<input v-model.trim="draft.query_condition_set_identity"
        aria-label="授权查询条件集" maxlength="64" required :disabled="unavailable" /></label>
      <label class="provider-verification__consent">
        <input v-model="confirmed" type="checkbox" :disabled="unavailable" />确认仅授权所列非个人场景
      </label>
      <button type="submit" :disabled="unavailable || !confirmed || !scopeReady">
        <ShieldCheck :size="16" aria-hidden="true" />签发验证授权
      </button>
    </form>
    <form v-else-if="isWorker" aria-label="选用 Provider 验证授权" @submit.prevent="loadAuthorization">
      <h3>Provider 验证授权</h3>
      <label class="provider-verification__wide">验证授权标识<input v-model.trim="identity" aria-label="验证授权标识"
        maxlength="49" required :disabled="unavailable" /></label>
      <button type="submit" :disabled="unavailable || !identityReady">
        <ShieldCheck :size="16" aria-hidden="true" />核验授权
      </button>
    </form>
    <p v-if="error" role="alert" class="provider-verification__error">{{ error }}</p>
    <p v-else-if="notice" role="status">{{ notice }}</p>
    <dl v-if="authorization">
      <dt>授权标识</dt><dd>{{ authorization.id }}</dd>
      <dt>维护修订</dt><dd>{{ authorization.item_revision }}</dd>
      <dt>验证路由</dt><dd>{{ authorization.route_identity }}</dd>
      <dt>调用准入</dt><dd>{{ authorization.acceptance_record_identity }}</dd>
      <dt>查询条件集</dt><dd>{{ authorization.query_condition_set_identity }}</dd>
    </dl>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, reactive, ref, watch } from 'vue';
import { ShieldCheck } from 'lucide-vue-next';
import { maintenanceApi } from '../api/maintenance';

const props = defineProps({ item: { type: Object, required: true }, context: { type: Object, required: true }, disabled: Boolean });
const emit = defineEmits(['authorization', 'busy']);
const busy = ref(false);
const error = ref('');
const notice = ref('');
const confirmed = ref(false);
const identity = ref('');
const authorization = ref(null);
const draft = reactive({ route_identity: '', admission_acceptance_identity: '', query_condition_set_identity: '' });
const isAdministrator = computed(() => props.context.is_administrator
  && props.item.administrator_identity === props.context.member_identity);
const isWorker = computed(() => props.item.work_owner === props.context.member_identity);
const unavailable = computed(() => props.disabled || busy.value);
const identityReady = computed(() => /^maintenance_item:[a-f0-9]{32}$/.test(identity.value));
const scopeReady = computed(() => /^provider_route:[a-f0-9]{64}$/.test(draft.route_identity)
  && /^delivery_acceptance_record:[A-Za-z0-9._/-]+$/.test(draft.admission_acceptance_identity)
  && /^[a-f0-9]{64}$/.test(draft.query_condition_set_identity));
let sequence = 0;

const clearAuthorization = () => {
  authorization.value = null;
  notice.value = '';
  emit('authorization', null);
};
const act = async (operation, message) => {
  if (unavailable.value) return;
  const current = ++sequence;
  busy.value = true;
  emit('busy', true);
  error.value = '';
  clearAuthorization();
  try {
    const record = await operation();
    if (current !== sequence) return;
    if (record.item_identity !== props.item.id || record.item_revision !== props.item.revision
      || record.work_owner !== props.item.work_owner) {
      throw new Error('授权与当前维护修订不匹配。');
    }
    authorization.value = record;
    notice.value = message;
    emit('authorization', record);
  } catch (failure) {
    if (current === sequence) error.value = failure.message || '验证授权操作失败。';
  } finally {
    if (current === sequence) {
      busy.value = false;
      confirmed.value = false;
      emit('busy', false);
    }
  }
};
const sign = () => {
  if (!isAdministrator.value || !confirmed.value || !scopeReady.value) return;
  return act(() => maintenanceApi.authorizeProviderVerification(props.item.id, {
    expected_revision: props.item.revision, ...draft
  }), '授权已签发');
};
const loadAuthorization = () => {
  if (!isWorker.value || !identityReady.value) return;
  return act(() => maintenanceApi.providerVerificationAuthorization(identity.value), '授权已核验');
};
watch(identity, clearAuthorization);
watch(() => [props.item.id, props.item.revision, props.context.member_identity], () => {
  sequence += 1;
  busy.value = false;
  confirmed.value = false;
  identity.value = '';
  error.value = '';
  clearAuthorization();
  emit('busy', false);
});
onBeforeUnmount(() => { sequence += 1; });
</script>

<style scoped>
.provider-verification { min-width: 0; padding-top: 16px; border-top: 1px solid var(--color-rule); }
.provider-verification form { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.provider-verification h3 { grid-column: 1 / -1; font-size: 15px; margin: 0; font-weight: 600; }
.provider-verification label { display: grid; gap: 6px; min-width: 0; }
.provider-verification input, .provider-verification button {
  min-width: 0; min-height: 36px; padding: 7px 9px; border: 1px solid var(--color-rule); border-radius: 4px;
  font: inherit; color: var(--color-ink); background: var(--color-paper-raised);
}
.provider-verification input { width: 100%; }
.provider-verification button { display: inline-flex; align-items: center; justify-content: center; gap: 7px; cursor: pointer; }
.provider-verification button:disabled { opacity: .55; cursor: default; }
.provider-verification__wide, .provider-verification__consent { grid-column: 1 / -1; }
.provider-verification .provider-verification__consent { display: flex; align-items: start; line-height: 1.6; }
.provider-verification__consent input { width: 18px; height: 18px; min-height: 18px; flex-shrink: 0; }
.provider-verification dl { display: grid; grid-template-columns: 88px minmax(0, 1fr); gap: 8px 12px; line-height: 1.6; }
.provider-verification dt { color: var(--color-ink-soft); }
.provider-verification dd { margin: 0; overflow-wrap: anywhere; }
.provider-verification p { padding: 10px 0; margin: 0; overflow-wrap: anywhere; }
.provider-verification__error { color: var(--color-warning); }
@media (max-width: 760px) { .provider-verification form { grid-template-columns: minmax(0, 1fr); } }
</style>
