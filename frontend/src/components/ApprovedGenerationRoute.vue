<script setup>
import { computed, onMounted, ref } from 'vue';
import { ArrowDown, ArrowUp, Plus, RefreshCw, Save, ShieldCheck, Trash2 } from 'lucide-vue-next';
import { apiAdapter } from '../api/adapters';

const newProvider = (name = 'primary') => ({
  provider: name, provider_type: 'openai', model: '', service_url: '', endpoint_class: 'public_https',
  data_scope: 'team_shared_pilot', timeout_seconds: 10, provider_api_key: ''
});
const providers = ref([newProvider()]);
const maxAttempts = ref(1);
const totalTimeout = ref(30);
const saved = ref(null);
const active = ref(null);
const currentIdentity = ref(null);
const busy = ref(true);
const error = ref('');
const status = ref('');
const acceptance = ref('');
const savedForm = ref('');
const payload = () => ({
  data_scope: 'team_shared_pilot', providers: providers.value,
  max_attempts: maxAttempts.value, total_timeout_seconds: totalTimeout.value
});
const dirty = computed(() => JSON.stringify(payload()) !== savedForm.value);
const canActivate = computed(() => saved.value && !dirty.value
  && /^delivery_acceptance_record:[A-Za-z0-9._/-]+$/.test(acceptance.value));

const reset = () => {
  providers.value = saved.value ? saved.value.providers.map((item) => ({
    ...newProvider(item.provider),
    ...Object.fromEntries(Object.keys(newProvider()).filter((key) => key !== 'provider_api_key').map((key) => [key, item[key]]))
  })) : [newProvider()];
  maxAttempts.value = saved.value?.max_attempts ?? 1;
  totalTimeout.value = saved.value?.total_timeout_seconds ?? 30;
  savedForm.value = JSON.stringify(payload());
  error.value = '';
};
const load = async () => {
  busy.value = true;
  error.value = '';
  try {
    const result = await apiAdapter.getGenerationRoute();
    saved.value = result.draft;
    active.value = result.active;
    currentIdentity.value = result.current_route_identity;
    reset();
  } catch {
    error.value = '生成路由加载失败';
  } finally {
    busy.value = false;
  }
};
const save = async () => {
  busy.value = true;
  error.value = '';
  status.value = '';
  try {
    saved.value = await apiAdapter.saveGenerationRoute(payload());
    reset();
    acceptance.value = '';
    status.value = '路由草稿已保存';
  } catch {
    error.value = '路由未保存，请检查供应商、凭据和预算';
  } finally {
    providers.value.forEach((item) => { item.provider_api_key = ''; });
    busy.value = false;
  }
};
const activate = async () => {
  if (!canActivate.value) return;
  busy.value = true;
  error.value = '';
  status.value = '正在验证路由';
  try {
    await apiAdapter.activateGenerationRoute({
      route_identity: saved.value.route_identity,
      expected_active_identity: currentIdentity.value,
      acceptance_record_identity: acceptance.value
    });
    await load();
    status.value = '批准路由已激活';
  } catch {
    status.value = '';
    error.value = '路由未激活：证据、权限、版本或连接验证未通过';
  } finally {
    busy.value = false;
  }
};
const add = () => {
  providers.value.push(newProvider(`fallback-${crypto.randomUUID().slice(0, 8)}`));
  maxAttempts.value = providers.value.length;
};
const remove = (index) => {
  providers.value.splice(index, 1);
  maxAttempts.value = Math.min(maxAttempts.value, providers.value.length);
};
const move = (index, direction) => {
  const destination = index + direction;
  [providers.value[index], providers.value[destination]] = [providers.value[destination], providers.value[index]];
};
onMounted(load);
</script>

<template>
  <section class="generation-route" aria-labelledby="generation-route-heading" :aria-busy="busy">
    <header class="generation-route__header">
      <h2 id="generation-route-heading">批准生成路由</h2>
      <button type="button" title="重新加载路由" aria-label="重新加载路由" :disabled="busy" @click="load"><RefreshCw :size="17" /></button>
    </header>
    <div class="generation-route__active">
      <strong>当前活动路由</strong>
      <span data-testid="active-route">{{ active?.route_identity ?? '尚无已激活路由' }}</span>
      <ol v-if="active">
        <li v-for="provider in active.providers" :key="provider.approval_identity">
          {{ provider.provider }} · {{ provider.model }} · {{ provider.data_scope }}
          <span>{{ provider.validation_evidence?.record_identity }}</span>
        </li>
      </ol>
    </div>
    <p v-if="error" role="alert" class="generation-route__error">{{ error }}</p>
    <p v-if="status" role="status">{{ status }}</p>
    <form @submit.prevent="save">
      <fieldset :disabled="busy">
        <section v-for="(provider, index) in providers" :key="index" class="generation-route__provider">
          <div class="generation-route__header">
            <h3>{{ index === 0 ? '主供应商' : `Fallback ${index}` }}</h3>
            <div class="generation-route__tools">
              <button type="button" :disabled="index === 0" :aria-label="`上移供应商 ${index + 1}`" title="上移供应商" @click="move(index, -1)"><ArrowUp :size="17" /></button>
              <button type="button" :disabled="index === providers.length - 1" :aria-label="`下移供应商 ${index + 1}`" title="下移供应商" @click="move(index, 1)"><ArrowDown :size="17" /></button>
              <button type="button" :disabled="providers.length === 1" :aria-label="`移除供应商 ${index + 1}`" title="移除供应商" @click="remove(index)"><Trash2 :size="17" /></button>
            </div>
          </div>
          <div class="generation-route__fields">
            <label>供应商标识<input v-model="provider.provider" :aria-label="`路由供应商标识 ${index + 1}`" required pattern="[A-Za-z0-9][A-Za-z0-9._-]*" /></label>
            <label>协议<select v-model="provider.provider_type" :aria-label="`路由协议 ${index + 1}`"><option value="openai">OpenAI Compatible</option><option value="ark">Ark</option><option value="anthropic">Anthropic</option></select></label>
            <label>模型<input v-model="provider.model" :aria-label="`路由模型 ${index + 1}`" required autocomplete="off" /></label>
            <label class="generation-route__wide">服务 URL<input v-model="provider.service_url" :aria-label="`路由服务 URL ${index + 1}`" type="url" required autocomplete="off" /></label>
            <label>端点类别<select v-model="provider.endpoint_class" :aria-label="`路由端点类别 ${index + 1}`"><option value="public_https">Public HTTPS</option><option value="private_https">Private HTTPS</option></select></label>
            <label>数据范围<select v-model="provider.data_scope" :aria-label="`路由数据范围 ${index + 1}`"><option value="team_shared_pilot">Team-Shared Pilot</option></select></label>
            <label>超时（秒）<input v-model.number="provider.timeout_seconds" :aria-label="`路由超时 ${index + 1}`" type="number" min="0.1" max="60" step="0.1" required /></label>
            <label>替换密钥<input v-model="provider.provider_api_key" :aria-label="`路由密钥 ${index + 1}`" type="password" autocomplete="new-password" required /></label>
          </div>
        </section>
        <button type="button" :disabled="providers.length >= 4" @click="add"><Plus :size="17" />添加 Fallback</button>
        <div class="generation-route__budgets">
          <label>最大尝试数<input v-model.number="maxAttempts" aria-label="路由最大尝试数" type="number" min="1" :max="providers.length" required /></label>
          <label>总预算（秒）<input v-model.number="totalTimeout" aria-label="路由总预算" type="number" min="0.1" max="60" step="0.1" required /></label>
        </div>
        <footer class="generation-route__actions">
          <button type="button" title="重置路由草稿" aria-label="重置路由草稿" @click="reset"><RefreshCw :size="17" /></button>
          <button type="submit" :disabled="!dirty"><Save :size="17" />保存路由</button>
          <span v-if="dirty">路由有未保存修改</span>
        </footer>
      </fieldset>
    </form>
    <div v-if="saved" class="generation-route__approval">
      <strong>已保存路由</strong><span>{{ saved.route_identity }}</span>
      <label>Delivery Acceptance Record<input v-model="acceptance" aria-label="路由验收记录" autocomplete="off" :disabled="busy" /></label>
      <button type="button" :disabled="busy || !canActivate" @click="activate"><ShieldCheck :size="17" />验证并激活路由</button>
    </div>
  </section>
</template>

<style scoped>
.generation-route { padding: 24px 0; border-bottom: 1px solid var(--color-rule); color: var(--color-ink); }
.generation-route__header, .generation-route__tools, .generation-route__actions { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
.generation-route h2 { font-size: 20px; margin: 0; }
.generation-route h3 { font-size: 15px; margin: 0; }
.generation-route fieldset { border: 0; padding: 0; margin: 0; min-width: 0; }
.generation-route__provider { padding: 20px 0; border-bottom: 1px solid var(--color-rule); margin-bottom: 16px; }
.generation-route__fields { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin-top: 16px; }
.generation-route__wide { grid-column: span 2; }
.generation-route label { display: grid; gap: 7px; min-width: 0; font-size: 13px; }
.generation-route input, .generation-route select { width: 100%; box-sizing: border-box; min-height: 40px; border: 1px solid var(--line-strong); border-radius: 4px; background: var(--color-paper-raised); color: var(--color-ink); padding: 8px; font: inherit; }
.generation-route button { display: inline-flex; align-items: center; justify-content: center; gap: 7px; min-width: 36px; min-height: 36px; padding: 7px 10px; border: 1px solid var(--line-strong); border-radius: 4px; background: var(--color-paper-raised); color: var(--color-ink); font: inherit; font-size: 13px; cursor: pointer; }
.generation-route button:disabled { opacity: .45; cursor: not-allowed; }
.generation-route :focus-visible { outline: 2px solid var(--color-focus); outline-offset: 2px; }
.generation-route__active, .generation-route__approval { display: grid; gap: 12px; margin-top: 18px; font-size: 13px; overflow-wrap: anywhere; }
.generation-route__active span, .generation-route__approval > span { font-family: var(--font-mono); font-size: 12px; }
.generation-route__active li { margin-bottom: 8px; }
.generation-route__active li span { display: block; }
.generation-route__budgets { display: grid; grid-template-columns: repeat(2, minmax(0, 180px)); gap: 16px; margin: 20px 0; }
.generation-route__actions { justify-content: flex-start; font-size: 13px; }
.generation-route__error { color: var(--color-danger); }
.generation-route__approval button { justify-self: start; }
@media (max-width: 700px) {
  .generation-route__fields { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .generation-route__wide { grid-column: span 2; }
}
@media (max-width: 420px) {
  .generation-route__fields { grid-template-columns: minmax(0, 1fr); }
  .generation-route__wide { grid-column: auto; }
}
</style>
