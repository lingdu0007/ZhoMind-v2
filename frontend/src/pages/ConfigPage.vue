<template>
  <section class="system-settings" aria-labelledby="system-settings-title">
    <header class="system-settings__header">
      <div>
        <p class="system-settings__eyebrow">系统运维</p>
        <h1 id="system-settings-title">系统设置</h1>
        <p class="system-settings__description">编辑服务端草稿，并在保存后由运行系统确认应用结果。</p>
      </div>
    </header>

    <div v-if="!isDesktop" class="system-settings__desktop-notice" role="status">
      <Monitor :size="18" aria-hidden="true" />
      <p>系统设置当前仅支持桌面工作区。</p>
    </div>

    <template v-else>
      <p v-if="loadError" class="system-settings__error" role="alert">
        <span>{{ loadError }}</span>
        <button type="button" @click="loadDraft">重新加载</button>
      </p>
      <p v-if="saveError" class="system-settings__error" role="alert">{{ saveError }}</p>
      <p v-if="applicationError" class="system-settings__error" role="alert">{{ applicationError }}</p>
      <p v-if="applicationState === 'failed'" class="system-settings__error" role="alert">
        {{ applicationFailureLabel }}
      </p>
      <p v-if="applicationState === 'active'" class="system-settings__success" role="status">设置已生效。</p>

      <form v-if="!loadError" class="system-settings__form" :aria-busy="controlsDisabled" @submit.prevent="saveAndApply">
        <section class="system-settings__section" aria-labelledby="model-provider-title">
          <div class="system-settings__section-heading">
            <h2 id="model-provider-title">模型与提供方</h2>
            <p>每次只配置一个生成服务商；保存的密钥仅可替换，不能读取或复制。</p>
          </div>
          <div class="system-settings__fields">
            <label :class="{ 'system-settings__field--changed': isChanged('provider_type') }" class="system-settings__field">
              <span>模型提供方</span>
              <select v-model="draft.provider_type" :disabled="controlsDisabled" aria-label="模型提供方">
                <option value="ark">Ark</option>
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic</option>
              </select>
              <small v-if="isChanged('provider_type')">模型提供方已修改</small>
              <small v-if="fieldErrors.provider_type" class="system-settings__field-error">{{ fieldErrors.provider_type }}</small>
            </label>
            <label :class="{ 'system-settings__field--changed': isChanged('model') }" class="system-settings__field">
              <span>模型</span>
              <input v-model="draft.model" :disabled="controlsDisabled" type="text" autocomplete="off" aria-label="生成模型" />
              <small v-if="isChanged('model')">模型已修改</small>
              <small v-if="fieldErrors.model" class="system-settings__field-error">{{ fieldErrors.model }}</small>
            </label>
            <label :class="{ 'system-settings__field--changed': isChanged('service_url') }" class="system-settings__field">
              <span>服务 URL</span>
              <input v-model="draft.service_url" :disabled="controlsDisabled" type="url" autocomplete="off" />
              <small v-if="isChanged('service_url')">服务 URL 已修改</small>
              <small v-if="fieldErrors.service_url" class="system-settings__field-error">{{ fieldErrors.service_url }}</small>
            </label>
            <label :class="{ 'system-settings__field--changed': Boolean(providerApiKey) }" class="system-settings__field">
              <span>Provider API 密钥</span>
              <input v-model="providerApiKey" :disabled="controlsDisabled" type="password" autocomplete="new-password" placeholder="仅在替换时填写" />
              <small v-if="providerApiKeyConfigured">Provider API 密钥已配置，内容已隐藏。</small>
              <small v-else>尚未配置 Provider API 密钥。</small>
              <small v-if="providerApiKey">Provider API 密钥已修改</small>
              <small v-if="fieldErrors.provider_api_key" class="system-settings__field-error">{{ fieldErrors.provider_api_key }}</small>
            </label>
          </div>
        </section>

        <footer class="system-settings__state-bar" aria-label="草稿状态">
          <div class="system-settings__state-copy">
            <strong>{{ lifecycleLabel }}</strong>
            <span>{{ savedVersionLabel }}</span>
            <span>{{ activeVersionLabel }}</span>
            <span>{{ lastModifiedLabel }}</span>
            <span v-if="application">{{ applicationAuditLabel }}</span>
          </div>
          <div class="system-settings__state-actions">
            <button type="button" :disabled="controlsDisabled || !dirty" @click="resetDraft">重置到已保存草稿</button>
            <button class="system-settings__save" type="submit" :disabled="controlsDisabled || (!dirty && applicationState !== 'failed')">
              {{ applyActionLabel }}
            </button>
          </div>
          <p>保存并应用前，运行系统不会变化。</p>
        </footer>
      </form>
    </template>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue';
import { Monitor } from 'lucide-vue-next';

import { apiAdapter } from '../api/adapters';

const DESKTOP_MIN_WIDTH = 641;
const emptyDraft = () => ({
  provider_type: 'ark',
  model: '',
  service_url: ''
});

const draft = reactive(emptyDraft());
const savedDraft = ref(emptyDraft());
const providerApiKey = ref('');
const providerApiKeyConfigured = ref(false);
const savedVersion = ref(null);
const activeVersion = ref(null);
const lastModified = ref(null);
const applicationState = ref('draft_only');
const application = ref(null);
const loading = ref(true);
const loadError = ref('');
const saveError = ref('');
const applicationError = ref('');
const fieldErrors = reactive({});
const isDesktop = ref(window.innerWidth >= DESKTOP_MIN_WIDTH);
let applicationRefreshTimer = null;

const dirty = computed(() => JSON.stringify(draft) !== JSON.stringify(savedDraft.value) || Boolean(providerApiKey.value));
const isApplying = computed(() => applicationState.value === 'applying');
const controlsDisabled = computed(() => loading.value || isApplying.value);
const savedVersionLabel = computed(() => (savedVersion.value === null ? '尚未保存草稿版本' : `已保存版本 ${savedVersion.value}`));
const activeVersionLabel = computed(() => (activeVersion.value === null ? '尚无生效版本' : `生效版本 ${activeVersion.value}`));
const lastModifiedLabel = computed(() => {
  if (!lastModified.value) return '尚无修改记录';
  return `最后修改：${lastModified.value.actor} · ${formatTimestamp(lastModified.value.at)}`;
});
const lifecycleLabel = computed(() => {
  if (dirty.value) return '存在未保存的草稿修改';
  if (applicationState.value === 'applying') return `正在应用版本 ${application.value?.version ?? savedVersion.value}。`;
  if (applicationState.value === 'active') return '当前保存版本已生效';
  if (applicationState.value === 'failed') return '已保存版本应用失败';
  if (applicationState.value === 'saved') return '草稿已保存，尚未应用';
  return '草稿与已保存版本一致';
});
const applicationFailureLabel = computed(() => `应用失败：${application.value?.message || '运行系统未接受该保存版本'}`);
const applicationAuditLabel = computed(() => {
  if (!application.value) return '';
  return `应用记录：${application.value.actor} · ${formatTimestamp(application.value.at)}`;
});
const applyActionLabel = computed(() => {
  if (isApplying.value) return `正在应用版本 ${application.value?.version ?? savedVersion.value}。`;
  if (!dirty.value && applicationState.value === 'failed') return `重试应用版本 ${savedVersion.value}`;
  return '保存并应用';
});

const cloneDraft = (source) => ({
  provider_type: source.provider_type,
  model: source.model,
  service_url: source.service_url
});

const clearFieldErrors = () => {
  Object.keys(fieldErrors).forEach((field) => delete fieldErrors[field]);
};

const applyDraftResponse = (data) => {
  const nextDraft = cloneDraft(data.draft);
  Object.assign(draft, nextDraft);
  savedDraft.value = cloneDraft(nextDraft);
  providerApiKey.value = '';
  providerApiKeyConfigured.value = Boolean(data.draft.provider_api_key?.configured);
  savedVersion.value = data.saved_version;
  activeVersion.value = data.active_version;
  lastModified.value = data.last_modified;
  applicationState.value = data.application_state || 'draft_only';
  application.value = data.application || null;
  clearFieldErrors();
};

const isChanged = (field) => draft[field] !== savedDraft.value[field];

const loadDraft = async () => {
  loading.value = true;
  loadError.value = '';
  saveError.value = '';
  applicationError.value = '';
  try {
    applyDraftResponse(await apiAdapter.getSystemSettingsDraft());
    if (isApplying.value) scheduleApplicationRefresh();
  } catch (error) {
    loadError.value = error.status === 404 ? '服务端尚未开启系统设置草稿。' : '加载系统设置草稿失败，请重新加载。';
  } finally {
    loading.value = false;
  }
};

const resetDraft = () => {
  Object.assign(draft, cloneDraft(savedDraft.value));
  providerApiKey.value = '';
  saveError.value = '';
  applicationError.value = '';
  clearFieldErrors();
};

const applySavedVersion = async (version) => {
  loading.value = true;
  applicationError.value = '';
  try {
    applyDraftResponse(await apiAdapter.applySystemSettingsVersion(version));
    if (isApplying.value) scheduleApplicationRefresh();
  } catch (error) {
    applicationError.value = error.detail?.fields
      ? `应用未开始：${Object.values(error.detail.fields).join('；')}`
      : '应用未开始，请刷新后重试。';
  } finally {
    loading.value = false;
  }
};

const saveAndApply = async () => {
  if (!dirty.value && applicationState.value === 'failed' && savedVersion.value !== null) {
    await applySavedVersion(savedVersion.value);
    return;
  }
  if (!dirty.value) return;

  loading.value = true;
  saveError.value = '';
  applicationError.value = '';
  clearFieldErrors();
  try {
    const saved = await apiAdapter.saveSystemSettingsDraft({
      ...cloneDraft(draft),
      provider_api_key: providerApiKey.value || null
    });
    applyDraftResponse(saved);
  } catch (error) {
    Object.assign(fieldErrors, error.detail?.fields || {});
    saveError.value = '草稿未保存，请修正标记字段后重试。';
    loading.value = false;
    return;
  }
  loading.value = false;
  await applySavedVersion(savedVersion.value);
};

const scheduleApplicationRefresh = () => {
  if (applicationRefreshTimer !== null) window.clearTimeout(applicationRefreshTimer);
  applicationRefreshTimer = window.setTimeout(async () => {
    await loadDraft();
  }, 350);
};

const formatTimestamp = (value) => {
  const timestamp = new Date(value);
  if (Number.isNaN(timestamp.getTime())) return '时间不可用';
  return new Intl.DateTimeFormat('zh-CN', { dateStyle: 'medium', timeStyle: 'short', hour12: false }).format(timestamp);
};

const updateViewport = () => {
  isDesktop.value = window.innerWidth >= DESKTOP_MIN_WIDTH;
};

onMounted(() => {
  window.addEventListener('resize', updateViewport);
  loadDraft();
});

onBeforeUnmount(() => {
  window.removeEventListener('resize', updateViewport);
  if (applicationRefreshTimer !== null) window.clearTimeout(applicationRefreshTimer);
});
</script>

<style scoped>
.system-settings {
  width: min(1100px, 100%);
  margin: 0 auto;
}

.system-settings__header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 24px;
  padding-bottom: 28px;
  border-bottom: 1px solid var(--color-rule);
}

.system-settings__eyebrow {
  margin: 0 0 8px;
  color: var(--color-moss);
  font-size: 12px;
  font-weight: 700;
}

.system-settings h1,
.system-settings h2,
.system-settings p {
  margin-top: 0;
}

.system-settings h1,
.system-settings h2 {
  font-family: var(--font-display);
  font-weight: 600;
}

.system-settings h1 {
  margin-bottom: 10px;
  font-size: 26px;
  color: var(--color-ink);
}

.system-settings h2 {
  margin-bottom: 5px;
  font-size: 18px;
  color: var(--color-ink);
}

.system-settings__description,
.system-settings__section-heading p {
  margin-bottom: 0;
  color: var(--color-ink-soft);
  font-size: 14px;
  line-height: 1.6;
}

.system-settings__form {
  padding-bottom: 28px;
}

.system-settings__section {
  display: grid;
  grid-template-columns: minmax(180px, 0.7fr) minmax(0, 1.5fr);
  gap: 40px;
  padding: 32px 0;
  border-bottom: 1px solid var(--color-rule);
}

.system-settings__fields {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 20px;
}

.system-settings__field {
  display: grid;
  gap: 7px;
  min-width: 0;
  color: var(--color-ink-soft);
  font-size: 13px;
  font-weight: 600;
}

.system-settings__field input,
.system-settings__field select {
  box-sizing: border-box;
  width: 100%;
  min-height: 40px;
  border: 1px solid var(--line-strong);
  border-radius: var(--radius-control);
  background: var(--color-paper-raised);
  color: var(--color-ink);
  font: inherit;
  font-weight: 400;
  padding: 8px 10px;
}

.system-settings__field input:focus-visible,
.system-settings__field select:focus-visible,
.system-settings__state-actions button:focus-visible,
.system-settings__error button:focus-visible {
  outline: 3px solid var(--color-focus);
  outline-offset: 2px;
}

.system-settings__field--changed input,
.system-settings__field--changed select {
  border-color: var(--color-copper);
}

.system-settings__field input:not(:disabled):hover,
.system-settings__field select:not(:disabled):hover {
  border-color: var(--color-copper);
}

.system-settings__field input:not(:disabled):active,
.system-settings__field select:not(:disabled):active {
  background: var(--color-paper-muted);
}

.system-settings__field small {
  color: var(--color-ink-soft);
  font-size: 12px;
  font-weight: 400;
  line-height: 1.4;
}

.system-settings__field-error {
  color: var(--color-danger) !important;
}

.system-settings__state-bar {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 14px 28px;
  margin-top: 24px;
  padding: 16px 18px;
  border: 1px solid var(--line-strong);
  border-radius: var(--radius-control);
  background: var(--color-paper-raised);
  box-shadow: 0 -4px 14px rgb(48 40 30 / 6%);
}

.system-settings__state-copy,
.system-settings__state-actions {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 9px 14px;
}

.system-settings__state-copy strong {
  color: var(--color-ink);
  font-size: 14px;
}

.system-settings__state-copy span {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
}

.system-settings__state-copy span,
.system-settings__state-bar p {
  color: var(--color-ink-soft);
  font-size: 12px;
}

.system-settings__state-bar p {
  grid-column: 1 / -1;
  margin-bottom: 0;
}

.system-settings__state-actions button,
.system-settings__error button {
  min-height: 36px;
  border: 1px solid var(--line-strong);
  border-radius: var(--radius-control);
  background: transparent;
  color: var(--color-ink);
  cursor: pointer;
  font: inherit;
  font-size: 13px;
  padding: 7px 10px;
}

.system-settings__state-actions button:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

.system-settings__state-actions button:not(:disabled):hover,
.system-settings__error button:hover {
  border-color: var(--color-copper);
  color: var(--color-copper-strong);
}

.system-settings__state-actions button:not(:disabled):active,
.system-settings__error button:active {
  background: var(--color-paper-muted);
}

.system-settings__state-actions .system-settings__save {
  border-color: var(--color-copper);
  background: var(--color-copper);
  color: var(--color-paper-raised);
}

.system-settings__state-actions .system-settings__save:not(:disabled):hover,
.system-settings__state-actions .system-settings__save:not(:disabled):active {
  border-color: var(--color-copper-strong);
  background: var(--color-copper-strong);
  color: var(--color-paper-raised);
}

.system-settings__error,
.system-settings__success,
.system-settings__desktop-notice {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 20px 0 0;
  padding: 11px 13px;
  border-left: 3px solid var(--color-warning);
  background: var(--color-warning-soft);
  color: var(--color-warning);
  font-size: 13px;
}

.system-settings__success {
  border-color: var(--color-moss);
  background: var(--color-moss-soft);
  color: var(--color-moss);
}

.system-settings__desktop-notice p {
  margin-bottom: 0;
}

@media (max-width: 900px) {
  .system-settings__section {
    grid-template-columns: 1fr;
    gap: 16px;
  }
}

@media (max-width: 640px) {
  .system-settings__header {
    padding-bottom: 20px;
  }
}
</style>
