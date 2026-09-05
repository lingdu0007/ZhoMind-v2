<template>
  <section class="reviewed-bundles" aria-labelledby="reviewed-bundles-title">
    <header class="reviewed-bundles__header">
      <div>
        <p class="reviewed-bundles__eyebrow">知识库运维</p>
        <h1 id="reviewed-bundles-title">Reviewed Release Bundles</h1>
        <p class="reviewed-bundles__description">导入已批准的 editorial export，并跟踪其独立 Candidate Build。</p>
      </div>
      <div v-if="isDesktop" class="reviewed-bundles__header-actions">
        <button type="button" :disabled="loading" @click="openImportDialog">
          <FileUp :size="16" aria-hidden="true" />
          <span>导入 Bundle</span>
        </button>
        <button type="button" :disabled="loading" @click="loadBundles">
          <RefreshCw :size="16" :class="{ 'reviewed-bundles__refresh-icon--spinning': loading }" aria-hidden="true" />
          <span>{{ loading ? '正在刷新' : '刷新' }}</span>
        </button>
      </div>
    </header>

    <div v-if="!isDesktop" class="reviewed-bundles__desktop-notice" role="status">
      <Monitor :size="18" aria-hidden="true" />
      <p>Reviewed Release Bundle 管理当前仅支持桌面工作区。</p>
    </div>

    <template v-else>
      <p v-if="listError" class="reviewed-bundles__error" role="alert">
        <span>{{ listError }}</span>
        <button type="button" @click="loadBundles">重新加载</button>
      </p>
      <p v-if="actionMessage" class="reviewed-bundles__success" role="status">{{ actionMessage }}</p>
      <p v-if="actionError" class="reviewed-bundles__error" role="alert">{{ actionError }}</p>

      <div class="reviewed-bundles__workspace" :aria-busy="loading">
        <section class="reviewed-bundles__inventory" aria-labelledby="reviewed-bundle-list-title">
          <header class="reviewed-bundles__section-header">
            <h2 id="reviewed-bundle-list-title">已导入 Bundle</h2>
            <p aria-live="polite">{{ bundles.length }} 个 Bundle</p>
          </header>

          <div class="reviewed-bundles__table-wrap">
            <table>
              <caption class="sr-only">Reviewed Release Bundle 列表</caption>
              <thead>
                <tr>
                  <th scope="col">Bundle</th>
                  <th scope="col">Editorial revision</th>
                  <th scope="col">导出时间</th>
                  <th scope="col">项目</th>
                </tr>
              </thead>
              <tbody>
                <tr v-if="loading && !bundles.length">
                  <td colspan="4" class="reviewed-bundles__state">正在加载 Bundle...</td>
                </tr>
                <tr v-else-if="!bundles.length">
                  <td colspan="4" class="reviewed-bundles__state">尚未导入 Reviewed Release Bundle。</td>
                </tr>
                <tr
                  v-for="bundle in bundles"
                  :key="bundle.bundle_id"
                  :class="{ 'reviewed-bundles__row--selected': selectedBundleId === bundle.bundle_id }"
                >
                  <td>
                    <button
                      type="button"
                      class="reviewed-bundles__bundle-select"
                      :aria-pressed="selectedBundleId === bundle.bundle_id"
                      @click="selectBundle(bundle.bundle_id)"
                    >
                      {{ bundle.bundle_id }}
                    </button>
                  </td>
                  <td class="reviewed-bundles__identifier">{{ bundle.editorial_source_revision || '-' }}</td>
                  <td class="reviewed-bundles__timestamp">{{ formatTime(bundle.exported_at) }}</td>
                  <td>{{ bundle.items?.length || 0 }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        <section class="reviewed-bundles__detail" aria-labelledby="reviewed-bundle-detail-title">
          <template v-if="selectedBundle">
            <header class="reviewed-bundles__section-header">
              <div>
                <p class="reviewed-bundles__detail-eyebrow">不可变导入记录</p>
                <h2 id="reviewed-bundle-detail-title">{{ selectedBundle.bundle_id }}</h2>
              </div>
              <span class="reviewed-bundles__status" :class="`reviewed-bundles__status--${bundleStateTone(selectedBundle.state)}`">
                {{ bundleStateLabel(selectedBundle.state) }}
              </span>
            </header>

            <dl class="reviewed-bundles__facts">
              <div>
                <dt>Schema version</dt>
                <dd>{{ selectedBundle.schema_version || '-' }}</dd>
              </div>
              <div>
                <dt>Editorial source revision</dt>
                <dd class="reviewed-bundles__identifier">{{ selectedBundle.editorial_source_revision || '-' }}</dd>
              </div>
              <div>
                <dt>导出时间</dt>
                <dd>{{ formatTime(selectedBundle.exported_at) }}</dd>
              </div>
              <div>
                <dt>Bundle SHA-256</dt>
                <dd class="reviewed-bundles__identifier">{{ selectedBundle.bundle_sha256 || '-' }}</dd>
              </div>
            </dl>

            <div class="reviewed-bundles__item-table-wrap">
              <table>
                <caption class="sr-only">Bundle 项目和 Candidate Build 状态</caption>
                <thead>
                  <tr>
                    <th scope="col">项目</th>
                    <th scope="col">操作</th>
                    <th scope="col">状态</th>
                    <th scope="col">输入 SHA-256</th>
                    <th scope="col">Candidate Build</th>
                    <th scope="col">允许的下一步</th>
                    <th scope="col"><span class="sr-only">任务操作</span></th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="item in selectedBundle.items" :key="item.bundle_item_id">
                    <td>
                      <p class="reviewed-bundles__identifier">{{ item.bundle_item_id }}</p>
                      <p class="reviewed-bundles__entry">{{ item.entry_identity }}</p>
                    </td>
                    <td>{{ operationLabel(item.operation) }}</td>
                    <td>
                      <span class="reviewed-bundles__status" :class="`reviewed-bundles__status--${itemStateTone(item.state)}`">
                        {{ itemStateLabel(item.state) }}
                      </span>
                      <p v-if="item.failure_reason" class="reviewed-bundles__failure">
                        {{ item.failure_reason.field }}: {{ item.failure_reason.message }}
                      </p>
                    </td>
                    <td class="reviewed-bundles__identifier">
                      <p>{{ item.artifact_sha256 || '-' }}</p>
                      <p class="reviewed-bundles__item-hash">{{ item.bundle_item_sha256 || '-' }}</p>
                    </td>
                    <td>
                      <template v-if="item.job_id">
                        <p class="reviewed-bundles__identifier">{{ item.job_id }}</p>
                        <p v-if="jobFor(item)" class="reviewed-bundles__job-line">
                          {{ jobStatusLabel(jobFor(item).status, jobFor(item).stage) }} · {{ jobFor(item).progress }}% · {{ jobFor(item).attempt }} 次
                        </p>
                        <p v-if="jobFor(item)?.failure_reason" class="reviewed-bundles__failure">
                          {{ jobFor(item).failure_reason.code }}: {{ jobFor(item).failure_reason.message }}
                        </p>
                        <p v-if="jobFor(item)?.derived_cleanup_pending" class="reviewed-bundles__failure">
                          派生数据仍待协调
                        </p>
                        <p v-if="jobFor(item)" class="reviewed-bundles__job-time">
                          更新于 {{ formatTime(jobFor(item).updated_at) }}
                        </p>
                      </template>
                      <span v-else>-</span>
                    </td>
                    <td class="reviewed-bundles__next-action">{{ nextActionLabel(nextActionFor(item)) }}</td>
                    <td class="reviewed-bundles__actions">
                      <button
                        v-if="canDispatch(item)"
                        type="button"
                        :disabled="Boolean(actionLoading[item.job_id])"
                        :aria-label="`开始 Candidate Build ${item.job_id}`"
                        title="开始 Candidate Build"
                        @click="dispatchJob(item.job_id)"
                      >
                        <Play :size="16" aria-hidden="true" />
                      </button>
                      <button
                        v-if="canRetry(item)"
                        type="button"
                        :disabled="Boolean(actionLoading[item.job_id])"
                        :aria-label="`重试 Candidate Build ${item.job_id}`"
                        title="重试 Candidate Build"
                        @click="retryJob(item.job_id)"
                      >
                        <RotateCcw :size="16" aria-hidden="true" />
                      </button>
                      <button
                        v-if="canCancel(item)"
                        type="button"
                        class="reviewed-bundles__cancel"
                        :disabled="Boolean(actionLoading[item.job_id])"
                        :aria-label="`取消 Candidate Build ${item.job_id}`"
                        title="取消 Candidate Build"
                        @click="cancelJob(item.job_id)"
                      >
                        <XCircle :size="16" aria-hidden="true" />
                      </button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </template>

          <p v-else class="reviewed-bundles__detail-empty" role="status">选择一个 Bundle 以查看其不可变输入和任务状态。</p>
        </section>
      </div>
    </template>

    <el-dialog
      v-model="importDialogVisible"
      class="reviewed-bundles__import-dialog"
      width="min(92vw, 780px)"
      :close-on-click-modal="false"
      @closed="resetImport"
    >
      <template #header>
        <div>
          <p class="reviewed-bundles__detail-eyebrow">仅接受已批准的 editorial export</p>
          <h2>导入 Reviewed Release Bundle</h2>
        </div>
      </template>

      <div class="reviewed-bundles__import-form">
        <div class="reviewed-bundles__import-tools">
          <input ref="bundleFileInput" class="sr-only" type="file" accept="application/json,.json" @change="readBundleFile" />
          <button type="button" :disabled="importLoading" @click="bundleFileInput?.click()">
            <FileText :size="16" aria-hidden="true" />
            <span>选择 JSON 文件</span>
          </button>
          <span v-if="selectedFilename" class="reviewed-bundles__selected-file">{{ selectedFilename }}</span>
        </div>
        <label for="reviewed-bundle-manifest">
          <span>Bundle manifest</span>
          <textarea
            id="reviewed-bundle-manifest"
            v-model="manifestText"
            :disabled="importLoading"
            spellcheck="false"
            placeholder="粘贴 reviewed_release_bundle/v1 JSON"
          />
        </label>
        <p v-if="importError" class="reviewed-bundles__error" role="alert">{{ importError }}</p>
      </div>

      <template #footer>
        <button type="button" class="reviewed-bundles__dialog-button" :disabled="importLoading" @click="importDialogVisible = false">
          取消
        </button>
        <button
          type="button"
          class="reviewed-bundles__dialog-button reviewed-bundles__dialog-button--primary"
          :disabled="importLoading || !manifestText.trim()"
          @click="importBundle"
        >
          <FileUp v-if="!importLoading" :size="16" aria-hidden="true" />
          <RefreshCw v-else :size="16" class="reviewed-bundles__refresh-icon--spinning" aria-hidden="true" />
          <span>{{ importLoading ? '正在导入' : '导入 Bundle' }}</span>
        </button>
      </template>
    </el-dialog>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { FileText, FileUp, Monitor, Play, RefreshCw, RotateCcw, XCircle } from 'lucide-vue-next';
import { apiAdapter } from '../api/adapters';

const POLL_DELAY_MS = 1000;

const bundles = ref([]);
const selectedBundleId = ref('');
const selectedBundle = ref(null);
const jobsById = ref({});
const loading = ref(false);
const listError = ref('');
const actionMessage = ref('');
const actionError = ref('');
const actionLoading = ref({});
const isDesktop = ref(true);
const isActive = ref(true);
const importDialogVisible = ref(false);
const manifestText = ref('');
const importLoading = ref(false);
const importError = ref('');
const selectedFilename = ref('');
const bundleFileInput = ref(null);
let pollTimer = null;

const bundleStateLabel = (state) =>
  ({
    received: '已接收',
    validating: '验证中',
    validated: '已验证',
    processing: '处理中',
    completed: '已完成',
    completed_with_rejections: '已完成，存在拒绝项'
  })[state] || state || '-';
const bundleStateTone = (state) => {
  if (state === 'completed') return 'success';
  if (state === 'completed_with_rejections') return 'danger';
  return 'neutral';
};
const operationLabel = (operation) =>
  ({
    create: '新建 (create)',
    replace: '替换计划 (replace)',
    no_op: '无变更 (no-op)',
    proposed_withdrawal: '撤回提案'
  })[operation] || operation || '-';
const itemStateLabel = (state) =>
  ({
    admitted: '已接纳',
    rejected: '已拒绝',
    no_op: '无变更',
    proposed_withdrawal: '撤回提案'
  })[state] || state || '-';
const itemStateTone = (state) => {
  if (state === 'admitted') return 'success';
  if (state === 'rejected') return 'danger';
  return 'neutral';
};
const jobStageLabel = (stage) =>
  ({
    queued: '排队中',
    parsing: '解析中',
    chunking: '分块中',
    indexing: '索引中'
  })[stage] || stage || '-';
const jobStatusLabel = (status, stage) =>
  ({
    candidate_ready: 'Candidate 已就绪',
    failed: '失败',
    canceled: '已取消',
    interrupted_retryable: '已中断，可重试',
    superseded: '已 supersede'
  })[status] || jobStageLabel(stage);
const nextActionLabel = (action) =>
  ({
    dispatch_candidate_build: '开始 Candidate Build',
    await_candidate_build: '等待构建',
    cancel_or_await_candidate_build: '等待或取消',
    await_candidate_inspection: '等待后续 inspection',
    correct_item_in_new_bundle: '在新 Bundle 中修正',
    review_explicit_no_op: '复核无变更',
    requires_t04_publication_workflow: '等待后续 publication workflow',
    retry_fixed_inputs: '可用固定输入重试',
    reconcile_derived_data_then_retry: '先协调派生数据',
    reconcile_derived_data: '先协调派生数据',
    import_new_bundle: '需要新 Bundle',
    none: '无可执行操作'
  })[action] || action || '-';

const formatTime = (value) => {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date
    .toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    })
    .replaceAll('/', '-');
};

const structuredReasonMessage = (detail) => {
  if (!Array.isArray(detail?.reasons)) return '';
  return detail.reasons
    .slice(0, 4)
    .map((reason) => {
      if (!reason || typeof reason !== 'object') return '';
      const field = typeof reason.field === 'string' ? reason.field : '';
      const message = typeof reason.message === 'string' ? reason.message : '';
      return field && message ? `${field}: ${message}` : message || field;
    })
    .filter(Boolean)
    .join('；');
};

const friendlyError = (error, fallback) => {
  if (error?.status === 401) return '登录状态已失效，请重新登录。';
  if (error?.status === 403) return '当前账户无权管理 Reviewed Release Bundle。';
  const structuredReason = structuredReasonMessage(error?.detail);
  if (structuredReason) return structuredReason;
  if (error?.message && !/^Request failed with status code \d+$/.test(error.message)) return error.message;
  return fallback;
};

const jobFor = (item) => (item?.job_id ? jobsById.value[item.job_id] : null);
const nextActionFor = (item) => jobFor(item)?.allowed_next_action || item?.allowed_next_action;
const canDispatch = (item) =>
  jobFor(item)?.status === 'queued' && jobFor(item)?.allowed_next_action === 'dispatch_candidate_build';
const canRetry = (item) =>
  ['retry_fixed_inputs', 'reconcile_derived_data_then_retry'].includes(jobFor(item)?.allowed_next_action);
const canCancel = (item) => jobFor(item)?.allowed_next_action === 'cancel_or_await_candidate_build';
const hasActiveSelectedJob = computed(
  () => selectedBundle.value?.items?.some((item) => canCancel(item) && !canDispatch(item)) || false
);

const mergeJob = (job) => {
  if (job?.job_id) jobsById.value = { ...jobsById.value, [job.job_id]: job };
};

const applyActionResult = (job, successMessage, failureFallback) => {
  mergeJob(job);
  if (job?.status === 'failed') {
    const reason = job.failure_reason;
    const detail = reason?.code && reason?.message ? `${reason.code}: ${reason.message}` : reason?.message || '';
    actionError.value = detail ? `${failureFallback} ${detail}` : failureFallback;
    return false;
  }
  actionMessage.value = successMessage;
  return true;
};

const refreshSelectedBundle = async () => {
  const bundleId = selectedBundleId.value;
  if (!bundleId) return;
  const refreshed = await apiAdapter.getReviewedReleaseBundle(bundleId);
  if (selectedBundleId.value === bundleId) selectedBundle.value = refreshed;
};

const loadSelectedJobs = async () => {
  if (!selectedBundle.value?.items?.length) return;
  const jobs = await Promise.all(
    selectedBundle.value.items
      .map((item) => item.job_id)
      .filter(Boolean)
      .map((jobId) => apiAdapter.getReviewedBundleJob(jobId))
  );
  jobs.forEach(mergeJob);
  await refreshSelectedBundle();
};

const schedulePoll = () => {
  if (pollTimer) clearTimeout(pollTimer);
  if (!isActive.value || !isDesktop.value || !hasActiveSelectedJob.value) return;
  pollTimer = setTimeout(async () => {
    try {
      await loadSelectedJobs();
    } catch {
      // The next explicit refresh remains available if transient polling fails.
    } finally {
      schedulePoll();
    }
  }, POLL_DELAY_MS);
};

const selectBundle = async (bundleId) => {
  selectedBundleId.value = bundleId;
  actionError.value = '';
  try {
    selectedBundle.value = await apiAdapter.getReviewedReleaseBundle(bundleId);
    await loadSelectedJobs();
    schedulePoll();
  } catch (error) {
    actionError.value = friendlyError(error, '加载 Bundle 详情失败。');
  }
};

const loadBundles = async () => {
  if (!isDesktop.value) return;
  loading.value = true;
  listError.value = '';
  try {
    const data = await apiAdapter.listReviewedReleaseBundles();
    bundles.value = data?.items || [];
    const nextId = selectedBundleId.value || bundles.value[0]?.bundle_id || '';
    if (nextId && bundles.value.some((bundle) => bundle.bundle_id === nextId)) await selectBundle(nextId);
    else {
      selectedBundleId.value = '';
      selectedBundle.value = null;
      jobsById.value = {};
    }
  } catch (error) {
    listError.value = friendlyError(error, '加载 Reviewed Release Bundle 失败，请重新加载。');
  } finally {
    loading.value = false;
  }
};

const openImportDialog = () => {
  importError.value = '';
  importDialogVisible.value = true;
};

const resetImport = () => {
  manifestText.value = '';
  selectedFilename.value = '';
  importError.value = '';
  importLoading.value = false;
  if (bundleFileInput.value) bundleFileInput.value.value = '';
};

const readBundleFile = async (event) => {
  const [file] = event.target.files || [];
  if (!file) return;
  try {
    manifestText.value = await file.text();
    selectedFilename.value = file.name;
    importError.value = '';
  } catch {
    importError.value = '无法读取所选 Bundle 文件。';
  }
};

const importBundle = async () => {
  importError.value = '';
  actionError.value = '';
  actionMessage.value = '';
  let manifest;
  try {
    manifest = JSON.parse(manifestText.value);
  } catch {
    importError.value = 'Bundle manifest 不是有效 JSON。';
    return;
  }

  importLoading.value = true;
  try {
    const imported = await apiAdapter.importReviewedReleaseBundle(manifest);
    importDialogVisible.value = false;
    actionMessage.value = `已导入 Bundle ${imported.bundle_id}。`;
    await loadBundles();
    await selectBundle(imported.bundle_id);
  } catch (error) {
    importError.value = friendlyError(error, '导入 Bundle 失败。');
  } finally {
    importLoading.value = false;
  }
};

const dispatchJob = async (jobId) => {
  actionMessage.value = '';
  actionError.value = '';
  actionLoading.value[jobId] = true;
  try {
    const job = await apiAdapter.dispatchReviewedBundleJob(jobId);
    if (applyActionResult(job, `Candidate Build ${jobId} 已进入队列。`, `Candidate Build ${jobId} 未能进入队列。`)) {
      schedulePoll();
    }
  } catch (error) {
    actionError.value = friendlyError(error, `开始 Candidate Build ${jobId} 失败。`);
  } finally {
    delete actionLoading.value[jobId];
  }
};

const retryJob = async (jobId) => {
  actionMessage.value = '';
  actionError.value = '';
  actionLoading.value[jobId] = true;
  try {
    const job = await apiAdapter.retryReviewedBundleJob(jobId);
    if (applyActionResult(job, `Candidate Build ${jobId} 已进入重试队列。`, `Candidate Build ${jobId} 未能进入重试队列。`)) {
      schedulePoll();
    }
  } catch (error) {
    actionError.value = friendlyError(error, `重试 Candidate Build ${jobId} 失败。`);
  } finally {
    delete actionLoading.value[jobId];
  }
};

const cancelJob = async (jobId) => {
  actionMessage.value = '';
  actionError.value = '';
  actionLoading.value[jobId] = true;
  try {
    const job = await apiAdapter.cancelReviewedBundleJob(jobId);
    if (applyActionResult(job, `Candidate Build ${jobId} 的取消结果已由服务端确认。`, `Candidate Build ${jobId} 的取消未获服务端确认。`)) {
      schedulePoll();
    }
  } catch (error) {
    actionError.value = friendlyError(error, `取消 Candidate Build ${jobId} 失败。`);
  } finally {
    delete actionLoading.value[jobId];
  }
};

const updateViewportScope = () => {
  const wasDesktop = isDesktop.value;
  isDesktop.value = window.innerWidth >= 768;
  if (!isDesktop.value && pollTimer) clearTimeout(pollTimer);
  if (!wasDesktop && isDesktop.value) loadBundles();
};

onMounted(() => {
  updateViewportScope();
  window.addEventListener('resize', updateViewportScope);
  if (isDesktop.value) loadBundles();
});

onBeforeUnmount(() => {
  isActive.value = false;
  if (pollTimer) clearTimeout(pollTimer);
  window.removeEventListener('resize', updateViewportScope);
});
</script>

<style scoped>
.reviewed-bundles { max-width: 1440px; margin: 0 auto; }
.reviewed-bundles__header, .reviewed-bundles__section-header { display: flex; align-items: flex-start; justify-content: space-between; gap: var(--space-5); }
.reviewed-bundles__header { padding-bottom: var(--space-5); border-bottom: 1px solid var(--color-rule); }
.reviewed-bundles__eyebrow, .reviewed-bundles__detail-eyebrow { margin: 0 0 var(--space-2); color: var(--color-moss); font-size: 12px; font-weight: 600; }
.reviewed-bundles h1, .reviewed-bundles h2 { margin: 0; font-family: var(--font-display); font-weight: 600; }
.reviewed-bundles h1 { font-size: 26px; line-height: 1.3; }
.reviewed-bundles h2 { font-size: 18px; line-height: 1.4; }
.reviewed-bundles__description { max-width: 640px; margin: var(--space-2) 0 0; color: var(--color-ink-soft); font-size: 14px; line-height: 1.7; }
.reviewed-bundles__header-actions, .reviewed-bundles__import-tools, .reviewed-bundles__actions { display: flex; align-items: center; gap: var(--space-2); }
.reviewed-bundles__header-actions button, .reviewed-bundles__import-tools button, .reviewed-bundles__dialog-button, .reviewed-bundles__error button, .reviewed-bundles__actions button { display: inline-flex; min-height: 32px; align-items: center; justify-content: center; gap: 6px; border: 1px solid var(--color-rule); border-radius: var(--radius-control); background: var(--color-paper-raised); color: var(--color-ink); font: inherit; font-size: 13px; cursor: pointer; }
.reviewed-bundles__header-actions button, .reviewed-bundles__import-tools button, .reviewed-bundles__dialog-button { padding: 0 var(--space-3); }
.reviewed-bundles__header-actions button:disabled, .reviewed-bundles__import-tools button:disabled, .reviewed-bundles__dialog-button:disabled, .reviewed-bundles__actions button:disabled { cursor: wait; opacity: 0.65; }
.reviewed-bundles__header-actions button:not(:disabled):hover, .reviewed-bundles__import-tools button:not(:disabled):hover, .reviewed-bundles__dialog-button:not(:disabled):hover, .reviewed-bundles__actions button:not(:disabled):hover { border-color: var(--color-copper); color: var(--color-copper-strong); }
.reviewed-bundles__refresh-icon--spinning { animation: reviewed-bundles-spin 0.9s linear infinite; }
.reviewed-bundles__desktop-notice, .reviewed-bundles__error, .reviewed-bundles__success { display: flex; align-items: center; gap: var(--space-3); margin: var(--space-5) 0 0; padding: var(--space-3) var(--space-4); border-left: 3px solid var(--color-warning); background: var(--color-warning-soft); color: var(--color-warning); font-size: 13px; line-height: 1.5; }
.reviewed-bundles__desktop-notice p, .reviewed-bundles__error, .reviewed-bundles__success { margin: 0; }
.reviewed-bundles__error { border-left-color: var(--color-danger); background: var(--color-danger-soft); color: var(--color-danger); }
.reviewed-bundles__success { border-left-color: var(--color-moss); background: var(--color-moss-soft); color: var(--color-moss); }
.reviewed-bundles__error button { min-height: 28px; margin-left: auto; padding: 0 var(--space-2); border-color: currentColor; background: transparent; color: inherit; }
.reviewed-bundles__workspace { display: grid; grid-template-columns: minmax(310px, 0.8fr) minmax(0, 1.7fr); min-height: 540px; margin-top: var(--space-5); border-top: 1px solid var(--color-rule); border-bottom: 1px solid var(--color-rule); }
.reviewed-bundles__inventory { border-right: 1px solid var(--color-rule); background: var(--color-paper-muted); }
.reviewed-bundles__inventory, .reviewed-bundles__detail { min-width: 0; padding: var(--space-4); }
.reviewed-bundles__section-header { min-height: 42px; margin-bottom: var(--space-3); }
.reviewed-bundles__section-header > p { margin: 4px 0 0; color: var(--color-ink-soft); font-size: 12px; }
.reviewed-bundles__table-wrap, .reviewed-bundles__item-table-wrap { overflow: auto; border-top: 1px solid var(--color-rule); border-bottom: 1px solid var(--color-rule); background: var(--color-paper-raised); }
.reviewed-bundles table { width: 100%; border-collapse: collapse; table-layout: fixed; }
.reviewed-bundles__inventory table { min-width: 610px; }
.reviewed-bundles__item-table-wrap table { min-width: 1180px; }
.reviewed-bundles th, .reviewed-bundles td { padding: 11px 10px; border-bottom: 1px solid var(--color-rule); color: var(--color-ink); font-size: 12px; line-height: 1.5; text-align: left; vertical-align: top; }
.reviewed-bundles th { position: sticky; top: 0; z-index: 1; background: var(--color-paper-muted); color: var(--color-ink-soft); font-size: 11px; font-weight: 600; }
.reviewed-bundles tbody tr:last-child td { border-bottom: 0; }
.reviewed-bundles__row--selected td { background: var(--color-moss-soft); }
.reviewed-bundles__bundle-select { width: 100%; padding: 0; border: 0; background: transparent; color: var(--color-ink); font: inherit; font-family: var(--font-mono); font-size: 12px; text-align: left; cursor: pointer; overflow-wrap: anywhere; }
.reviewed-bundles__bundle-select:hover { color: var(--color-copper-strong); }
.reviewed-bundles__identifier { margin: 0; overflow-wrap: anywhere; font-family: var(--font-mono); font-size: 11px; }
.reviewed-bundles__timestamp, .reviewed-bundles__job-time { color: var(--color-ink-soft); font-size: 11px; }
.reviewed-bundles__entry, .reviewed-bundles__job-line, .reviewed-bundles__failure { margin: var(--space-1) 0 0; overflow-wrap: anywhere; }
.reviewed-bundles__entry, .reviewed-bundles__job-line { color: var(--color-ink-soft); font-size: 11px; }
.reviewed-bundles__failure { color: var(--color-danger); font-size: 11px; }
.reviewed-bundles__job-time { margin: var(--space-1) 0 0; }
.reviewed-bundles__status { display: inline-flex; min-width: 72px; justify-content: center; padding: 2px 5px; border: 1px solid currentColor; border-radius: 3px; font-size: 11px; white-space: nowrap; }
.reviewed-bundles__status--success { color: var(--color-moss); }
.reviewed-bundles__status--danger { color: var(--color-danger); }
.reviewed-bundles__status--neutral { color: var(--color-ink-soft); }
.reviewed-bundles__facts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0; margin: 0 0 var(--space-5); border-top: 1px solid var(--color-rule); border-bottom: 1px solid var(--color-rule); }
.reviewed-bundles__facts div { min-width: 0; padding: var(--space-3) var(--space-2); border-bottom: 1px solid var(--color-rule); }
.reviewed-bundles__facts div:nth-last-child(-n + 2) { border-bottom: 0; }
.reviewed-bundles__facts dt { color: var(--color-ink-soft); font-size: 11px; }
.reviewed-bundles__facts dd { margin: var(--space-1) 0 0; color: var(--color-ink); font-size: 12px; overflow-wrap: anywhere; }
.reviewed-bundles__next-action { color: var(--color-ink-soft); overflow-wrap: anywhere; }
.reviewed-bundles__actions { justify-content: flex-end; }
.reviewed-bundles__actions button { width: 30px; min-width: 30px; padding: 0; }
.reviewed-bundles__actions .reviewed-bundles__cancel { border-color: var(--color-danger); color: var(--color-danger); }
.reviewed-bundles__detail-empty, .reviewed-bundles__state { color: var(--color-ink-soft); font-size: 13px; text-align: center; }
.reviewed-bundles__detail-empty { margin: 150px 0; }
.reviewed-bundles__state { height: 180px; }
.reviewed-bundles__import-form { display: grid; gap: var(--space-4); }
.reviewed-bundles__selected-file { color: var(--color-ink-soft); font-family: var(--font-mono); font-size: 12px; overflow-wrap: anywhere; }
.reviewed-bundles__import-form label { display: grid; gap: var(--space-2); color: var(--color-ink); font-size: 13px; font-weight: 600; }
.reviewed-bundles__import-form textarea { width: 100%; min-height: 300px; resize: vertical; padding: var(--space-3); border: 1px solid var(--color-rule); border-radius: 3px; background: var(--color-paper-muted); color: var(--color-ink); font-family: var(--font-mono); font-size: 12px; line-height: 1.55; }
.reviewed-bundles__import-form textarea:focus-visible, .reviewed-bundles__bundle-select:focus-visible, .reviewed-bundles button:focus-visible { outline: 2px solid var(--color-focus); outline-offset: 2px; }
.reviewed-bundles__dialog-button--primary { border-color: var(--color-moss); background: var(--color-moss); color: var(--color-paper-raised); }
.reviewed-bundles__dialog-button--primary:not(:disabled):hover { border-color: var(--color-moss); background: var(--color-moss); color: var(--color-paper-raised); opacity: 0.88; }
@keyframes reviewed-bundles-spin { to { transform: rotate(360deg); } }
@media (max-width: 1180px) {
  .reviewed-bundles__workspace { grid-template-columns: minmax(290px, 0.72fr) minmax(0, 1.45fr); }
  .reviewed-bundles__inventory, .reviewed-bundles__detail { padding: var(--space-3); }
}
</style>
