<template>
  <section class="indexing-jobs" aria-labelledby="indexing-jobs-title">
    <header class="indexing-jobs__header">
      <div>
        <p class="indexing-jobs__eyebrow">知识库运维</p>
        <h1 id="indexing-jobs-title">构建任务</h1>
        <p class="indexing-jobs__description">查看异步构建进度，并在服务端允许时停止尚未完成的任务。</p>
      </div>
      <button v-if="isDesktop" class="indexing-jobs__refresh" type="button" :disabled="loading" @click="loadJobs">
        <RefreshCw :size="16" :class="{ 'indexing-jobs__refresh-icon--spinning': loading }" aria-hidden="true" />
        <span>{{ loading ? '正在刷新' : '刷新' }}</span>
      </button>
    </header>

    <div v-if="!isDesktop" class="indexing-jobs__desktop-notice" role="status">
      <Monitor :size="18" aria-hidden="true" />
      <p>构建任务当前仅支持桌面工作区。</p>
    </div>

    <template v-else>
      <div class="indexing-jobs__controls" aria-label="构建任务控制">
        <fieldset class="indexing-jobs__filters">
          <legend>任务状态筛选</legend>
          <label v-for="option in filterOptions" :key="option.value" class="indexing-jobs__filter-option">
            <input v-model="filterMode" type="radio" name="job-status-filter" :value="option.value" />
            <span>{{ option.label }}</span>
          </label>
        </fieldset>
        <p class="indexing-jobs__count" aria-live="polite">显示 {{ filteredJobs.length }} 个任务</p>
      </div>

      <p v-if="focusedJobId" class="indexing-jobs__scope" role="status">
        正在查看任务 {{ focusedJobId }}。
        <button type="button" @click="clearFocus">显示全部任务</button>
      </p>
      <p v-else-if="focusedDocumentId" class="indexing-jobs__scope" role="status">
        正在查看文档 {{ focusedDocumentId }} 的构建任务。
        <button type="button" @click="clearFocus">显示全部任务</button>
      </p>

      <p v-if="listError" class="indexing-jobs__error" role="alert">
        <span>{{ listError }}</span>
        <button type="button" @click="loadJobs">重新加载</button>
      </p>
      <p v-if="actionMessage" class="indexing-jobs__success" role="status">{{ actionMessage }}</p>
      <p v-if="actionError" class="indexing-jobs__error" role="alert">{{ actionError }}</p>

      <div class="indexing-jobs__table-wrap" :aria-busy="loading">
        <table>
          <caption class="sr-only">构建任务列表</caption>
          <thead>
            <tr>
              <th scope="col">任务 ID</th>
              <th scope="col">文档 ID</th>
              <th scope="col">状态</th>
              <th scope="col">阶段</th>
              <th scope="col">进度</th>
              <th scope="col">运行信息</th>
              <th scope="col">更新时间</th>
              <th scope="col"><span class="sr-only">操作</span></th>
            </tr>
          </thead>
          <tbody>
            <tr v-if="loading && !jobs.length">
              <td colspan="8" class="indexing-jobs__state">正在加载构建任务...</td>
            </tr>
            <tr v-else-if="!filteredJobs.length">
              <td colspan="8" class="indexing-jobs__state">{{ emptyStateText }}</td>
            </tr>
            <tr v-for="job in filteredJobs" :key="job.job_id">
              <td class="indexing-jobs__identifier">{{ job.job_id || '-' }}</td>
              <td class="indexing-jobs__identifier">{{ job.document_id || '-' }}</td>
              <td>
                <span class="indexing-jobs__status" :class="`indexing-jobs__status--${jobStatusMeta(job.status).tone}`">
                  {{ jobStatusMeta(job.status).label }}
                </span>
              </td>
              <td>{{ jobStageLabel(job.stage) }}</td>
              <td>
                <div class="indexing-jobs__progress" :aria-label="`任务 ${job.job_id} 进度 ${jobProgress(job)}%`">
                  <span :style="{ width: `${jobProgress(job)}%` }" />
                  <b>{{ jobProgress(job) }}%</b>
                </div>
              </td>
              <td class="indexing-jobs__message">{{ job.message || '-' }}</td>
              <td class="indexing-jobs__timestamp">{{ formatTime(job.updated_at) }}</td>
              <td class="indexing-jobs__action">
                <button
                  v-if="canCancelJob(job)"
                  type="button"
                  :aria-label="`取消任务 ${job.job_id}`"
                  :disabled="Boolean(cancellationLoading[job.job_id])"
                  @click="cancelJob(job)"
                >
                  <XCircle :size="15" aria-hidden="true" />
                  <span>{{ cancellationLoading[job.job_id] ? '正在取消' : '取消' }}</span>
                </button>
                <span v-else class="indexing-jobs__terminal-label">已结束</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </template>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { Monitor, RefreshCw, XCircle } from 'lucide-vue-next';
import { apiAdapter } from '../api/adapters';

const TERMINAL_JOB_STATUSES = ['succeeded', 'failed', 'canceled'];
const ACTIVE_JOB_STATUSES = ['queued', 'running'];
const POLL_DELAY_MS = 800;

const filterOptions = [
  { value: 'all', label: '全部状态' },
  { value: 'active', label: '仅进行中' },
  { value: 'terminal', label: '仅已结束' }
];

const route = useRoute();
const router = useRouter();

const statusMetadata = {
  queued: { label: '排队中 (queued)', tone: 'neutral' },
  running: { label: '执行中 (running)', tone: 'warning' },
  succeeded: { label: '已成功 (succeeded)', tone: 'success' },
  failed: { label: '失败 (failed)', tone: 'danger' },
  canceled: { label: '已取消 (canceled)', tone: 'neutral' }
};

const stageLabels = {
  queued: '排队中 (queued)',
  uploaded: '已上传 (uploaded)',
  parsing: '解析中 (parsing)',
  chunking: '分块中 (chunking)',
  indexing: '索引中 (indexing)',
  completed: '已完成 (completed)',
  failed: '失败 (failed)'
};

const jobs = ref([]);
const filterMode = ref('all');
const loading = ref(false);
const listError = ref('');
const actionMessage = ref('');
const actionError = ref('');
const cancellationLoading = ref({});
const isDesktop = ref(true);
const isActive = ref(true);
const pollingTimers = new Map();

const focusedJobId = computed(() => (typeof route.query.job === 'string' ? route.query.job : ''));
const focusedDocumentId = computed(() => (typeof route.query.document === 'string' ? route.query.document : ''));

const filteredJobs = computed(() => {
  let items = jobs.value;
  if (focusedJobId.value) items = items.filter((job) => job.job_id === focusedJobId.value);
  else if (focusedDocumentId.value) items = items.filter((job) => job.document_id === focusedDocumentId.value);
  if (filterMode.value === 'active') return items.filter((job) => ACTIVE_JOB_STATUSES.includes(job.status));
  if (filterMode.value === 'terminal') return items.filter((job) => TERMINAL_JOB_STATUSES.includes(job.status));
  return items;
});

const emptyStateText = computed(() => {
  if (filterMode.value === 'active') return '当前没有进行中的构建任务。';
  if (filterMode.value === 'terminal') return '当前没有已结束的构建任务。';
  return '当前没有构建任务。';
});

const jobStatusMeta = (status) => statusMetadata[status] || { label: status || '-', tone: 'neutral' };
const jobStageLabel = (stage) => stageLabels[stage] || stage || '-';
const canCancelJob = (job) => ACTIVE_JOB_STATUSES.includes(job?.status);

const jobProgress = (job) => {
  const value = Number(job?.progress);
  if (!Number.isFinite(value)) return 0;
  return Math.min(100, Math.max(0, Math.round(value)));
};

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
      hour12: false
    })
    .replaceAll('/', '-');
};

const getFriendlyError = (error, fallback) => {
  if (error?.status === 401) return '登录状态已失效，请重新登录。';
  if (error?.status === 403) return '当前账户无权查看构建任务。';
  if (error?.message && !/^Request failed with status code \d+$/.test(error.message)) return error.message;
  return fallback;
};

const sortByUpdatedAtDesc = (items) =>
  [...items].sort((left, right) => new Date(right.updated_at || 0).getTime() - new Date(left.updated_at || 0).getTime());

const clearPolling = (jobId) => {
  const timer = pollingTimers.get(jobId);
  if (timer) clearTimeout(timer);
  pollingTimers.delete(jobId);
};

const clearAllPolling = () => {
  pollingTimers.forEach((timer) => clearTimeout(timer));
  pollingTimers.clear();
};

const mergeJob = (job) => {
  if (!job?.job_id) return;
  const index = jobs.value.findIndex((item) => item.job_id === job.job_id);
  if (index >= 0) jobs.value[index] = { ...jobs.value[index], ...job };
  else jobs.value.push(job);
  jobs.value = sortByUpdatedAtDesc(jobs.value);
};

const schedulePoll = (jobId) => {
  if (!isActive.value || !isDesktop.value || !jobId) return;
  clearPolling(jobId);
  pollingTimers.set(jobId, setTimeout(() => pollJob(jobId), POLL_DELAY_MS));
};

const pollJob = async (jobId) => {
  if (!isActive.value || !isDesktop.value) return;

  try {
    const job = await apiAdapter.getDocumentJob(jobId);
    if (!isActive.value) return;
    mergeJob(job);
    if (TERMINAL_JOB_STATUSES.includes(job?.status)) {
      clearPolling(jobId);
      return;
    }
    schedulePoll(jobId);
  } catch (error) {
    clearPolling(jobId);
    if (isActive.value && error?.status !== 404) {
      listError.value = getFriendlyError(error, `任务 ${jobId} 刷新失败，请重新加载。`);
    }
  }
};

const scheduleActiveJobs = () => {
  clearAllPolling();
  jobs.value.filter(canCancelJob).forEach((job) => schedulePoll(job.job_id));
};

const loadJobs = async () => {
  if (!isDesktop.value) return;

  loading.value = true;
  listError.value = '';
  try {
    const data = await apiAdapter.listDocumentJobs({ page: 1, page_size: 100 });
    const loadedJobs = data?.items || [];
    if (focusedJobId.value) {
      const focusedJob = await apiAdapter.getDocumentJob(focusedJobId.value);
      const existingIndex = loadedJobs.findIndex((job) => job.job_id === focusedJob?.job_id);
      if (existingIndex >= 0) loadedJobs[existingIndex] = { ...loadedJobs[existingIndex], ...focusedJob };
      else if (focusedJob?.job_id) loadedJobs.push(focusedJob);
    }
    jobs.value = sortByUpdatedAtDesc(loadedJobs);
    scheduleActiveJobs();
  } catch (error) {
    listError.value = getFriendlyError(error, '加载构建任务失败，请重新加载。');
  } finally {
    loading.value = false;
  }
};

const cancelJob = async (job) => {
  if (!canCancelJob(job) || cancellationLoading.value[job.job_id]) return;

  actionMessage.value = '';
  actionError.value = '';
  cancellationLoading.value[job.job_id] = true;
  try {
    const confirmedJob = await apiAdapter.cancelDocumentJob(job.job_id);
    if (confirmedJob?.job_id) {
      mergeJob(confirmedJob);
      if (TERMINAL_JOB_STATUSES.includes(confirmedJob.status)) clearPolling(confirmedJob.job_id);
      else schedulePoll(confirmedJob.job_id);
    } else {
      await loadJobs();
    }
    actionMessage.value = `任务 ${job.job_id} 的取消结果已由服务端确认。`;
  } catch (error) {
    actionError.value = `取消任务 ${job.job_id} 失败：${getFriendlyError(error, '请稍后重试。')}`;
  } finally {
    delete cancellationLoading.value[job.job_id];
  }
};

const clearFocus = () => router.replace({ name: 'indexing-jobs' });

const updateViewportScope = () => {
  const wasDesktop = isDesktop.value;
  isDesktop.value = window.innerWidth >= 768;
  if (!isDesktop.value) clearAllPolling();
  if (!wasDesktop && isDesktop.value) loadJobs();
};

onMounted(() => {
  updateViewportScope();
  window.addEventListener('resize', updateViewportScope);
  if (isDesktop.value) loadJobs();
});

onBeforeUnmount(() => {
  isActive.value = false;
  clearAllPolling();
  window.removeEventListener('resize', updateViewportScope);
});
</script>

<style scoped>
.indexing-jobs { max-width: 1280px; margin: 0 auto; }
.indexing-jobs__header { display: flex; align-items: flex-start; justify-content: space-between; gap: var(--space-5); padding-bottom: var(--space-5); border-bottom: 1px solid var(--color-rule); }
.indexing-jobs__eyebrow { margin: 0 0 var(--space-2); color: var(--color-moss); font-size: 12px; font-weight: 600; }
.indexing-jobs h1 { margin: 0; font-family: var(--font-display); font-size: 26px; font-weight: 600; line-height: 1.3; }
.indexing-jobs__description { max-width: 620px; margin: var(--space-2) 0 0; color: var(--color-ink-soft); font-size: 14px; line-height: 1.7; }
.indexing-jobs__refresh, .indexing-jobs__error button, .indexing-jobs__action button { display: inline-flex; align-items: center; justify-content: center; gap: 6px; min-height: 32px; border: 1px solid var(--color-rule); border-radius: var(--radius-control); background: var(--color-paper-raised); color: var(--color-ink); font: inherit; font-size: 13px; cursor: pointer; }
.indexing-jobs__refresh { min-width: 82px; padding: 0 var(--space-3); }
.indexing-jobs__refresh:disabled, .indexing-jobs__action button:disabled { cursor: wait; opacity: 0.65; }
.indexing-jobs__refresh:not(:disabled):hover, .indexing-jobs__error button:hover, .indexing-jobs__action button:not(:disabled):hover { border-color: var(--color-copper); color: var(--color-copper-strong); }
.indexing-jobs__refresh:not(:disabled):active, .indexing-jobs__error button:active, .indexing-jobs__action button:not(:disabled):active { background: var(--color-paper-muted); }
.indexing-jobs__refresh-icon--spinning { animation: indexing-jobs-spin 0.9s linear infinite; }
.indexing-jobs__controls { display: flex; align-items: center; justify-content: space-between; gap: var(--space-4); min-height: 64px; padding: var(--space-4) 0; }
.indexing-jobs__filters { display: inline-flex; min-width: 340px; margin: 0; padding: 3px; border: 1px solid var(--color-rule); border-radius: var(--radius-control); }
.indexing-jobs__filters legend { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0, 0, 0, 0); }
.indexing-jobs__filter-option { position: relative; display: grid; flex: 1; min-width: 106px; min-height: 30px; place-items: center; color: var(--color-ink-soft); cursor: pointer; }
.indexing-jobs__filter-option input { position: absolute; opacity: 0; }
.indexing-jobs__filter-option span { width: 100%; padding: 6px 8px; border-radius: 3px; font-size: 13px; text-align: center; }
.indexing-jobs__filter-option:hover span { color: var(--color-ink); }
.indexing-jobs__filter-option:active span { background: var(--color-paper-muted); }
.indexing-jobs__filter-option input:checked + span { background: var(--color-paper-muted); color: var(--color-ink); font-weight: 600; }
.indexing-jobs__filter-option input:focus-visible + span, .indexing-jobs__refresh:focus-visible, .indexing-jobs__error button:focus-visible, .indexing-jobs__action button:focus-visible { outline: 2px solid var(--color-focus); outline-offset: 2px; }
.indexing-jobs__count { margin: 0; color: var(--color-ink-soft); font-size: 13px; }
.indexing-jobs__error, .indexing-jobs__success, .indexing-jobs__desktop-notice, .indexing-jobs__scope { display: flex; align-items: center; gap: var(--space-3); margin: 0 0 var(--space-4); padding: var(--space-3) var(--space-4); border-left: 3px solid var(--color-danger); background: var(--color-danger-soft); color: var(--color-danger); font-size: 13px; line-height: 1.5; }
.indexing-jobs__error span, .indexing-jobs__desktop-notice p { margin: 0; }
.indexing-jobs__error button { margin-left: auto; padding: 0 var(--space-2); border-color: currentColor; background: transparent; color: inherit; }
.indexing-jobs__success { border-left-color: var(--color-moss); background: var(--color-moss-soft); color: var(--color-moss); }
.indexing-jobs__scope { border-left-color: var(--color-moss); background: var(--color-moss-soft); color: var(--color-moss); }
.indexing-jobs__scope button { margin-left: auto; min-height: 28px; padding: 0 var(--space-2); border: 1px solid currentColor; border-radius: var(--radius-control); background: transparent; color: inherit; font: inherit; font-size: 12px; cursor: pointer; }
.indexing-jobs__scope button:hover { background: var(--color-paper-raised); }
.indexing-jobs__scope button:active { background: var(--color-paper-muted); }
.indexing-jobs__desktop-notice { align-items: flex-start; margin-top: var(--space-5); border-left-color: var(--color-warning); background: var(--color-warning-soft); color: var(--color-warning); }
.indexing-jobs__table-wrap { overflow-x: auto; border-top: 1px solid var(--color-rule); border-bottom: 1px solid var(--color-rule); background: var(--color-paper-raised); }
.indexing-jobs table { width: 100%; min-width: 1120px; border-collapse: collapse; table-layout: fixed; }
.indexing-jobs th, .indexing-jobs td { padding: 13px 12px; border-bottom: 1px solid var(--color-rule); color: var(--color-ink); font-size: 13px; line-height: 1.45; text-align: left; vertical-align: middle; }
.indexing-jobs th { position: sticky; top: 0; z-index: 1; background: var(--color-paper-muted); color: var(--color-ink-soft); font-size: 12px; font-weight: 600; }
.indexing-jobs tbody tr:last-child td { border-bottom: 0; }
.indexing-jobs th:nth-child(1), .indexing-jobs th:nth-child(2) { width: 14%; }
.indexing-jobs th:nth-child(3) { width: 12%; }
.indexing-jobs th:nth-child(4) { width: 11%; }
.indexing-jobs th:nth-child(5) { width: 14%; }
.indexing-jobs th:nth-child(6) { width: 15%; }
.indexing-jobs th:nth-child(7) { width: 12%; }
.indexing-jobs th:nth-child(8) { width: 8%; }
.indexing-jobs td.indexing-jobs__identifier, .indexing-jobs td.indexing-jobs__timestamp { overflow: hidden; font-family: var(--font-mono); font-size: 12px; text-overflow: ellipsis; white-space: nowrap; }
.indexing-jobs__message { overflow-wrap: anywhere; color: var(--color-ink-soft); }
.indexing-jobs__status { display: inline-flex; min-width: 124px; align-items: center; justify-content: center; padding: 3px 6px; border: 1px solid currentColor; border-radius: 3px; font-size: 12px; white-space: nowrap; }
.indexing-jobs__status--neutral { color: var(--color-ink-soft); }
.indexing-jobs__status--warning { color: var(--color-warning); }
.indexing-jobs__status--success { color: var(--color-moss); }
.indexing-jobs__status--danger { color: var(--color-danger); }
.indexing-jobs__progress { position: relative; display: flex; width: 132px; height: 20px; align-items: center; overflow: hidden; border: 1px solid var(--color-rule); border-radius: 3px; background: var(--color-paper-muted); }
.indexing-jobs__progress span { position: absolute; inset: 0 auto 0 0; background: var(--color-moss-soft); }
.indexing-jobs__progress b { position: relative; width: 100%; font-family: var(--font-mono); font-size: 11px; font-weight: 500; text-align: center; }
.indexing-jobs__action { text-align: right; }
.indexing-jobs__action button { min-width: 62px; padding: 0 var(--space-2); border-color: var(--color-danger); color: var(--color-danger); }
.indexing-jobs__terminal-label { color: var(--color-ink-soft); font-size: 12px; white-space: nowrap; }
.indexing-jobs__state { height: 192px; color: var(--color-ink-soft); text-align: center; }
@keyframes indexing-jobs-spin { to { transform: rotate(360deg); } }
@media (max-width: 1024px) { .indexing-jobs__header { gap: var(--space-4); } }
</style>
