<template>
  <section class="document-library" aria-labelledby="document-library-title">
    <header class="document-library__header">
      <div>
        <p class="document-library__eyebrow">知识库运维</p>
        <h1 id="document-library-title">文档库</h1>
        <p class="document-library__description">查看已上传文档，并为新的资料创建索引构建任务。</p>
      </div>
      <button v-if="isDesktop" class="document-library__refresh" type="button" :disabled="loading" @click="loadDocuments">
        <RefreshCw :size="16" :class="{ 'document-library__refresh-icon--spinning': loading }" aria-hidden="true" />
        <span>{{ loading ? '正在刷新' : '刷新' }}</span>
      </button>
    </header>

    <div v-if="!isDesktop" class="document-library__desktop-notice" role="status">
      <Monitor :size="18" aria-hidden="true" />
      <p>文档库当前仅支持桌面工作区。</p>
    </div>

    <template v-else>
      <UploadPanel @uploaded="handleUploaded" />

      <div v-if="uploadResult" class="document-library__upload-result" role="status">
        <CircleCheck :size="18" aria-hidden="true" />
        <div>
          <p>已为 {{ uploadResult.filename }} 创建初始构建任务。</p>
          <p class="document-library__identifiers">文档 ID：{{ uploadResult.document_id || '-' }}<br />构建任务 ID：{{ uploadResult.job_id || '-' }}</p>
        </div>
        <button
          v-if="uploadResult.job_id"
          type="button"
          :aria-label="`查看构建任务 ${uploadResult.job_id}`"
          @click="openJob(uploadResult.job_id)"
        >
          查看构建任务
        </button>
      </div>

      <div class="document-library__controls" aria-label="文档库控制">
        <label class="document-library__search" for="document-library-search">
          <Search :size="16" aria-hidden="true" />
          <span class="sr-only">按文件名搜索</span>
          <input id="document-library-search" v-model="keyword" type="search" placeholder="按文件名搜索" />
        </label>
        <label class="document-library__filter" for="document-library-status">
          <span>状态</span>
          <select id="document-library-status" v-model="statusFilter" aria-label="按状态筛选">
            <option v-for="option in statusFilterOptions" :key="option.value" :value="option.value">{{ option.label }}</option>
          </select>
        </label>
        <p class="document-library__count" aria-live="polite">显示 {{ filteredDocuments.length }} 个文档</p>
      </div>

      <p v-if="listError" class="document-library__error" role="alert">
        <span>{{ listError }}</span>
        <button type="button" @click="loadDocuments">重新加载</button>
      </p>

      <div class="document-library__table-wrap" :aria-busy="loading">
        <table>
          <caption class="sr-only">文档库列表</caption>
          <thead>
            <tr>
              <th scope="col">文件名</th>
              <th scope="col">类型</th>
              <th scope="col">大小</th>
              <th scope="col">状态</th>
              <th scope="col">分块数</th>
              <th scope="col">上传时间</th>
              <th scope="col"><span class="sr-only">文档操作</span></th>
            </tr>
          </thead>
          <tbody>
            <tr v-if="loading && !documents.length">
              <td colspan="7" class="document-library__state">正在加载文档库...</td>
            </tr>
            <tr v-else-if="!filteredDocuments.length">
              <td colspan="7" class="document-library__state">{{ emptyStateText }}</td>
            </tr>
            <tr v-for="document in filteredDocuments" :key="document.document_id">
              <td class="document-library__filename">{{ document.filename || '-' }}</td>
              <td>{{ formatFileType(document.file_type) }}</td>
              <td class="document-library__numeric">{{ formatFileSize(document.file_size) }}</td>
              <td>
                <span class="document-library__status" :class="`document-library__status--${documentStatusMeta(document.status).tone}`">
                  {{ documentStatusMeta(document.status).label }}
                </span>
              </td>
              <td class="document-library__numeric">{{ formatChunkCount(document.chunk_count) }}</td>
              <td class="document-library__timestamp">{{ formatTime(document.uploaded_at) }}</td>
              <td class="document-library__action">
                <button
                  type="button"
                  :aria-label="`查看文档 ${document.document_id} 的构建任务`"
                  @click="openDocumentJobs(document.document_id)"
                >
                  查看任务
                </button>
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
import { useRouter } from 'vue-router';
import { CircleCheck, Monitor, RefreshCw, Search } from 'lucide-vue-next';
import UploadPanel from '../components/UploadPanel.vue';
import { apiAdapter } from '../api/adapters';

const statusFilterOptions = [
  { value: 'all', label: '全部状态' },
  { value: 'pending', label: '待处理' },
  { value: 'processing', label: '处理中' },
  { value: 'ready', label: '可检索' },
  { value: 'failed', label: '构建失败' },
  { value: 'deleting', label: '删除中' }
];

const documentStatuses = {
  pending: { label: '待处理 (pending)', tone: 'neutral' },
  processing: { label: '处理中 (processing)', tone: 'warning' },
  ready: { label: '可检索 (ready)', tone: 'success' },
  failed: { label: '构建失败 (failed)', tone: 'danger' },
  deleting: { label: '删除中 (deleting)', tone: 'warning' }
};

const router = useRouter();
const documents = ref([]);
const keyword = ref('');
const statusFilter = ref('all');
const loading = ref(false);
const listError = ref('');
const uploadResult = ref(null);
const isDesktop = ref(true);

const filteredDocuments = computed(() => {
  const query = keyword.value.trim().toLowerCase();
  return documents.value.filter((document) => {
    const matchesStatus = statusFilter.value === 'all' || document.status === statusFilter.value;
    const matchesFilename = !query || document.filename?.toLowerCase().includes(query);
    return matchesStatus && matchesFilename;
  });
});

const emptyStateText = computed(() => {
  if (documents.value.length && (keyword.value.trim() || statusFilter.value !== 'all')) {
    return '没有符合当前筛选条件的文档。';
  }
  return '当前文档库为空。';
});

const documentStatusMeta = (status) => documentStatuses[status] || { label: status || '-', tone: 'neutral' };

const formatFileType = (fileType) => (fileType ? String(fileType).toUpperCase() : '-');

const formatFileSize = (value) => {
  const bytes = Number(value);
  if (!Number.isFinite(bytes)) return '-';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
};

const formatChunkCount = (value) => (Number.isFinite(Number(value)) ? Number(value) : '-');

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

const loadErrorMessage = (error) => {
  if (error?.status === 401) return '登录状态已失效，请重新登录。';
  if (error?.status === 403) return '当前账户无权查看文档库。';
  return '加载文档库失败，请重新加载。';
};

const loadDocuments = async () => {
  if (!isDesktop.value) return;

  loading.value = true;
  listError.value = '';
  try {
    const data = await apiAdapter.listDocuments({ page: 1, page_size: 200 });
    documents.value = data?.items || [];
  } catch (error) {
    listError.value = loadErrorMessage(error);
  } finally {
    loading.value = false;
  }
};

const handleUploaded = async (result) => {
  uploadResult.value = result;
  await loadDocuments();
};

const openJob = (jobId) => router.push({ name: 'indexing-jobs', query: { job: jobId } });

const openDocumentJobs = (documentId) => router.push({ name: 'indexing-jobs', query: { document: documentId } });

const updateViewportScope = () => {
  const wasDesktop = isDesktop.value;
  isDesktop.value = window.innerWidth >= 768;
  if (!wasDesktop && isDesktop.value) loadDocuments();
};

onMounted(() => {
  updateViewportScope();
  window.addEventListener('resize', updateViewportScope);
  if (isDesktop.value) loadDocuments();
});

onBeforeUnmount(() => window.removeEventListener('resize', updateViewportScope));
</script>

<style scoped>
.document-library { max-width: 1280px; margin: 0 auto; }
.document-library__header { display: flex; align-items: flex-start; justify-content: space-between; gap: var(--space-5); padding-bottom: var(--space-5); border-bottom: 1px solid var(--color-rule); }
.document-library__eyebrow { margin: 0 0 var(--space-2); color: var(--color-moss); font-size: 12px; font-weight: 600; }
.document-library h1 { margin: 0; font-family: var(--font-display); font-size: 26px; font-weight: 600; line-height: 1.3; }
.document-library__description { max-width: 620px; margin: var(--space-2) 0 0; color: var(--color-ink-soft); font-size: 14px; line-height: 1.7; }
.document-library__refresh, .document-library__error button, .document-library__upload-result button, .document-library__action button { min-height: 32px; display: inline-flex; align-items: center; justify-content: center; gap: 6px; border: 1px solid var(--color-rule); border-radius: var(--radius-control); background: var(--color-paper-raised); color: var(--color-ink); font: inherit; font-size: 13px; cursor: pointer; }
.document-library__refresh { min-width: 82px; padding: 0 var(--space-3); }
.document-library__refresh:disabled { cursor: wait; opacity: 0.65; }
.document-library__refresh:not(:disabled):hover, .document-library__error button:hover, .document-library__upload-result button:hover, .document-library__action button:hover { border-color: var(--color-copper); color: var(--color-copper-strong); }
.document-library__refresh:not(:disabled):active, .document-library__error button:active, .document-library__upload-result button:active, .document-library__action button:active { background: var(--color-paper-muted); }
.document-library__refresh-icon--spinning { animation: document-library-spin 0.9s linear infinite; }
.document-library__desktop-notice, .document-library__error, .document-library__upload-result { display: flex; align-items: center; gap: var(--space-3); margin: var(--space-5) 0 0; padding: var(--space-3) var(--space-4); border-left: 3px solid var(--color-warning); background: var(--color-warning-soft); color: var(--color-warning); font-size: 13px; line-height: 1.5; }
.document-library__desktop-notice p, .document-library__upload-result p { margin: 0; }
.document-library__error { border-left-color: var(--color-danger); background: var(--color-danger-soft); color: var(--color-danger); }
.document-library__error button { margin-left: auto; padding: 0 var(--space-2); border-color: currentColor; background: transparent; color: inherit; }
.document-library__upload-result { align-items: flex-start; border-left-color: var(--color-moss); background: var(--color-moss-soft); color: var(--color-moss); }
.document-library__upload-result > div { min-width: 0; }
.document-library__identifiers { margin-top: var(--space-1) !important; font-family: var(--font-mono); font-size: 12px; }
.document-library__upload-result button { margin-left: auto; flex: 0 0 auto; padding: 0 var(--space-3); border-color: currentColor; background: transparent; color: inherit; }
.document-library__controls { display: flex; align-items: center; gap: var(--space-3); min-height: 64px; padding: var(--space-4) 0; }
.document-library__search { display: flex; width: min(360px, 45%); min-width: 220px; align-items: center; gap: var(--space-2); padding: 0 var(--space-3); border: 1px solid var(--color-rule); border-radius: var(--radius-control); background: var(--color-paper-raised); color: var(--color-ink-soft); }
.document-library__search:focus-within { border-color: var(--color-copper); }
.document-library__search input { width: 100%; min-width: 0; height: 32px; border: 0; background: transparent; color: var(--color-ink); font-size: 13px; }
.document-library__filter { display: inline-flex; align-items: center; gap: var(--space-2); color: var(--color-ink-soft); font-size: 13px; }
.document-library__filter select { min-height: 34px; border: 1px solid var(--color-rule); border-radius: var(--radius-control); background: var(--color-paper-raised); color: var(--color-ink); padding: 0 var(--space-2); }
.document-library__filter select:hover { border-color: var(--color-copper); }
.document-library__filter select:active { background: var(--color-paper-muted); }
.document-library__count { margin: 0 0 0 auto; color: var(--color-ink-soft); font-size: 13px; white-space: nowrap; }
.document-library__table-wrap { overflow-x: auto; border-top: 1px solid var(--color-rule); border-bottom: 1px solid var(--color-rule); background: var(--color-paper-raised); }
.document-library table { width: 100%; min-width: 1080px; border-collapse: collapse; table-layout: fixed; }
.document-library th, .document-library td { padding: 13px 12px; border-bottom: 1px solid var(--color-rule); color: var(--color-ink); font-size: 13px; line-height: 1.45; text-align: left; vertical-align: middle; }
.document-library th { position: sticky; top: 0; z-index: 1; background: var(--color-paper-muted); color: var(--color-ink-soft); font-size: 12px; font-weight: 600; }
.document-library tbody tr:last-child td { border-bottom: 0; }
.document-library th:nth-child(1) { width: 25%; }
.document-library th:nth-child(2) { width: 9%; }
.document-library th:nth-child(3) { width: 11%; }
.document-library th:nth-child(4) { width: 16%; }
.document-library th:nth-child(5) { width: 10%; }
.document-library th:nth-child(6) { width: 19%; }
.document-library th:nth-child(7) { width: 10%; }
.document-library__filename { overflow: hidden; font-weight: 600; text-overflow: ellipsis; white-space: nowrap; }
.document-library__numeric, .document-library__timestamp { font-family: var(--font-mono); font-size: 12px; }
.document-library__timestamp { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.document-library__status { display: inline-flex; min-width: 124px; align-items: center; justify-content: center; padding: 3px 6px; border: 1px solid currentColor; border-radius: 3px; font-size: 12px; white-space: nowrap; }
.document-library__status--neutral { color: var(--color-ink-soft); }
.document-library__status--warning { color: var(--color-warning); }
.document-library__status--success { color: var(--color-moss); }
.document-library__status--danger { color: var(--color-danger); }
.document-library__action { text-align: right; }
.document-library__action button { min-width: 72px; padding: 0 var(--space-2); }
.document-library__state { height: 192px; color: var(--color-ink-soft); text-align: center; }
@keyframes document-library-spin { to { transform: rotate(360deg); } }
@media (max-width: 1024px) { .document-library__header { gap: var(--space-4); } }
</style>
