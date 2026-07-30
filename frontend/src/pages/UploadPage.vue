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
      <p v-if="deleteSuccess" class="document-library__success" role="status">{{ deleteSuccess }}</p>
      <p v-if="deleteError" class="document-library__error" role="alert">{{ deleteError }}</p>

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
                <div v-if="rebuildJobFor(document)" class="document-library__rebuild-status" role="status">
                  <p>已创建重建任务 <span class="document-library__rebuild-job-id">{{ rebuildJobFor(document).jobId }}</span>。</p>
                  <p v-if="rebuildJobFor(document).hasPublishedChunks">
                    当前已发布版本的 {{ rebuildJobFor(document).publishedChunkCount }} 个分块仍可用于检索；候选分块尚未发布。
                  </p>
                  <p v-else>候选分块尚未发布，请等待构建任务完成后再查看。</p>
                  <button
                    type="button"
                    :aria-label="`查看构建任务 ${rebuildJobFor(document).jobId}`"
                    @click="openJob(rebuildJobFor(document).jobId)"
                  >
                    查看构建任务
                  </button>
                </div>
                <div v-else-if="publishedContinuityMessage(document)" class="document-library__rebuild-status" role="status">
                  <p>{{ publishedContinuityMessage(document) }}</p>
                </div>
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
                <button
                  v-if="canInspectChunks(document)"
                  type="button"
                  class="document-library__icon-action"
                  :aria-label="`查看文档 ${document.document_id} 的已发布分块`"
                  :title="'查看已发布分块'"
                  @click="inspectChunks(document)"
                >
                  <FileSearch :size="16" aria-hidden="true" />
                </button>
                <button
                  v-if="canRebuild(document)"
                  type="button"
                  class="document-library__icon-action"
                  :aria-label="`重新构建文档 ${document.document_id}`"
                  :title="'重新构建文档'"
                  @click="openRebuild(document)"
                >
                  <RefreshCw :size="16" aria-hidden="true" />
                </button>
                <button
                  v-if="canDelete(document)"
                  type="button"
                  class="document-library__icon-action document-library__icon-action--danger"
                  :aria-label="`删除文档 ${document.document_id}`"
                  :title="'删除文档'"
                  :disabled="Boolean(deletionLoading[document.document_id])"
                  @click="deleteDocument(document)"
                >
                  <RefreshCw
                    v-if="deletionLoading[document.document_id]"
                    :size="16"
                    class="document-library__refresh-icon--spinning"
                    aria-hidden="true"
                  />
                  <Trash2 v-else :size="16" aria-hidden="true" />
                </button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </template>

    <el-dialog
      v-model="chunksDialogVisible"
      class="document-library__chunks-dialog"
      width="min(92vw, 860px)"
      :close-on-click-modal="false"
      @closed="resetChunkInspection"
    >
      <template #header>
        <div>
          <p class="document-library__dialog-eyebrow">已发布版本</p>
          <h2>已发布分块</h2>
        </div>
      </template>

      <div v-if="inspectedDocument" class="document-library__chunks" :aria-busy="chunksLoading">
        <p class="document-library__chunks-identity">文档 ID：{{ inspectedDocument.document_id }}</p>
        <p class="document-library__chunks-name">{{ inspectedDocument.filename }}</p>

        <p v-if="chunksLoading" class="document-library__chunks-state" role="status">正在加载已发布分块...</p>
        <div v-else-if="chunkInspectionError" class="document-library__chunks-error" role="alert">
          <p>{{ chunkInspectionError }}</p>
          <button type="button" @click="loadChunks(chunkPagination.page)">重新检查</button>
        </div>
        <p v-else-if="!chunks.length" class="document-library__chunks-state" role="status">
          当前已发布版本没有可展示的分块。
        </p>
        <ol v-else class="document-library__chunk-list">
          <li v-for="chunk in chunks" :key="chunk.chunk_id" class="document-library__chunk">
            <header>
              <span>分块 {{ Number(chunk.chunk_index) + 1 }}</span>
              <span class="document-library__chunk-id">{{ chunk.chunk_id }}</span>
            </header>
            <p class="document-library__chunk-content">{{ chunk.content || '（分块内容为空）' }}</p>
            <p v-if="chunk.keywords?.length" class="document-library__chunk-detail">关键词：{{ chunk.keywords.join('、') }}</p>
            <p v-if="chunk.generated_questions?.length" class="document-library__chunk-detail">
              生成问题：{{ chunk.generated_questions.join('、') }}
            </p>
            <dl v-if="metadataEntries(chunk.metadata).length" class="document-library__chunk-metadata">
              <template v-for="entry in metadataEntries(chunk.metadata)" :key="entry.key">
                <dt>{{ entry.key }}</dt>
                <dd>{{ entry.value }}</dd>
              </template>
            </dl>
          </li>
        </ol>

        <footer v-if="chunkTotalPages > 1 && !chunksLoading && !chunkInspectionError" class="document-library__chunk-pagination">
          <button type="button" aria-label="上一页" :disabled="chunkPagination.page <= 1" @click="loadChunks(chunkPagination.page - 1)">
            <ChevronLeft :size="16" aria-hidden="true" />
          </button>
          <span>第 {{ chunkPagination.page }} / {{ chunkTotalPages }} 页</span>
          <button
            type="button"
            aria-label="下一页"
            :disabled="chunkPagination.page >= chunkTotalPages"
            @click="loadChunks(chunkPagination.page + 1)"
          >
            <ChevronRight :size="16" aria-hidden="true" />
          </button>
        </footer>
      </div>
    </el-dialog>

    <el-dialog
      v-model="rebuildDialogVisible"
      class="document-library__rebuild-dialog"
      width="min(92vw, 520px)"
      :close-on-click-modal="false"
      @closed="resetRebuild"
    >
      <template #header>
        <div>
          <p class="document-library__dialog-eyebrow">重建已存在文档</p>
          <h2>重新构建文档</h2>
        </div>
      </template>

      <div v-if="rebuildTarget" class="document-library__rebuild-form">
        <p class="document-library__rebuild-document">{{ rebuildTarget.filename }}</p>
        <p v-if="hasPublishedGeneration(rebuildTarget)" class="document-library__rebuild-continuity">
          当前已发布版本的 {{ rebuildTarget.chunk_count }} 个分块将在新任务运行时继续用于检索；候选分块尚未发布。
        </p>
        <label for="document-rebuild-strategy">
          <span>重建分块策略</span>
          <select id="document-rebuild-strategy" v-model="rebuildStrategy" aria-label="重建分块策略" :disabled="rebuildLoading">
            <option v-for="strategy in supportedRebuildStrategies" :key="strategy.value" :value="strategy.value">
              {{ strategy.label }}
            </option>
          </select>
        </label>
        <p class="document-library__rebuild-help">策略选项与服务端当前接受的重建契约保持一致。</p>
        <p v-if="rebuildError" class="document-library__rebuild-error" role="alert">{{ rebuildError }}</p>
      </div>

      <template #footer>
        <button type="button" class="document-library__dialog-button" :disabled="rebuildLoading" @click="rebuildDialogVisible = false">取消</button>
        <button type="button" class="document-library__dialog-button document-library__dialog-button--primary" :disabled="rebuildLoading" @click="submitRebuild">
          {{ rebuildLoading ? '正在创建' : '创建重建任务' }}
        </button>
      </template>
    </el-dialog>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { useRouter } from 'vue-router';
import { ElMessageBox } from 'element-plus';
import { ChevronLeft, ChevronRight, CircleCheck, FileSearch, Monitor, RefreshCw, Search, Trash2 } from 'lucide-vue-next';
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
const chunksDialogVisible = ref(false);
const inspectedDocument = ref(null);
const chunks = ref([]);
const chunksLoading = ref(false);
const chunkInspectionError = ref('');
const chunkPagination = ref({ page: 1, page_size: 10, total: 0 });
const rebuildDialogVisible = ref(false);
const rebuildTarget = ref(null);
const rebuildStrategy = ref('general');
const rebuildLoading = ref(false);
const rebuildError = ref('');
const rebuildJobs = ref({});
const deletionLoading = ref({});
const deleteSuccess = ref('');
const deleteError = ref('');

const CHUNK_PAGE_SIZE = 10;
const supportedRebuildStrategies = [
  { value: 'general', label: '通用策略 (general)' },
  { value: 'paper', label: '论文策略 (paper)' },
  { value: 'qa', label: '问答策略 (qa)' }
];
let chunkRequestEpoch = 0;

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

const chunkTotalPages = computed(() =>
  Math.max(1, Math.ceil(Number(chunkPagination.value.total || 0) / Number(chunkPagination.value.page_size || CHUNK_PAGE_SIZE)))
);

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

const chunkInspectionErrorMessage = (error) => {
  if (error?.status === 401) return '登录状态已失效，请重新登录。';
  if (error?.status === 403) return '当前账户无权查看文档分块。';
  if (error?.status === 409) return '当前文档尚未产生可查看的已发布分块，请等待构建完成后重试。';
  return '加载已发布分块失败，请重新检查。';
};

const hasPublishedGeneration = (document) => Number(document?.published_generation) > 0;

const canInspectChunks = (document) => document.status === 'ready' && hasPublishedGeneration(document);

const canRebuild = (document) => ['ready', 'failed'].includes(document.status);

const canDelete = (document) => document.status !== 'deleting';

const rebuildJobFor = (document) => rebuildJobs.value[document.document_id] || null;

const publishedContinuityMessage = (document) => {
  if (rebuildJobFor(document) || !hasPublishedGeneration(document) || document.status === 'ready') return '';

  const prefix = document.status === 'failed' ? '最近一次重建失败；' : '当前正在重新构建；';
  return `${prefix}当前已发布版本的 ${formatChunkCount(document.chunk_count)} 个分块仍可用于检索。候选分块未发布。`;
};

const metadataEntries = (metadata) =>
  Object.entries(metadata || {}).map(([key, value]) => ({
    key,
    value: typeof value === 'string' ? value : JSON.stringify(value)
  }));

const loadChunks = async (page = 1) => {
  if (!inspectedDocument.value || chunksLoading.value) return;

  const documentId = inspectedDocument.value.document_id;
  const requestEpoch = ++chunkRequestEpoch;
  chunksLoading.value = true;
  chunkInspectionError.value = '';
  try {
    const data = await apiAdapter.getDocumentChunks(documentId, {
      page,
      page_size: CHUNK_PAGE_SIZE
    });
    if (requestEpoch !== chunkRequestEpoch || inspectedDocument.value?.document_id !== documentId) return;

    chunks.value = Array.isArray(data?.items) ? data.items : [];
    chunkPagination.value = {
      page: Number(data?.pagination?.page) || page,
      page_size: Number(data?.pagination?.page_size) || CHUNK_PAGE_SIZE,
      total: Number(data?.pagination?.total) || 0
    };
  } catch (error) {
    if (requestEpoch !== chunkRequestEpoch || inspectedDocument.value?.document_id !== documentId) return;
    chunkInspectionError.value = chunkInspectionErrorMessage(error);
  } finally {
    if (requestEpoch === chunkRequestEpoch) chunksLoading.value = false;
  }
};

const inspectChunks = (document) => {
  chunkRequestEpoch += 1;
  inspectedDocument.value = document;
  chunks.value = [];
  chunksLoading.value = false;
  chunkPagination.value = { page: 1, page_size: CHUNK_PAGE_SIZE, total: 0 };
  chunkInspectionError.value = '';
  chunksDialogVisible.value = true;
  loadChunks(1);
};

const resetChunkInspection = () => {
  chunkRequestEpoch += 1;
  inspectedDocument.value = null;
  chunks.value = [];
  chunksLoading.value = false;
  chunkInspectionError.value = '';
};

const rebuildErrorMessage = (error) => {
  if (error?.status === 401) return '登录状态已失效，请重新登录。';
  if (error?.status === 403) return '当前账户无权重新构建文档。';
  if (error?.status === 409) return '文档当前正在变更，请刷新后重试。';
  if (error?.status >= 400 && error?.status < 500) return '重建策略被服务端拒绝，请重新选择后重试。';
  return '创建重建任务失败，请稍后重试。';
};

const openRebuild = (document) => {
  rebuildTarget.value = document;
  rebuildStrategy.value = 'general';
  rebuildError.value = '';
  rebuildDialogVisible.value = true;
};

const resetRebuild = () => {
  rebuildTarget.value = null;
  rebuildError.value = '';
  rebuildLoading.value = false;
};

const submitRebuild = async () => {
  if (!rebuildTarget.value || rebuildLoading.value) return;

  rebuildLoading.value = true;
  rebuildError.value = '';
  try {
    const target = rebuildTarget.value;
    const job = await apiAdapter.buildDocument(target.document_id, { chunk_strategy: rebuildStrategy.value });
    if (!job?.job_id) throw new Error('rebuild job id is missing');

    rebuildJobs.value = {
      ...rebuildJobs.value,
      [target.document_id]: {
        jobId: job.job_id,
        hasPublishedChunks: hasPublishedGeneration(target),
        publishedChunkCount: Number(target.chunk_count) || 0
      }
    };
    documents.value = documents.value.map((document) =>
      document.document_id === target.document_id ? { ...document, status: 'pending' } : document
    );
    rebuildDialogVisible.value = false;
  } catch (error) {
    rebuildError.value = rebuildErrorMessage(error);
  } finally {
    rebuildLoading.value = false;
  }
};

const deleteErrorMessage = (error) => {
  if (error?.status === 401) return '登录状态已失效，请重新登录。';
  if (error?.status === 403) return '当前账户无权删除文档。';
  if (error?.status === 404) return '文档已不在文档库中，请刷新确认当前状态。';
  if (error?.status === 409) return '文档当前正在变更，请刷新后重试。';
  return '请稍后重试。';
};

const deleteDocument = async (document) => {
  try {
    await ElMessageBox.confirm(`确认删除文档“${document.filename}”？此操作不能撤销。`, '删除文档', {
      type: 'warning',
      confirmButtonText: '删除',
      cancelButtonText: '取消'
    });
  } catch (reason) {
    if (reason === 'cancel' || reason === 'close') return;
    return;
  }

  deletionLoading.value = { ...deletionLoading.value, [document.document_id]: true };
  deleteSuccess.value = '';
  deleteError.value = '';
  try {
    const result = await apiAdapter.deleteDocument(document.filename);
    if (!result?.success_ids?.includes(document.document_id)) {
      deleteError.value = `删除文档 ${document.filename} 失败：服务端未确认删除，请刷新后重试。`;
      return;
    }

    documents.value = documents.value.filter((item) => item.document_id !== document.document_id);
    const { [document.document_id]: _removedRebuild, ...remainingRebuildJobs } = rebuildJobs.value;
    rebuildJobs.value = remainingRebuildJobs;
    deleteSuccess.value = `文档 ${document.filename} 已由服务端确认删除。`;
  } catch (error) {
    deleteError.value = `删除文档 ${document.filename} 失败：${deleteErrorMessage(error)}`;
  } finally {
    const { [document.document_id]: _completed, ...remainingLoading } = deletionLoading.value;
    deletionLoading.value = remainingLoading;
  }
};

const loadDocuments = async () => {
  if (!isDesktop.value) return;

  loading.value = true;
  listError.value = '';
  try {
    const data = await apiAdapter.listDocuments({ page: 1, page_size: 200 });
    const items = Array.isArray(data?.items) ? data.items : [];
    documents.value = items;
    rebuildJobs.value = Object.fromEntries(
      Object.entries(rebuildJobs.value).filter(([documentId]) => {
        const document = items.find((item) => item.document_id === documentId);
        return document && ['pending', 'processing'].includes(document.status);
      })
    );
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
.document-library__refresh:not(:disabled):hover, .document-library__error button:not(:disabled):hover, .document-library__upload-result button:not(:disabled):hover, .document-library__action button:not(:disabled):hover { border-color: var(--color-copper); color: var(--color-copper-strong); }
.document-library__refresh:not(:disabled):active, .document-library__error button:not(:disabled):active, .document-library__upload-result button:not(:disabled):active, .document-library__action button:not(:disabled):active { background: var(--color-paper-muted); }
.document-library__error button:disabled, .document-library__upload-result button:disabled, .document-library__action button:disabled, .document-library__rebuild-status button:disabled, .document-library__chunks-error button:disabled, .document-library__chunk-pagination button:disabled { cursor: not-allowed; opacity: 0.55; }
.document-library__refresh-icon--spinning { animation: document-library-spin 0.9s linear infinite; }
.document-library__desktop-notice, .document-library__error, .document-library__success, .document-library__upload-result { display: flex; align-items: center; gap: var(--space-3); margin: var(--space-5) 0 0; padding: var(--space-3) var(--space-4); border-left: 3px solid var(--color-warning); background: var(--color-warning-soft); color: var(--color-warning); font-size: 13px; line-height: 1.5; }
.document-library__desktop-notice p, .document-library__upload-result p { margin: 0; }
.document-library__error { border-left-color: var(--color-danger); background: var(--color-danger-soft); color: var(--color-danger); }
.document-library__success { border-left-color: var(--color-moss); background: var(--color-moss-soft); color: var(--color-moss); }
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
.document-library th:nth-child(1) { width: 22%; }
.document-library th:nth-child(2) { width: 8%; }
.document-library th:nth-child(3) { width: 10%; }
.document-library th:nth-child(4) { width: 22%; }
.document-library th:nth-child(5) { width: 8%; }
.document-library th:nth-child(6) { width: 15%; }
.document-library th:nth-child(7) { width: 15%; }
.document-library__filename { overflow: hidden; font-weight: 600; text-overflow: ellipsis; white-space: nowrap; }
.document-library__numeric, .document-library__timestamp { font-family: var(--font-mono); font-size: 12px; }
.document-library__timestamp { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.document-library__status { display: inline-flex; min-width: 124px; align-items: center; justify-content: center; padding: 3px 6px; border: 1px solid currentColor; border-radius: 3px; font-size: 12px; white-space: nowrap; }
.document-library__status--neutral { color: var(--color-ink-soft); }
.document-library__status--warning { color: var(--color-warning); }
.document-library__status--success { color: var(--color-moss); }
.document-library__status--danger { color: var(--color-danger); }
.document-library__rebuild-status { margin-top: var(--space-2); color: var(--color-ink-soft); font-size: 12px; line-height: 1.5; }
.document-library__rebuild-status p { margin: 0; }
.document-library__rebuild-status p + p { margin-top: var(--space-1); }
.document-library__rebuild-job-id { font-family: var(--font-mono); }
.document-library__rebuild-status button { min-height: auto; margin-top: var(--space-2); padding: 0; border: 0; background: transparent; color: var(--color-copper-strong); font: inherit; text-decoration: underline; cursor: pointer; }
.document-library__rebuild-status button:hover { color: var(--color-copper); }
.document-library__rebuild-status button:active { color: var(--color-ink); }
.document-library__action { display: flex; justify-content: flex-end; gap: var(--space-2); text-align: right; }
.document-library__action button { min-width: 72px; padding: 0 var(--space-2); }
.document-library__action .document-library__icon-action { width: 32px; min-width: 32px; padding: 0; }
.document-library__action .document-library__icon-action--danger { border-color: var(--color-danger); color: var(--color-danger); }
.document-library__state { height: 192px; color: var(--color-ink-soft); text-align: center; }
.document-library__dialog-eyebrow { margin: 0 0 var(--space-1); color: var(--color-moss); font-size: 12px; font-weight: 600; }
.document-library__chunks-dialog h2 { margin: 0; font-family: var(--font-display); font-size: 20px; font-weight: 600; }
.document-library__chunks { min-height: 240px; }
.document-library__chunks-identity, .document-library__chunks-name { margin: 0; color: var(--color-ink-soft); font-size: 13px; }
.document-library__chunks-identity { font-family: var(--font-mono); }
.document-library__chunks-name { margin-top: var(--space-1); }
.document-library__chunks-state { padding: var(--space-6) var(--space-4); color: var(--color-ink-soft); text-align: center; }
.document-library__chunks-error { display: flex; align-items: center; justify-content: space-between; gap: var(--space-3); margin-top: var(--space-4); padding: var(--space-3) var(--space-4); border-left: 3px solid var(--color-danger); background: var(--color-danger-soft); color: var(--color-danger); font-size: 13px; line-height: 1.5; }
.document-library__chunks-error p { margin: 0; }
.document-library__chunks-error button, .document-library__chunk-pagination button { min-height: 32px; border: 1px solid currentColor; border-radius: var(--radius-control); background: transparent; color: inherit; font: inherit; cursor: pointer; }
.document-library__chunks-error button { flex: 0 0 auto; padding: 0 var(--space-2); }
.document-library__chunks-error button:hover, .document-library__chunk-pagination button:not(:disabled):hover { background: var(--color-paper-raised); }
.document-library__chunks-error button:active, .document-library__chunk-pagination button:not(:disabled):active { background: var(--color-paper-muted); }
.document-library__chunk-list { display: grid; gap: var(--space-3); margin: var(--space-4) 0 0; padding: 0; list-style: none; }
.document-library__chunk { padding: var(--space-4); border: 1px solid var(--color-rule); background: var(--color-paper-raised); }
.document-library__chunk header { display: flex; justify-content: space-between; gap: var(--space-3); color: var(--color-ink-soft); font-size: 12px; }
.document-library__chunk-id { overflow: hidden; font-family: var(--font-mono); text-overflow: ellipsis; white-space: nowrap; }
.document-library__chunk-content { margin: var(--space-3) 0; white-space: pre-wrap; line-height: 1.7; }
.document-library__chunk-detail { margin: var(--space-2) 0 0; color: var(--color-ink-soft); font-size: 13px; line-height: 1.5; }
.document-library__chunk-metadata { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: var(--space-1) var(--space-3); margin: var(--space-3) 0 0; color: var(--color-ink-soft); font-family: var(--font-mono); font-size: 12px; }
.document-library__chunk-metadata dt, .document-library__chunk-metadata dd { margin: 0; overflow-wrap: anywhere; }
.document-library__chunk-pagination { display: flex; align-items: center; justify-content: center; gap: var(--space-3); margin-top: var(--space-4); color: var(--color-ink-soft); font-family: var(--font-mono); font-size: 12px; }
.document-library__chunk-pagination button { width: 32px; padding: 0; color: var(--color-ink-soft); }
.document-library__chunk-pagination button:disabled { cursor: not-allowed; opacity: 0.45; }
.document-library__rebuild-document { margin: 0; color: var(--color-ink); font-weight: 600; overflow-wrap: anywhere; }
.document-library__rebuild-continuity { margin: var(--space-3) 0 0; padding: var(--space-3); border-left: 3px solid var(--color-moss); background: var(--color-moss-soft); color: var(--color-moss); font-size: 13px; line-height: 1.6; }
.document-library__rebuild-form label { display: grid; gap: var(--space-2); margin-top: var(--space-4); color: var(--color-ink-soft); font-size: 13px; }
.document-library__rebuild-form select { min-height: 36px; border: 1px solid var(--color-rule); border-radius: var(--radius-control); background: var(--color-paper-raised); color: var(--color-ink); padding: 0 var(--space-2); }
.document-library__rebuild-form select:not(:disabled):hover { border-color: var(--color-copper); }
.document-library__rebuild-form select:not(:disabled):active { background: var(--color-paper-muted); }
.document-library__rebuild-form select:disabled { cursor: wait; opacity: 0.65; }
.document-library__rebuild-help { margin: var(--space-2) 0 0; color: var(--color-ink-soft); font-size: 12px; line-height: 1.5; }
.document-library__rebuild-error { margin: var(--space-3) 0 0; padding: var(--space-3); border-left: 3px solid var(--color-danger); background: var(--color-danger-soft); color: var(--color-danger); font-size: 13px; line-height: 1.5; }
.document-library__dialog-button { min-height: 34px; margin-left: var(--space-2); padding: 0 var(--space-3); border: 1px solid var(--color-rule); border-radius: var(--radius-control); background: var(--color-paper-raised); color: var(--color-ink); font: inherit; cursor: pointer; }
.document-library__dialog-button--primary { border-color: var(--color-copper); background: var(--color-copper); color: var(--color-paper-raised); }
.document-library__dialog-button:not(:disabled):hover { border-color: var(--color-copper-strong); background: var(--color-paper-muted); color: var(--color-copper-strong); }
.document-library__dialog-button--primary:not(:disabled):hover { border-color: var(--color-copper-strong); background: var(--color-copper-strong); color: var(--color-paper-raised); }
.document-library__dialog-button:not(:disabled):active { background: var(--color-paper-muted); }
.document-library__dialog-button--primary:not(:disabled):active { background: var(--color-ink); border-color: var(--color-ink); color: var(--color-paper-raised); }
.document-library__dialog-button:disabled { cursor: wait; opacity: 0.65; }
@keyframes document-library-spin { to { transform: rotate(360deg); } }
@media (max-width: 1024px) { .document-library__header { gap: var(--space-4); } }
</style>
