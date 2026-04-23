<template>
  <section class="upload-page">
    <div class="top-bar">
      <h1>文档上传</h1>
      <el-button v-if="authStore.isAdmin" class="btn-ghost" @click="refreshAll">刷新</el-button>
    </div>

    <div v-if="!authStore.isLoggedIn" class="card notice">请先在聊天页登录。</div>
    <div v-else-if="!authStore.isAdmin" class="card notice">仅管理员可操作文档管理功能。</div>
    <template v-else>
      <UploadPanel @uploaded="handleUploaded" />

      <div class="stat-row">
        <div class="stat-card" v-for="stat in stats" :key="stat.label">
          <p class="stat-label">{{ stat.label }}</p>
          <p class="stat-value">{{ stat.value }}</p>
        </div>
      </div>

      <div class="card table-card">
        <div class="table-header">
          <h3>构建任务（最近）</h3>
          <el-button class="btn-ghost" @click="loadJobs">刷新任务</el-button>
        </div>

        <div class="job-table-wrap">
          <el-table class="table-minimal" :data="jobs" empty-text="暂无任务">
            <el-table-column prop="job_id" label="任务ID" min-width="220" show-overflow-tooltip />
            <el-table-column prop="document_id" label="文档ID" min-width="220" show-overflow-tooltip />
            <el-table-column label="状态" width="120">
              <template #default="scope">
                <el-tag size="small" :type="jobStatusMeta(scope.row.status).type">
                  {{ jobStatusMeta(scope.row.status).label }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column label="阶段" width="140">
              <template #default="scope">
                {{ jobStageLabel(scope.row.stage) }}
              </template>
            </el-table-column>
            <el-table-column label="进度" width="180">
              <template #default="scope">
                <el-progress :percentage="Number(scope.row.progress || 0)" :stroke-width="10" />
              </template>
            </el-table-column>
            <el-table-column label="信息" min-width="180" show-overflow-tooltip>
              <template #default="scope">
                {{ scope.row.message || '-' }}
              </template>
            </el-table-column>
            <el-table-column label="更新时间" width="200" show-overflow-tooltip>
              <template #default="scope">
                {{ formatTime(scope.row.updated_at) }}
              </template>
            </el-table-column>
            <el-table-column label="操作" width="120">
              <template #default="scope">
                <el-button
                  v-if="canCancelJob(scope.row)"
                  link
                  type="warning"
                  :loading="Boolean(jobCancelLoadingMap[scope.row.job_id])"
                  @click="cancelJob(scope.row)"
                >
                  取消任务
                </el-button>
                <span v-else>-</span>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </div>

      <div class="card table-card">
        <div class="table-header">
          <h3>文档列表</h3>
          <div class="table-tools">
            <el-input v-model="keyword" class="search-input" placeholder="搜索文件" clearable />
            <el-select v-model="batchChunkStrategy" placeholder="批量分块策略" style="width: 140px">
              <el-option v-for="item in chunkStrategyOptions" :key="item.value" :label="item.label" :value="item.value" />
            </el-select>
            <el-button
              class="btn-ghost"
              :disabled="!selectedDocIds.length"
              :loading="batchBuildLoading"
              @click="handleBatchBuild"
            >
              批量分块({{ selectedDocIds.length }})
            </el-button>
            <el-button
              class="btn-ghost danger-btn"
              :disabled="!selectedDocIds.length"
              :loading="batchDeleteLoading"
              @click="handleBatchDelete"
            >
              批量删除({{ selectedDocIds.length }})
            </el-button>
            <el-button class="btn-ghost" @click="loadDocs">刷新</el-button>
          </div>
        </div>
        <el-table
          ref="docsTableRef"
          class="table-minimal"
          :data="filteredDocs"
          v-loading="docsLoading"
          row-key="document_id"
          @selection-change="handleSelectionChange"
        >
          <el-table-column type="selection" width="52" :reserve-selection="true" />
          <el-table-column prop="filename" label="文件名" min-width="220" />
          <el-table-column prop="file_type" label="类型" width="120" />
          <el-table-column label="大小" width="120">
            <template #default="scope">
              {{ formatFileSize(scope.row.file_size) }}
            </template>
          </el-table-column>
          <el-table-column label="状态" width="120">
            <template #default="scope">
              <el-tag size="small" :type="documentStatusMeta(scope.row.status).type">
                {{ documentStatusMeta(scope.row.status).label }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column label="分块策略" width="150">
            <template #default="scope">
              <el-select
                size="small"
                style="width: 120px"
                :model-value="resolveDocumentStrategy(scope.row)"
                @change="(value) => updateDocumentStrategy(scope.row.document_id, value)"
              >
                <el-option
                  v-for="item in chunkStrategyOptions"
                  :key="item.value"
                  :label="item.label"
                  :value="item.value"
                />
              </el-select>
            </template>
          </el-table-column>
          <el-table-column prop="chunk_count" label="分块数" width="100" />
          <el-table-column label="上传时间" width="200">
            <template #default="scope">
              {{ formatTime(scope.row.uploaded_at) }}
            </template>
          </el-table-column>
          <el-table-column label="操作" width="220" fixed="right">
            <template #default="scope">
              <el-button
                link
                type="primary"
                :loading="Boolean(rowBuildLoadingMap[scope.row.document_id])"
                @click="buildSingleDocument(scope.row)"
              >
                执行分块
              </el-button>
              <el-button
                link
                type="primary"
                :disabled="scope.row.status !== 'ready'"
                @click="openChunkDialog(scope.row)"
              >
                查看分块
              </el-button>
              <el-button
                link
                type="danger"
                :loading="Boolean(rowDeleteLoadingMap[scope.row.document_id])"
                @click="removeSingleDocument(scope.row)"
              >
                删除
              </el-button>
            </template>
          </el-table-column>
        </el-table>
      </div>

      <el-dialog
        v-model="chunkDialog.visible"
        title="分块结果"
        width="72%"
        :close-on-click-modal="false"
        @closed="resetChunkDialog"
      >
        <div v-loading="chunkDialog.loading" class="chunk-dialog-body">
          <div class="chunk-dialog-meta">
            <span>文件：{{ chunkDialog.filename || '-' }}</span>
            <span>文档ID：{{ chunkDialog.documentId || '-' }}</span>
          </div>

          <el-empty
            v-if="!chunkDialog.loading && !chunkDialog.items.length"
            description="暂无分块结果"
            style="padding: 40px 0"
          />

          <div v-for="item in chunkDialog.items" :key="item.chunk_id" class="chunk-item">
            <div class="chunk-item-header">
              <span>Chunk #{{ item.chunk_index }}</span>
              <span>{{ item.chunk_id }}</span>
            </div>
            <div class="chunk-item-row">
              <strong>content</strong>
              <p>{{ item.content || '-' }}</p>
            </div>
            <div class="chunk-item-row">
              <strong>keywords</strong>
              <p>{{ formatList(item.keywords) }}</p>
            </div>
            <div class="chunk-item-row">
              <strong>generated_questions</strong>
              <p>{{ formatList(item.generated_questions) }}</p>
            </div>
            <div class="chunk-item-row">
              <strong>metadata</strong>
              <pre>{{ formatMetadata(item.metadata) }}</pre>
            </div>
          </div>
        </div>

        <template #footer>
          <el-pagination
            v-if="chunkDialog.total > 0"
            background
            layout="prev, pager, next, total"
            :current-page="chunkDialog.page"
            :page-size="chunkDialog.pageSize"
            :total="chunkDialog.total"
            @current-change="loadChunkPage"
          />
        </template>
      </el-dialog>
    </template>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { ElMessage, ElMessageBox } from 'element-plus';
import UploadPanel from '../components/UploadPanel.vue';
import { apiAdapter } from '../api/adapters';
import { useAuthStore } from '../store/auth';

const TERMINAL_JOB_STATUS = ['succeeded', 'failed', 'canceled'];

const documentStatusMap = {
  pending: { label: '待处理', type: 'info' },
  processing: { label: '处理中', type: 'warning' },
  ready: { label: '可检索', type: 'success' },
  failed: { label: '构建失败', type: 'danger' },
  deleting: { label: '删除中', type: 'warning' }
};

const jobStatusMap = {
  queued: { label: '排队中', type: 'info' },
  running: { label: '执行中', type: 'warning' },
  succeeded: { label: '已成功', type: 'success' },
  failed: { label: '失败', type: 'danger' },
  canceled: { label: '已取消', type: 'info' }
};

const jobStageMap = {
  queued: '排队中',
  uploaded: '已上传',
  parsing: '解析中',
  chunking: '分块中',
  completed: '已完成',
  failed: '失败'
};

const chunkStrategyOptions = [
  { value: 'padding', label: '补齐分块' },
  { value: 'general', label: '通用分块' },
  { value: 'book', label: '书籍分块' },
  { value: 'paper', label: '论文分块' },
  { value: 'resume', label: '简历分块' },
  { value: 'table', label: '表格分块' },
  { value: 'qa', label: '问答分块' }
];

const authStore = useAuthStore();
const docsTableRef = ref(null);
const docs = ref([]);
const jobs = ref([]);
const selectedDocs = ref([]);
const docsLoading = ref(false);
const batchBuildLoading = ref(false);
const batchDeleteLoading = ref(false);
const rowBuildLoadingMap = ref({});
const rowDeleteLoadingMap = ref({});
const jobCancelLoadingMap = ref({});
const keyword = ref('');
const batchChunkStrategy = ref('general');
const docStrategyMap = ref({});

const chunkDialog = ref({
  visible: false,
  documentId: '',
  filename: '',
  items: [],
  loading: false,
  page: 1,
  pageSize: 5,
  total: 0
});

const pollingTimers = new Map();

const selectedDocIds = computed(() => selectedDocs.value.map((item) => item.document_id).filter(Boolean));

const stats = computed(() => {
  const queuedOrRunning = jobs.value.filter((item) => ['queued', 'running'].includes(item.status)).length;
  return [
    { label: '文档总数', value: docs.value.length },
    {
      label: '总分块',
      value: docs.value.reduce((sum, item) => sum + (item.chunk_count || 0), 0)
    },
    { label: '进行中任务', value: queuedOrRunning }
  ];
});

const filteredDocs = computed(() => {
  const query = keyword.value.trim().toLowerCase();
  if (!query) return docs.value;
  return docs.value.filter((doc) => doc.filename?.toLowerCase().includes(query));
});

const getFriendlyError = (error, fallback = '请求失败') => {
  if (error?.status === 401) return '登录状态已失效，请重新登录';
  if (error?.status === 403) return '仅管理员可操作文档管理功能';
  const message = error?.message || fallback;
  if (error?.code && !message.includes(error.code)) {
    return `${message} (${error.code})`;
  }
  return message;
};

const documentStatusMeta = (status) => documentStatusMap[status] || { label: status || '-', type: 'info' };
const jobStatusMeta = (status) => jobStatusMap[status] || { label: status || '-', type: 'info' };
const jobStageLabel = (stage) => jobStageMap[stage] || stage || '-';

const formatTime = (value) => {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString('zh-CN', { hour12: false });
};

const formatFileSize = (size) => {
  if (size === null || size === undefined || Number.isNaN(Number(size))) return '-';
  const bytes = Number(size);
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
};

const formatList = (list) => {
  if (!Array.isArray(list) || !list.length) return '-';
  return list.join('、');
};

const formatMetadata = (metadata) => {
  if (!metadata) return '{}';
  try {
    return JSON.stringify(metadata, null, 2);
  } catch {
    return String(metadata);
  }
};

const sortByUpdatedAtDesc = (items) =>
  [...items].sort((a, b) => new Date(b.updated_at || 0).getTime() - new Date(a.updated_at || 0).getTime());

const mergeJob = (job) => {
  if (!job?.job_id) return;
  const index = jobs.value.findIndex((item) => item.job_id === job.job_id);
  if (index >= 0) {
    jobs.value[index] = { ...jobs.value[index], ...job };
  } else {
    jobs.value.unshift(job);
  }
  jobs.value = sortByUpdatedAtDesc(jobs.value).slice(0, 20);
};

const canCancelJob = (job) => ['queued', 'running'].includes(job?.status);

const clearPolling = (jobId) => {
  const timer = pollingTimers.get(jobId);
  if (timer) {
    clearTimeout(timer);
    pollingTimers.delete(jobId);
  }
};

const clearAllPolling = () => {
  pollingTimers.forEach((timer) => clearTimeout(timer));
  pollingTimers.clear();
};

const schedulePoll = (jobId, delayMs = 2000) => {
  clearPolling(jobId);
  const timer = setTimeout(() => pollJob(jobId), delayMs);
  pollingTimers.set(jobId, timer);
};

const pollJob = async (jobId) => {
  if (!jobId || !authStore.isLoggedIn || !authStore.isAdmin) return;

  try {
    const job = await apiAdapter.getDocumentJob(jobId);
    mergeJob(job);

    if (TERMINAL_JOB_STATUS.includes(job?.status)) {
      clearPolling(jobId);
      if (job.status === 'succeeded') {
        await loadDocs();
      }
      return;
    }

    schedulePoll(jobId, job?.status === 'queued' ? 4000 : 2000);
  } catch (error) {
    clearPolling(jobId);
    if (error?.status !== 404) {
      ElMessage.error(getFriendlyError(error, `任务 ${jobId} 查询失败`));
    }
  }
};

const enqueueJob = (item, fallbackDocumentId = '') => {
  const jobId = item?.job_id;
  if (!jobId) return;

  mergeJob({
    job_id: jobId,
    document_id: item?.document_id || fallbackDocumentId,
    status: item?.status || 'queued',
    stage: item?.stage || 'queued',
    progress: item?.progress || 0,
    message: item?.message || '',
    updated_at: item?.updated_at || new Date().toISOString()
  });
  schedulePoll(jobId, 1000);
};

const dropJobsByDocumentIds = (documentIds = []) => {
  const idSet = new Set(documentIds);
  jobs.value.forEach((job) => {
    if (idSet.has(job.document_id)) {
      clearPolling(job.job_id);
    }
  });
  jobs.value = jobs.value.filter((job) => !idSet.has(job.document_id));
};

const resolveDocumentStrategy = (doc) =>
  docStrategyMap.value[doc.document_id] || doc.chunk_strategy || batchChunkStrategy.value;

const updateDocumentStrategy = (documentId, strategy) => {
  docStrategyMap.value[documentId] = strategy;
};

const handleSelectionChange = (rows) => {
  selectedDocs.value = rows || [];
};

const syncLocalStateAfterDocsReload = () => {
  const docIdSet = new Set(docs.value.map((item) => item.document_id));
  selectedDocs.value = selectedDocs.value.filter((item) => docIdSet.has(item.document_id));

  Object.keys(docStrategyMap.value).forEach((documentId) => {
    if (!docIdSet.has(documentId)) {
      delete docStrategyMap.value[documentId];
    }
  });

  if (chunkDialog.value.visible && !docIdSet.has(chunkDialog.value.documentId)) {
    resetChunkDialog();
  }
};

const loadDocs = async () => {
  if (!authStore.isLoggedIn || !authStore.isAdmin) return;
  docsLoading.value = true;
  try {
    const data = await apiAdapter.listDocuments({ page: 1, page_size: 200 });
    docs.value = data?.items || [];
    syncLocalStateAfterDocsReload();
  } catch (error) {
    ElMessage.error(getFriendlyError(error, '加载文档列表失败'));
  } finally {
    docsLoading.value = false;
  }
};

const loadJobs = async () => {
  if (!authStore.isLoggedIn || !authStore.isAdmin) return;
  try {
    const data = await apiAdapter.listDocumentJobs({ page: 1, page_size: 20 });
    jobs.value = sortByUpdatedAtDesc(data?.items || []).slice(0, 20);

    clearAllPolling();
    jobs.value.forEach((job) => {
      if (canCancelJob(job)) {
        schedulePoll(job.job_id, job.status === 'queued' ? 4000 : 2000);
      }
    });
  } catch (error) {
    ElMessage.error(getFriendlyError(error, '加载任务列表失败'));
  }
};

const refreshAll = async () => {
  await Promise.all([loadDocs(), loadJobs()]);
};

const handleUploaded = async (payload) => {
  const jobId = payload?.job_id;
  if (!jobId) {
    ElMessage.warning('上传成功，但未获取到任务ID');
    await loadDocs();
    return;
  }

  enqueueJob(
    {
      job_id: payload.job_id,
      document_id: payload.document_id,
      status: 'queued',
      stage: 'queued',
      progress: 0
    },
    payload?.document_id
  );
};

const buildSingleDocument = async (doc) => {
  const documentId = doc?.document_id;
  if (!documentId) return;

  rowBuildLoadingMap.value[documentId] = true;
  try {
    const response = await apiAdapter.buildDocument(documentId, {
      chunk_strategy: resolveDocumentStrategy(doc)
    });
    enqueueJob(response, documentId);
    ElMessage.success('单文件分块任务已入队');
  } catch (error) {
    ElMessage.error(getFriendlyError(error, '单文件分块失败'));
  } finally {
    delete rowBuildLoadingMap.value[documentId];
  }
};

const handleBatchBuild = async () => {
  if (!selectedDocIds.value.length) {
    ElMessage.warning('请先选择文档');
    return;
  }

  batchBuildLoading.value = true;
  try {
    const response = await apiAdapter.batchBuildDocuments({
      document_ids: selectedDocIds.value,
      chunk_strategy: batchChunkStrategy.value
    });
    const items = response?.items || [];
    items.forEach((item) => enqueueJob(item, item.document_id));
    ElMessage.success(`批量分块已入队 ${items.length} 个任务`);
  } catch (error) {
    ElMessage.error(getFriendlyError(error, '批量分块失败'));
  } finally {
    batchBuildLoading.value = false;
  }
};

const doBatchDelete = async (documentIds, confirmText) => {
  if (!documentIds.length) return;

  await ElMessageBox.confirm(confirmText, '提示', { type: 'warning' });

  const response = await apiAdapter.batchDeleteDocuments({ document_ids: documentIds });
  const successIds = response?.success_ids || [];
  const failedItems = response?.failed_items || [];

  if (successIds.length) {
    dropJobsByDocumentIds(successIds);
    docsTableRef.value?.clearSelection();
    selectedDocs.value = [];
    ElMessage.success(`已删除 ${successIds.length} 个文档`);
  }

  if (failedItems.length) {
    const detail = failedItems
      .slice(0, 3)
      .map((item) => `${item.document_id}: ${item.message}`)
      .join('；');
    ElMessage.warning(`部分删除失败：${detail}${failedItems.length > 3 ? '；...' : ''}`);
  }

  await loadDocs();
};

const handleBatchDelete = async () => {
  if (!selectedDocIds.value.length) {
    ElMessage.warning('请先选择文档');
    return;
  }

  batchDeleteLoading.value = true;
  try {
    await doBatchDelete(selectedDocIds.value, `确认删除已选 ${selectedDocIds.value.length} 个文档？`);
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error(getFriendlyError(error, '批量删除失败'));
    }
  } finally {
    batchDeleteLoading.value = false;
  }
};

const removeSingleDocument = async (doc) => {
  const documentId = doc?.document_id;
  if (!documentId) return;

  rowDeleteLoadingMap.value[documentId] = true;
  try {
    await doBatchDelete([documentId], `确认删除文档 ${doc.filename}？`);
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error(getFriendlyError(error, '删除文档失败'));
    }
  } finally {
    delete rowDeleteLoadingMap.value[documentId];
  }
};

const cancelJob = async (job) => {
  if (!canCancelJob(job)) return;
  jobCancelLoadingMap.value[job.job_id] = true;
  try {
    await apiAdapter.cancelDocumentJob(job.job_id);
    ElMessage.success('取消任务请求已提交');
    await loadJobs();
  } catch (error) {
    ElMessage.error(getFriendlyError(error, '取消任务失败'));
  } finally {
    delete jobCancelLoadingMap.value[job.job_id];
  }
};

const resetChunkDialog = () => {
  chunkDialog.value = {
    ...chunkDialog.value,
    visible: false,
    documentId: '',
    filename: '',
    items: [],
    loading: false,
    page: 1,
    total: 0
  };
};

const loadChunkPage = async (page = 1) => {
  if (!chunkDialog.value.documentId) return;
  chunkDialog.value = {
    ...chunkDialog.value,
    page,
    loading: true
  };

  try {
    const data = await apiAdapter.getDocumentChunks(chunkDialog.value.documentId, {
      page,
      page_size: chunkDialog.value.pageSize
    });
    chunkDialog.value = {
      ...chunkDialog.value,
      items: data?.items || [],
      page: data?.pagination?.page || page,
      total: data?.pagination?.total || 0
    };
  } catch (error) {
    chunkDialog.value = {
      ...chunkDialog.value,
      items: [],
      total: 0
    };
    if (error?.status === 409 || error?.code === 'DOC_CHUNK_RESULT_NOT_READY') {
      ElMessage.warning('分块结果未就绪，请稍后再试');
      return;
    }
    ElMessage.error(getFriendlyError(error, '加载分块结果失败'));
  } finally {
    chunkDialog.value = {
      ...chunkDialog.value,
      loading: false
    };
  }
};

const openChunkDialog = async (doc) => {
  if (doc?.status !== 'ready') {
    ElMessage.warning('文档未就绪，暂不可查看分块结果');
    return;
  }

  chunkDialog.value = {
    ...chunkDialog.value,
    visible: true,
    documentId: doc.document_id,
    filename: doc.filename,
    items: [],
    page: 1,
    total: 0
  };
  await loadChunkPage(1);
};

onMounted(refreshAll);

onBeforeUnmount(() => {
  clearAllPolling();
});
</script>

<style scoped>
.upload-page {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.table-card {
  margin-top: 16px;
}

.job-table-wrap {
  width: 100%;
  padding: 0 24px 24px;
  overflow-x: auto;
}

.job-table-wrap :deep(.el-table) {
  width: 100%;
}

.table-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 24px;
  border-bottom: 1px solid var(--line-soft);
  gap: 12px;
}

.table-header h3 {
  margin: 0;
}

.table-tools {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.search-input {
  width: 220px;
}

.danger-btn {
  border-color: rgba(220, 38, 38, 0.2);
  color: #dc2626;
}

.chunk-dialog-body {
  max-height: 60vh;
  overflow-y: auto;
  padding-right: 8px;
}

.chunk-dialog-meta {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 13px;
  color: var(--text-soft);
  margin-bottom: 12px;
}

.chunk-item {
  border: 1px solid var(--line-soft);
  border-radius: 12px;
  padding: 12px;
  margin-bottom: 12px;
  background: #fff;
}

.chunk-item-header {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 12px;
  color: var(--text-soft);
  margin-bottom: 8px;
}

.chunk-item-row {
  margin-bottom: 8px;
}

.chunk-item-row strong {
  display: block;
  font-size: 12px;
  margin-bottom: 4px;
}

.chunk-item-row p {
  margin: 0;
  line-height: 1.6;
  word-break: break-word;
}

.chunk-item-row pre {
  margin: 0;
  background: var(--bg-shell);
  border: 1px solid var(--line-soft);
  border-radius: 8px;
  padding: 8px;
  overflow-x: auto;
  font-size: 12px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
}

@media (max-width: 900px) {
  .table-header {
    align-items: flex-start;
    flex-direction: column;
  }

  .table-tools {
    width: 100%;
    justify-content: flex-start;
  }

  .search-input {
    width: 100%;
  }

  .chunk-dialog-meta {
    flex-direction: column;
  }
}
</style>
