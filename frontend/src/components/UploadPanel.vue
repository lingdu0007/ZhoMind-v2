<template>
  <section class="document-upload" aria-labelledby="document-upload-title">
    <div class="document-upload__copy">
      <p class="document-upload__eyebrow">初始导入</p>
      <h2 id="document-upload-title">上传文档</h2>
      <p>支持的格式：TXT、Markdown (.md)、PDF。</p>
      <p>初始上传将自动使用通用分块策略。</p>
    </div>

    <div class="document-upload__controls">
      <label
        class="document-upload__picker"
        :class="{ 'document-upload__picker--disabled': loading }"
        for="document-file"
        :aria-disabled="loading"
      >
        <Upload :size="18" aria-hidden="true" />
        <span>选择文档</span>
      </label>
      <input
        id="document-file"
        ref="fileInput"
        class="document-upload__input"
        type="file"
        accept=".txt,.md,.pdf,text/plain,text/markdown,application/pdf"
        aria-describedby="document-upload-formats"
        :disabled="loading"
        @change="handleFileChange"
      />
      <p id="document-upload-formats" class="sr-only">仅支持 TXT、Markdown 和 PDF。</p>
      <p class="document-upload__selection" aria-live="polite">
        {{ file ? `${file.name} (${formatFileSize(file.size)})` : '尚未选择文件' }}
      </p>
      <button class="document-upload__submit" type="button" :disabled="!file || loading" @click="submit">
        <Upload :size="16" aria-hidden="true" />
        <span>{{ loading ? '正在上传' : '上传文档' }}</span>
      </button>
    </div>

    <p v-if="errorMessage" class="document-upload__error" role="alert">{{ errorMessage }}</p>
  </section>
</template>

<script setup>
import { ref } from 'vue';
import { Upload } from 'lucide-vue-next';
import { apiAdapter } from '../api/adapters';

const emit = defineEmits(['uploaded']);

const supportedExtensions = new Set(['txt', 'md', 'pdf']);
const fileInput = ref(null);
const file = ref(null);
const loading = ref(false);
const errorMessage = ref('');

const extensionOf = (filename) => filename?.split('.').pop()?.toLowerCase() || '';

const formatFileSize = (size) => {
  const bytes = Number(size);
  if (!Number.isFinite(bytes)) return '-';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const handleFileChange = (event) => {
  const selectedFile = event.target.files?.[0] || null;
  errorMessage.value = '';

  if (!selectedFile) {
    file.value = null;
    return;
  }

  if (!supportedExtensions.has(extensionOf(selectedFile.name))) {
    file.value = null;
    event.target.value = '';
    errorMessage.value = '不支持的文件格式。仅支持 TXT、Markdown (.md) 和 PDF。';
    return;
  }

  file.value = selectedFile;
};

const uploadErrorMessage = (error) => {
  if (error?.status === 401) return '登录状态已失效，请重新登录。';
  if (error?.status === 403) return '当前账户无权上传文档。';
  if (error?.status === 415) return '服务端不支持该文件格式。仅支持 TXT、Markdown 和 PDF。';
  if (error?.status >= 400 && error?.status < 500) return '上传被服务端拒绝，请检查文件名或内容后重试。';
  return '上传失败，请稍后重试。';
};

const submit = async () => {
  if (!file.value || loading.value) return;

  loading.value = true;
  errorMessage.value = '';
  try {
    const selectedFile = file.value;
    const formData = new FormData();
    formData.append('file', selectedFile);
    const result = await apiAdapter.uploadDocument(formData);

    file.value = null;
    if (fileInput.value) fileInput.value.value = '';
    emit('uploaded', {
      document_id: result?.document_id || '',
      job_id: result?.job_id || '',
      filename: selectedFile.name
    });
  } catch (error) {
    errorMessage.value = uploadErrorMessage(error);
  } finally {
    loading.value = false;
  }
};
</script>

<style scoped>
.document-upload { display: grid; grid-template-columns: minmax(0, 1fr) minmax(330px, 0.9fr); gap: var(--space-5); padding: var(--space-5) 0; border-bottom: 1px solid var(--color-rule); }
.document-upload__copy h2 { margin: 0; font-size: 18px; font-weight: 600; }
.document-upload__copy p { margin: var(--space-2) 0 0; color: var(--color-ink-soft); font-size: 13px; line-height: 1.6; }
.document-upload__copy .document-upload__eyebrow { margin: 0 0 var(--space-2); color: var(--color-moss); font-size: 12px; font-weight: 600; }
.document-upload__controls { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; align-items: center; gap: var(--space-3); }
.document-upload__picker, .document-upload__submit { min-height: 34px; display: inline-flex; align-items: center; justify-content: center; gap: 6px; border-radius: var(--radius-control); font: inherit; font-size: 13px; cursor: pointer; }
.document-upload__picker { padding: 0 var(--space-3); border: 1px solid var(--color-rule); background: var(--color-paper-raised); color: var(--color-ink); }
.document-upload__picker:hover { border-color: var(--color-copper); color: var(--color-copper-strong); }
.document-upload__picker:active { background: var(--color-paper-muted); }
.document-upload__controls:focus-within .document-upload__picker { outline: 2px solid var(--color-focus); outline-offset: 2px; }
.document-upload__picker--disabled { cursor: not-allowed; opacity: 0.65; }
.document-upload__input { position: absolute; width: 1px; height: 1px; overflow: hidden; opacity: 0; pointer-events: none; }
.document-upload__selection { min-width: 0; margin: 0; overflow: hidden; color: var(--color-ink-soft); font-family: var(--font-mono); font-size: 12px; text-overflow: ellipsis; white-space: nowrap; }
.document-upload__submit { min-width: 102px; padding: 0 var(--space-3); border: 1px solid var(--color-copper); background: var(--color-copper); color: var(--color-paper-raised); }
.document-upload__submit:not(:disabled):hover { border-color: var(--color-copper-strong); background: var(--color-copper-strong); }
.document-upload__submit:not(:disabled):active { transform: translateY(1px); }
.document-upload__submit:disabled { cursor: not-allowed; opacity: 0.65; }
.document-upload__error { grid-column: 1 / -1; margin: 0; padding: var(--space-3) var(--space-4); border-left: 3px solid var(--color-danger); background: var(--color-danger-soft); color: var(--color-danger); font-size: 13px; }
@media (max-width: 1024px) { .document-upload { grid-template-columns: 1fr; } }
</style>
