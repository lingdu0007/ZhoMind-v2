<template>
  <div class="upload-card card">
    <el-upload
      ref="uploadRef"
      drag
      :auto-upload="false"
      :on-change="handleFileChange"
      :on-remove="handleFileRemove"
      :limit="1"
      :show-file-list="true"
      accept=".txt,.md,.pdf"
      class="upload-drop"
    >
      <el-icon><upload-filled /></el-icon>
      <div class="el-upload__text">拖拽文件到这里，或 <em>点击选择</em></div>
      <template #tip>
        <div class="el-upload__tip">仅支持 txt / md / pdf 格式</div>
      </template>
    </el-upload>

    <div class="actions">
      <el-button class="btn-primary" :loading="loading" @click="submit">上传并解析</el-button>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import { ElMessage } from 'element-plus';
import { UploadFilled } from '@element-plus/icons-vue';
import { apiAdapter } from '../api/adapters';

const emit = defineEmits(['uploaded']);

const uploadRef = ref(null);
const file = ref(null);
const loading = ref(false);

const handleFileChange = (rawFile) => {
  file.value = rawFile.raw;
};

const handleFileRemove = () => {
  file.value = null;
};

const submit = async () => {
  if (!file.value) {
    ElMessage.warning('请先选择文件');
    return;
  }
  loading.value = true;
  try {
    const selectedFile = file.value;
    const formData = new FormData();
    formData.append('file', selectedFile);
    const result = await apiAdapter.uploadDocument(formData);

    const jobId = result?.job_id || '';
    const documentId = result?.document_id || '';

    ElMessage.success(`已入队，正在构建索引${jobId ? `（任务ID: ${jobId}）` : ''}`);
    file.value = null;
    uploadRef.value?.clearFiles();
    emit('uploaded', {
      job_id: jobId,
      document_id: documentId,
      filename: selectedFile?.name || ''
    });
  } catch (error) {
    const message = error?.message || '上传失败';
    ElMessage.error(error?.code ? `${message} (${error.code})` : message);
  } finally {
    loading.value = false;
  }
};
</script>

<style scoped>
.upload-card {
  padding: 32px;
}

.upload-drop {
  border: 1px dashed var(--line-strong);
  border-radius: 16px;
  background: var(--bg-shell);
  padding: 12px 0;
}

.actions {
  margin-top: 24px;
  display: flex;
  justify-content: flex-end;
}
</style>
