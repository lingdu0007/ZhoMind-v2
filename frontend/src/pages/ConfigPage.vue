<template>
  <section class="config-page">
    <div class="top-bar">
      <h1>配置页</h1>
      <div class="top-actions">
        <el-button class="btn-ghost" @click="load">重置</el-button>
        <el-button type="primary" :loading="configStore.loading" @click="save">保存全部</el-button>
      </div>
    </div>

    <div class="config-grid">
      <div class="card config-card">
        <header>
          <h3>模型配置</h3>
          <p>LLM / Embedding / Rerank</p>
        </header>
        <el-form label-width="120px">
          <el-form-item label="LLM 模型">
            <el-input v-model="configStore.config.llm_model" />
          </el-form-item>
          <el-form-item label="Embedding">
            <el-input v-model="configStore.config.embedding_model" />
          </el-form-item>
          <el-form-item label="Rerank">
            <el-input v-model="configStore.config.rerank_model" />
          </el-form-item>
        </el-form>
        <footer>
          <small v-if="savedAt">已保存 · {{ savedAt }}</small>
        </footer>
      </div>

      <div class="card config-card">
        <header>
          <h3>检索参数</h3>
          <p>TopK / 阈值 / 混合权重</p>
        </header>
        <div class="slider-row">
          <label>Top K</label>
          <el-input-number v-model="configStore.config.top_k" :min="1" :max="50" />
        </div>
        <div class="slider-row">
          <label>相似度阈值</label>
          <el-slider v-model="configStore.config.score_threshold" :min="0" :max="1" :step="0.01" />
        </div>
        <div class="slider-row">
          <label>Dense 权重</label>
          <el-slider v-model="configStore.config.hybrid_dense_weight" :min="0" :max="1" :step="0.01" />
        </div>
        <div class="slider-row">
          <label>Sparse 权重</label>
          <el-slider v-model="configStore.config.hybrid_sparse_weight" :min="0" :max="1" :step="0.01" />
        </div>
        <p class="ratio">当前：Dense {{ denseLabel }} · Sparse {{ sparseLabel }}</p>
      </div>

      <div class="card config-card">
        <header>
          <h3>存储配置</h3>
          <el-tag type="success">手动管理</el-tag>
        </header>
        <el-form label-width="120px">
          <el-form-item label="Milvus URI">
            <div class="input-copy">
              <el-input v-model="configStore.config.milvus_uri" />
              <el-button text @click="copy(configStore.config.milvus_uri)">复制</el-button>
            </div>
          </el-form-item>
          <el-form-item label="Redis URL">
            <div class="input-copy">
              <el-input v-model="configStore.config.redis_url" />
              <el-button text @click="copy(configStore.config.redis_url)">复制</el-button>
            </div>
          </el-form-item>
          <el-form-item label="BM25 路径">
            <el-input v-model="configStore.config.storage_path" />
          </el-form-item>
        </el-form>
      </div>

      <div class="card config-card">
        <header>
          <h3>安全 & API</h3>
          <p>硅基流动密钥 / 回调 / 限流</p>
        </header>
        <el-form label-width="120px">
          <el-form-item label="API Key">
            <div class="input-copy">
              <el-input v-model="configStore.config.silicon_api_key" type="password" show-password />
              <el-button text @click="copy(configStore.config.silicon_api_key)">复制</el-button>
            </div>
          </el-form-item>
          <el-form-item label="回调 URL">
            <el-input v-model="configStore.config.callback_url" />
          </el-form-item>
          <el-form-item label="限流（次/分）">
            <el-input-number v-model="configStore.config.rate_limit_per_minute" :min="1" :max="500" />
          </el-form-item>
        </el-form>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue';
import { ElMessage } from 'element-plus';
import { useConfigStore } from '../store/config';

const configStore = useConfigStore();
const savedAt = ref('');

const denseLabel = computed(() => configStore.config.hybrid_dense_weight?.toFixed(2) ?? '0.00');
const sparseLabel = computed(() => configStore.config.hybrid_sparse_weight?.toFixed(2) ?? '0.00');

const load = async () => {
  try {
    await configStore.fetchConfig();
    ElMessage.success('配置已加载');
  } catch (error) {
    ElMessage.error(error.message || '加载配置失败');
  }
};

const save = async () => {
  try {
    await configStore.saveConfig();
    savedAt.value = new Date().toLocaleTimeString('zh-CN', { hour12: false });
    ElMessage.success('配置保存成功');
  } catch (error) {
    ElMessage.error(error.message || '保存配置失败');
  }
};

const copy = async (text) => {
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    ElMessage.success('已复制');
  } catch {
    ElMessage.warning('复制失败，请手动选择');
  }
};

onMounted(load);
</script>

<style scoped>
.config-page {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.top-actions {
  display: flex;
  gap: 12px;
}

.config-card footer {
  display: flex;
  justify-content: flex-end;
  color: var(--text-muted);
}

.slider-row {
  display: flex;
  align-items: center;
  gap: 16px;
}

.slider-row label {
  width: 120px;
  font-size: 14px;
  color: var(--text-muted);
}

.ratio {
  margin: 8px 0 0;
  font-size: 13px;
  color: var(--text-muted);
}

.input-copy {
  display: flex;
  align-items: center;
  gap: 8px;
}
</style>
