<template>
  <section class="knowledge-review" :aria-busy="loading">
    <header class="knowledge-review__header">
      <div>
        <p>Editorial work</p>
        <h1>知识复核</h1>
      </div>
      <span v-if="!loading">{{ pendingCount }} 条待处理</span>
    </header>

    <p v-if="error" class="knowledge-review__state" role="alert">{{ error }}</p>
    <p v-else-if="loading" class="knowledge-review__state" role="status">正在加载复核队列…</p>
    <p v-else-if="items.length === 0" class="knowledge-review__state">当前没有复核事项。</p>

    <div v-else class="knowledge-review__list">
      <article
        v-for="item in items"
        :key="item.id"
        class="review-item"
        :aria-label="`${item.subject_id} ${kindLabel(item.kind)}`"
      >
        <header>
          <div>
            <span>{{ kindLabel(item.kind) }}</span>
            <h2>{{ item.subject_id }}</h2>
          </div>
          <strong :class="`review-item__status--${item.status}`">{{ statusLabel(item.status) }}</strong>
        </header>
        <dl>
          <div v-if="item.metadata.label"><dt>反馈类型</dt><dd>{{ feedbackLabel(item.metadata.label) }}</dd></div>
          <div v-if="item.metadata.knowledge_edition"><dt>知识版本</dt><dd>{{ item.metadata.knowledge_edition }}</dd></div>
          <div v-if="item.metadata.review_date"><dt>复核日期</dt><dd>{{ item.metadata.review_date }}</dd></div>
          <div v-if="item.metadata.failed_source_count"><dt>不可用来源</dt><dd>{{ item.metadata.failed_source_count }}</dd></div>
          <div v-if="item.metadata.candidate_version"><dt>版本变化</dt><dd>{{ item.metadata.published_version }} → {{ item.metadata.candidate_version }}</dd></div>
        </dl>
        <p v-if="item.metadata.note" class="review-item__note">{{ item.metadata.note }}</p>
        <div class="review-item__classification">
          <label>
            <span>质量分级</span>
            <select v-model="item.draftClassification" aria-label="质量分级">
              <option value="p0">P0</option>
              <option value="p1">P1</option>
              <option value="p2">P2</option>
              <option value="p3">P3</option>
              <option value="no_action">无需处理</option>
            </select>
          </label>
          <label>
            <span>处理状态</span>
            <select v-model="item.draftStatus" aria-label="处理状态">
              <option value="reviewed">已复核</option>
              <option value="dismissed">已忽略</option>
            </select>
          </label>
          <button type="button" :disabled="item.saving" @click="classify(item)">
            <Save :size="15" aria-hidden="true" />
            <span>保存分类</span>
          </button>
        </div>
      </article>
    </div>
  </section>
</template>

<script setup>
import { onMounted, ref } from 'vue';
import { Save } from 'lucide-vue-next';
import { apiAdapter } from '../api/adapters';

const loading = ref(true);
const error = ref('');
const pendingCount = ref(0);
const items = ref([]);

const kindLabel = (kind) => ({
  feedback_signal: '用户反馈',
  source_link_failure: '来源失效',
  release_change: '版本变化',
  review_age: '复核到期'
}[kind] || kind);

const feedbackLabel = (label) => ({
  helpful: '有帮助',
  insufficient_evidence: '证据不足',
  outdated: '已过时',
  out_of_scope: '超出范围'
}[label] || label);

const statusLabel = (status) => ({ pending: '待处理', reviewed: '已复核', dismissed: '已忽略' }[status] || status);

const load = async () => {
  loading.value = true;
  error.value = '';
  try {
    const payload = await apiAdapter.listKnowledgeReviewQueue();
    pendingCount.value = payload.pending_count || 0;
    items.value = (payload.items || []).map((item) => ({
      ...item,
      draftClassification: item.classification || 'p2',
      draftStatus: item.status === 'dismissed' ? 'dismissed' : 'reviewed',
      saving: false
    }));
  } catch (requestError) {
    error.value = requestError.message || '复核队列加载失败。';
  } finally {
    loading.value = false;
  }
};

const classify = async (item) => {
  item.saving = true;
  try {
    const updated = await apiAdapter.classifyKnowledgeReviewItem(item.id, {
      classification: item.draftClassification,
      status: item.draftStatus
    });
    item.status = updated.status;
    item.classification = updated.classification;
    pendingCount.value = items.value.filter((candidate) => candidate.status === 'pending').length;
  } catch (requestError) {
    error.value = requestError.message || '复核分类保存失败。';
  } finally {
    item.saving = false;
  }
};

onMounted(load);
</script>

<style scoped>
.knowledge-review {
  width: min(100%, 960px);
  margin: 0 auto;
}

.knowledge-review__header,
.review-item header,
.review-item__classification {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
}

.knowledge-review__header {
  padding-bottom: 20px;
  border-bottom: 1px solid var(--color-rule);
}

.knowledge-review__header p,
.knowledge-review__header h1,
.review-item h2,
.review-item p,
.review-item dl,
.review-item dd {
  margin: 0;
}

.knowledge-review__header p,
.review-item header span {
  color: var(--color-copper-strong);
  font-family: var(--font-mono);
  font-size: 11px;
  line-height: 1.5;
}

.knowledge-review__header h1 {
  margin-top: 4px;
  font-family: var(--font-display);
  font-size: 30px;
  font-weight: 600;
}

.knowledge-review__header > span,
.knowledge-review__state {
  color: var(--color-ink-soft);
  font-size: 13px;
}

.knowledge-review__state {
  padding: 32px 0;
}

.knowledge-review__list {
  display: grid;
}

.review-item {
  padding: 22px 0;
  border-bottom: 1px solid var(--color-rule);
}

.review-item h2 {
  margin-top: 3px;
  overflow-wrap: anywhere;
  font-size: 16px;
  font-weight: 600;
}

.review-item header strong {
  font-size: 12px;
}

.review-item__status--pending {
  color: var(--color-warning);
}

.review-item__status--reviewed {
  color: var(--color-moss);
}

.review-item__status--dismissed {
  color: var(--color-ink-soft);
}

.review-item dl {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 24px;
  padding-top: 14px;
}

.review-item dl div {
  display: flex;
  gap: 8px;
  font-size: 12px;
}

.review-item dt {
  color: var(--color-ink-soft);
}

.review-item__note {
  margin-top: 14px !important;
  padding-left: 12px;
  border-left: 2px solid var(--color-moss);
  color: var(--color-ink);
  font-size: 13px;
  line-height: 1.65;
}

.review-item__classification {
  justify-content: flex-start;
  margin-top: 16px;
}

.review-item__classification label {
  display: grid;
  gap: 4px;
  color: var(--color-ink-soft);
  font-size: 11px;
}

.review-item__classification select,
.review-item__classification button {
  height: 34px;
  border: 1px solid var(--color-rule);
  border-radius: var(--radius-control);
  background: var(--color-paper-raised);
  color: var(--color-ink);
  font: inherit;
  font-size: 12px;
}

.review-item__classification select {
  min-width: 112px;
  padding: 0 8px;
}

.review-item__classification button {
  align-self: end;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 0 10px;
  cursor: pointer;
}

@media (max-width: 640px) {
  .review-item__classification {
    align-items: stretch;
    flex-direction: column;
    gap: 10px;
  }

  .review-item__classification select,
  .review-item__classification button {
    width: 100%;
  }

  .review-item__classification button {
    justify-content: center;
  }
}
</style>
