<template>
  <section class="retrieval-diagnostics" aria-label="检索诊断">
    <details>
      <summary>
        <span>检索诊断</span>
        <span class="retrieval-diagnostics__scope">仅系统管理员</span>
      </summary>

      <div class="retrieval-diagnostics__content">
        <section class="retrieval-diagnostics__section" aria-label="检索时间线">
          <h3>检索时间线</h3>
          <ol v-if="timeline.length" class="retrieval-diagnostics__timeline">
            <li v-for="(item, index) in timeline" :key="`${item.step}-${index}`">
              <code>{{ item.step }}</code>
            </li>
          </ol>
          <p v-else>未返回检索时间线。</p>
        </section>

        <dl class="retrieval-diagnostics__facts">
          <div>
            <dt>候选数</dt>
            <dd>召回候选 {{ displayCount(candidateCounts.retrieved) }}</dd>
            <dd>重排候选 {{ displayCount(candidateCounts.reranked) }}</dd>
          </div>
          <div>
            <dt>证据门禁</dt>
            <dd>{{ gateLabel }}</dd>
            <dd>{{ displayValue(evidenceGate.reason) }}</dd>
          </div>
          <div>
            <dt>回退状态</dt>
            <dd>{{ fallbackLabel }}</dd>
            <dd>{{ displayValue(fallback.final_provider) }}</dd>
          </div>
        </dl>

        <section v-if="providerErrors.length" class="retrieval-diagnostics__section" aria-label="Provider 错误">
          <h3>Provider 错误</h3>
          <ul class="retrieval-diagnostics__errors">
            <li v-for="(error, index) in providerErrors" :key="`${error.stage}-${error.code}-${index}`">
              <code>{{ displayValue(error.stage) }}</code>
              <span>{{ displayValue(error.code) }}</span>
              <span>{{ displayValue(error.type) }}</span>
            </li>
          </ul>
        </section>

        <section class="retrieval-diagnostics__section" aria-label="脱敏 trace 预览">
          <h3>脱敏 trace 预览</h3>
          <pre>{{ tracePreview || '未返回 trace 预览。' }}</pre>
        </section>
      </div>
    </details>
  </section>
</template>

<script setup>
import { computed } from 'vue';

const props = defineProps({
  diagnostics: {
    type: Object,
    required: true
  }
});

const timeline = computed(() => (Array.isArray(props.diagnostics?.timeline) ? props.diagnostics.timeline : []));
const candidateCounts = computed(() => props.diagnostics?.candidate_counts || {});
const evidenceGate = computed(() => props.diagnostics?.evidence_gate || {});
const fallback = computed(() => props.diagnostics?.fallback || {});
const providerErrors = computed(() => (Array.isArray(props.diagnostics?.provider_errors) ? props.diagnostics.provider_errors : []));
const tracePreview = computed(() => (typeof props.diagnostics?.trace_preview === 'string' ? props.diagnostics.trace_preview : ''));

const displayValue = (value) => (value === null || value === undefined || value === '' ? '不可用' : String(value));
const displayCount = (value) => (Number.isInteger(value) && value >= 0 ? value : '不可用');

const gateLabel = computed(() => {
  if (evidenceGate.value.outcome === 'passed') return '门禁通过';
  if (evidenceGate.value.outcome === 'rejected') return '门禁拒绝';
  return '门禁不可用';
});

const fallbackLabel = computed(() => {
  if (fallback.value.state === 'used' && Number.isInteger(fallback.value.hops)) return `已回退 ${fallback.value.hops} 次`;
  if (fallback.value.state === 'not_used') return '未使用回退';
  return '回退状态不可用';
});
</script>

<style scoped>
.retrieval-diagnostics {
  margin-top: 12px;
  border-top: 1px solid var(--color-rule);
  color: var(--color-ink-soft);
  font-size: 12px;
}

.retrieval-diagnostics details {
  padding-top: 10px;
}

.retrieval-diagnostics summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  color: var(--color-ink-soft);
  cursor: pointer;
  font-weight: 600;
  list-style: none;
}

.retrieval-diagnostics summary::-webkit-details-marker {
  display: none;
}

.retrieval-diagnostics summary::before {
  content: '▸';
  color: var(--color-copper-strong);
  font-size: 14px;
}

.retrieval-diagnostics details[open] summary::before {
  content: '▾';
}

.retrieval-diagnostics__scope {
  color: var(--color-ink-soft);
  font-size: 11px;
  font-weight: 400;
}

.retrieval-diagnostics__content {
  display: grid;
  gap: 14px;
  margin-top: 12px;
  padding: 14px 0 2px;
  border-top: 1px solid var(--color-rule);
}

.retrieval-diagnostics details[open] .retrieval-diagnostics__content {
  animation: retrieval-diagnostics-expand 200ms ease-out both;
}

.retrieval-diagnostics__section h3,
.retrieval-diagnostics__section p {
  margin: 0;
}

.retrieval-diagnostics__section h3,
.retrieval-diagnostics__facts dt {
  color: var(--color-ink);
  font-size: 12px;
  font-weight: 600;
}

.retrieval-diagnostics__section p {
  margin-top: 6px;
}

.retrieval-diagnostics__timeline,
.retrieval-diagnostics__errors {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin: 8px 0 0;
  padding: 0;
  list-style: none;
}

.retrieval-diagnostics__timeline li,
.retrieval-diagnostics__errors li {
  padding: 4px 6px;
  border: 1px solid var(--color-rule);
  background: var(--color-paper-muted);
}

.retrieval-diagnostics__facts {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin: 0;
}

.retrieval-diagnostics__facts div {
  min-width: 0;
  padding-left: 8px;
  border-left: 2px solid var(--color-rule);
}

.retrieval-diagnostics__facts dt,
.retrieval-diagnostics__facts dd {
  margin: 0;
}

.retrieval-diagnostics__facts dd {
  margin-top: 4px;
  overflow-wrap: anywhere;
}

.retrieval-diagnostics__errors li {
  display: flex;
  gap: 6px;
  overflow-wrap: anywhere;
}

.retrieval-diagnostics pre {
  max-height: 180px;
  margin: 8px 0 0;
  padding: 10px;
  overflow: auto;
  border: 1px solid var(--color-rule);
  background: var(--color-paper-muted);
  color: var(--color-ink-soft);
  font-size: 11px;
  line-height: 1.55;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

@keyframes retrieval-diagnostics-expand {
  from {
    opacity: 0;
    transform: translateY(-4px);
  }

  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (prefers-reduced-motion: reduce) {
  .retrieval-diagnostics details[open] .retrieval-diagnostics__content {
    animation: none;
  }
}

@media (max-width: 640px) {
  .retrieval-diagnostics__facts {
    grid-template-columns: 1fr;
  }
}
</style>
