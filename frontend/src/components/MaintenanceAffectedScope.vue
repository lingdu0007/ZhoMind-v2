<template>
  <section class="affected-scope" aria-label="具体影响范围">
    <h3>具体影响范围</h3>
    <p v-if="!scope">范围记录不可用</p>
    <template v-else>
      <dl v-for="entry in scope.entry_versions" :key="entry.publication_identity">
        <dt>条目</dt><dd>{{ entry.entry_identity }}</dd>
        <dt>发布版本</dt><dd>{{ entry.publication_identity }}</dd>
        <dt>编辑修订</dt><dd>{{ entry.revision_identity }}</dd>
      </dl>
      <dl v-for="gap in scope.gap_contexts" :key="`${gap.query_condition_set_identity}:${gap.reason}`">
        <dt>知识缺口</dt><dd>{{ reasons[gap.reason] || gap.reason }}</dd>
        <dt>条件集</dt><dd>{{ gap.query_condition_set_identity }}</dd>
      </dl>
    </template>
  </section>
</template>

<script setup>
defineProps({ scope: { type: Object, default: null } });
const reasons = {
  no_eligible_published_evidence: '没有合格的已发布证据',
  decision_not_covered: '决策尚未覆盖',
  decisive_condition_missing: '缺少决定性条件',
  material_evidence_conflict: '关键证据冲突',
  assurance_support_missing: '缺少保证依据',
  evidence_budget_exceeded: '超出证据预算',
  knowledge_needs_review: '知识需要评审'
};
</script>

<style scoped>
.affected-scope { min-width: 0; padding: 16px 0; border-top: 1px solid var(--color-rule); }
.affected-scope h3 { margin: 0; font-size: 14px; font-weight: 600; }
.affected-scope dl { display: grid; grid-template-columns: 78px minmax(0, 1fr); gap: 8px 12px; line-height: 1.6; }
.affected-scope dt { color: var(--color-ink-soft); }
.affected-scope dd { margin: 0; overflow-wrap: anywhere; }
</style>
