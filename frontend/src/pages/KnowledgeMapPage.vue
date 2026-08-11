<template>
  <section class="knowledge-map" :aria-busy="loading">
    <header class="knowledge-map__header">
      <div>
        <p>Published knowledge</p>
        <h1>知识地图</h1>
      </div>
      <span v-if="!loading">{{ totalEntries }} 条已发布决策</span>
    </header>

    <p v-if="error" class="knowledge-map__state" role="alert">{{ error }}</p>
    <p v-else-if="loading" class="knowledge-map__state" role="status">正在加载已发布知识…</p>
    <p v-else-if="themes.length === 0" class="knowledge-map__state">当前没有可浏览的已发布决策。</p>

    <section v-for="theme in themes" :key="theme.domain" class="knowledge-theme">
      <header class="knowledge-theme__header">
        <h2>{{ theme.label }}</h2>
        <span>{{ theme.entries.length }}</span>
      </header>
      <article
        v-for="entry in theme.entries"
        :key="entry.entry_id"
        class="knowledge-entry"
        :aria-label="entry.title"
      >
        <div class="knowledge-entry__identity">
          <span>{{ entry.entry_id }}</span>
          <h3>{{ entry.title }}</h3>
        </div>
        <p class="knowledge-entry__summary">{{ entry.approved_summary }}</p>
        <dl class="knowledge-entry__cues">
          <div><dt>知识版本</dt><dd>{{ entry.publication_version }}</dd></div>
          <div><dt>复核日期</dt><dd>{{ entry.review_date }}</dd></div>
          <div><dt>公开来源</dt><dd>{{ entry.public_source_count }} 个公开来源</dd></div>
        </dl>
        <div class="knowledge-entry__versions">
          <span v-for="version in entry.applicable_versions" :key="version">{{ version }}</span>
        </div>
        <ul class="knowledge-entry__sources" aria-label="公开来源">
          <li v-for="source in entry.sources" :key="source.url">
            <a
              v-if="safeSourceUrl(source.url)"
              :href="safeSourceUrl(source.url)"
              target="_blank"
              rel="noopener noreferrer"
            >{{ source.title }}</a>
            <span>{{ source.authority }} · {{ source.version }}</span>
          </li>
        </ul>
        <button type="button" class="knowledge-entry__ask" @click="ask(entry.suggested_query)">
          <MessageCircle :size="16" aria-hidden="true" />
          <span>基于此条提问</span>
        </button>
      </article>
    </section>
  </section>
</template>

<script setup>
import { onMounted, ref } from 'vue';
import { useRouter } from 'vue-router';
import { MessageCircle } from 'lucide-vue-next';
import { apiAdapter } from '../api/adapters';
import { getEvidenceSourceUrl } from '../app/evidence-summary';

const router = useRouter();
const loading = ref(true);
const error = ref('');
const totalEntries = ref(0);
const themes = ref([]);

const safeSourceUrl = (url) => getEvidenceSourceUrl({ source_url: url });

const load = async () => {
  loading.value = true;
  error.value = '';
  try {
    const payload = await apiAdapter.getKnowledgeMap();
    totalEntries.value = payload.total_entries || 0;
    themes.value = Array.isArray(payload.themes) ? payload.themes : [];
  } catch (requestError) {
    error.value = requestError.message || '知识地图加载失败。';
  } finally {
    loading.value = false;
  }
};

const ask = (query) => router.push({ name: 'chat', query: { q: query } });

onMounted(load);
</script>

<style scoped>
.knowledge-map {
  width: min(100%, 1040px);
  margin: 0 auto;
}

.knowledge-map__header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 24px;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--color-rule);
}

.knowledge-map__header p,
.knowledge-map__header h1,
.knowledge-theme__header h2,
.knowledge-entry h3,
.knowledge-entry p,
.knowledge-entry dl,
.knowledge-entry dd {
  margin: 0;
}

.knowledge-map__header p,
.knowledge-entry__identity span {
  color: var(--color-copper-strong);
  font-family: var(--font-mono);
  font-size: 11px;
  line-height: 1.5;
}

.knowledge-map__header h1 {
  margin-top: 4px;
  font-family: var(--font-display);
  font-size: 30px;
  font-weight: 600;
  line-height: 1.25;
}

.knowledge-map__header > span,
.knowledge-theme__header span {
  color: var(--color-ink-soft);
  font-size: 12px;
}

.knowledge-map__state {
  padding: 32px 0;
  color: var(--color-ink-soft);
}

.knowledge-theme {
  padding-top: 28px;
}

.knowledge-theme__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding-bottom: 10px;
  border-bottom: 2px solid var(--color-ink);
}

.knowledge-theme__header h2 {
  font-family: var(--font-display);
  font-size: 20px;
  font-weight: 600;
  line-height: 1.4;
}

.knowledge-entry {
  display: grid;
  grid-template-columns: minmax(180px, 0.8fr) minmax(260px, 1.5fr) minmax(180px, 0.8fr);
  gap: 14px 24px;
  padding: 22px 0;
  border-bottom: 1px solid var(--color-rule);
}

.knowledge-entry h3 {
  margin-top: 4px;
  overflow-wrap: anywhere;
  font-size: 16px;
  font-weight: 600;
  line-height: 1.45;
}

.knowledge-entry__summary {
  color: var(--color-ink);
  font-size: 14px;
  line-height: 1.75;
}

.knowledge-entry__cues {
  display: grid;
  gap: 6px;
}

.knowledge-entry__cues div {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 12px;
  line-height: 1.5;
}

.knowledge-entry__cues dt {
  color: var(--color-ink-soft);
}

.knowledge-entry__versions,
.knowledge-entry__sources,
.knowledge-entry__ask {
  grid-column: 2 / -1;
}

.knowledge-entry__versions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.knowledge-entry__versions span {
  padding: 3px 7px;
  border: 1px solid var(--color-rule);
  border-radius: var(--radius-control);
  color: var(--color-ink-soft);
  font-size: 11px;
}

.knowledge-entry__sources {
  display: grid;
  gap: 7px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.knowledge-entry__sources li {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 8px 16px;
  font-size: 12px;
  line-height: 1.5;
}

.knowledge-entry__sources a {
  color: var(--color-copper-strong);
  font-weight: 600;
  text-underline-offset: 3px;
}

.knowledge-entry__sources span {
  color: var(--color-ink-soft);
}

.knowledge-entry__ask {
  justify-self: start;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  min-height: 34px;
  padding: 6px 10px;
  border: 1px solid var(--color-rule);
  border-radius: var(--radius-control);
  background: var(--color-paper-raised);
  color: var(--color-ink);
  cursor: pointer;
  font: inherit;
  font-size: 12px;
}

.knowledge-entry__ask:hover {
  border-color: var(--color-copper);
  color: var(--color-copper-strong);
}

@media (max-width: 760px) {
  .knowledge-map__header {
    align-items: flex-start;
  }

  .knowledge-entry {
    grid-template-columns: minmax(0, 1fr);
  }

  .knowledge-entry__versions,
  .knowledge-entry__sources,
  .knowledge-entry__ask {
    grid-column: 1;
  }
}
</style>
