<template>
  <div class="operations-workspace">
    <header>
      <h1>运行状态</h1>
      <button type="button" aria-label="刷新运行状态" title="刷新运行状态" :disabled="loading" @click="refresh">
        <RefreshCw :size="18" aria-hidden="true" />
      </button>
    </header>
    <p v-if="error" role="alert">{{ error }}</p>
    <section aria-label="请求准入" :aria-busy="loading">
      <h2>请求准入</h2>
      <p v-if="!data && loading" role="status">正在读取</p>
      <dl v-if="data">
        <div><dt>执行中</dt><dd data-testid="admission-executing">{{ data.admission.executing }} / {{ data.admission.configuration.max_executing }}</dd></div>
        <div><dt>排队中</dt><dd data-testid="admission-queued">{{ data.admission.queued }} / {{ data.admission.configuration.max_queued }}</dd></div>
        <div><dt>排队超时</dt><dd>{{ data.admission.configuration.queue_timeout_seconds }} s</dd></div>
        <div><dt>配置版本</dt><dd>{{ data.admission.configuration.version }}</dd></div>
      </dl>
      <code v-if="data">{{ data.admission.configuration.identity }}</code>
    </section>
    <section v-if="data" aria-label="构建任务">
      <h2>构建任务</h2>
      <dl>
        <div><dt>执行中</dt><dd>{{ data.documents.running_builds }}</dd></div>
        <div><dt>排队中</dt><dd>{{ data.documents.queued_builds }}</dd></div>
      </dl>
    </section>
    <section v-if="data" aria-label="阶段耗时">
      <h2>阶段耗时</h2>
      <p v-if="!data.events.length">暂无运维事件</p>
      <div v-else class="table-scroll">
        <table>
          <thead><tr><th>请求</th><th>路径</th><th>执行状态</th><th>结果</th><th>耗时 (ms)</th></tr></thead>
          <tbody>
            <tr v-for="(event, index) in data.events" :key="`${event.request_id}-${index}`">
              <td><code :title="event.request_id">{{ event.request_id.slice(0, 8) }}</code></td>
              <td>{{ event.route_class }}</td>
              <td>{{ event.dimensions.execution_state || '-' }}</td>
              <td>{{ event.dimensions.outcome || '-' }}</td>
              <td>
                <span v-for="(duration, stage) in event.dimensions.stage_durations_ms || {}" :key="stage" class="stage">
                  {{ stage }}: {{ duration }}
                </span>
                <span class="stage">total: {{ event.duration_ms }}</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
    <section v-if="data" aria-label="规范化失败">
      <h2>近期失败</h2>
      <p v-if="!data.failures.length">暂无失败记录</p>
      <ul v-else>
        <li v-for="(failure, index) in data.failures" :key="index">
          <span>{{ failure.kind }}</span> <code>{{ failure.code }}</code>
        </li>
      </ul>
      <nav aria-label="恢复操作">
        <RouterLink to="/reviewed-bundles">构建重试</RouterLink>
        <RouterLink to="/config">生成路由</RouterLink>
      </nav>
      <ul aria-label="可用重试">
        <li v-for="(action, index) in data.retry_actions" :key="index">
          <RouterLink :to="retryTarget(action.action)">{{ retryLabel(action.action) }}</RouterLink>
          <code v-if="action.job_id">{{ action.job_id }}</code>
        </li>
      </ul>
    </section>
    <section v-if="data" aria-label="生成路由结果">
      <h2>生成路由结果</h2>
      <p v-if="!data.generation.route_executions.length">暂无路由执行</p>
      <ul v-else>
        <li v-for="event in data.generation.route_executions" :key="event.request_id">
          <code>{{ event.request_id.slice(0, 8) }}</code>
          <span>{{ event.route_reason }}</span>
          <span v-for="attempt in event.attempts" :key="attempt.attempt">
            #{{ attempt.attempt }} {{ attempt.latency_ms }} ms {{ attempt.error_code || '-' }}
          </span>
        </li>
      </ul>
    </section>
  </div>
</template>

<script setup>
import { onMounted, onBeforeUnmount, ref } from 'vue';
import { RefreshCw } from 'lucide-vue-next';
import { apiAdapter } from '../api/adapters';

const data = ref(null);
const loading = ref(true);
const error = ref('');
let active = true;
let timer;
const retryTarget = (action) => action === 'retry_generation_settings_apply' ? '/config'
  : action === 'retry_document_build' ? '/documents' : '/reviewed-bundles';
const retryLabel = (action) => ({
  retry_candidate_build: 'Candidate 构建重试', retry_document_build: '文档构建重试',
  retry_generation_settings_apply: '生成配置重试'
})[action] || '查看恢复状态';
const refresh = async () => {
  clearTimeout(timer);
  loading.value = true;
  error.value = '';
  try {
    const result = await apiAdapter.getOperations();
    if (active) data.value = result;
  } catch {
    if (active) {
      data.value = null;
      error.value = '运行状态读取失败';
    }
  } finally {
    if (active) {
      loading.value = false;
      clearTimeout(timer);
      timer = setTimeout(refresh, 3000);
    }
  }
};
onMounted(refresh);
onBeforeUnmount(() => {
  active = false;
  clearTimeout(timer);
});
</script>

<style scoped>
.operations-workspace { max-width: 1200px; margin: 0 auto; color: var(--color-ink); }
header { display: flex; align-items: center; justify-content: space-between; gap: 16px; }
h1 { font-size: 28px; }
h2 { font-size: 18px; margin: 0 0 18px; }
button { display: grid; place-items: center; width: 36px; height: 36px; border: 1px solid var(--color-rule); background: transparent; border-radius: 4px; cursor: pointer; }
section { padding: 24px 0; border-top: 1px solid var(--color-rule); }
dl { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; }
dt { font-size: 13px; color: var(--color-ink-soft); }
dd { margin: 8px 0; font-size: 20px; }
code { overflow-wrap: anywhere; font-size: 12px; }
.table-scroll { overflow-x: auto; }
table { width: 100%; min-width: 640px; border-collapse: collapse; font-size: 13px; text-align: left; }
th, td { padding: 12px 8px; border-bottom: 1px solid var(--color-rule); vertical-align: top; overflow-wrap: anywhere; }
th { font-weight: 500; }
.stage { display: block; white-space: nowrap; }
ul { padding: 0; list-style: none; }
li, nav { display: flex; flex-wrap: wrap; gap: 12px; padding: 8px 0; }
@media (max-width: 640px) { dl { grid-template-columns: repeat(2, minmax(0, 1fr)); } h1 { font-size: 24px; } }
</style>
