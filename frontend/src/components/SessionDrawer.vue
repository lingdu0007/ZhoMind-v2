<template>
  <aside class="session-drawer" aria-label="最近会话" :aria-busy="loading">
    <header class="drawer-header">
      <div>
        <h2>最近会话</h2>
        <p>按更新时间排序</p>
      </div>
      <button type="button" class="drawer-refresh" :disabled="loading" @click="$emit('refresh')">{{ loading ? '刷新中…' : '刷新' }}</button>
    </header>

    <button type="button" class="new-session" :disabled="loading" @click="$emit('start')">新建会话</button>

    <p v-if="sessions.length === 0" class="session-empty">还没有会话</p>
    <ul v-else class="session-items">
      <li v-for="item in sessions" :key="item.session_id || item.id" class="session-row">
        <button
          type="button"
          class="session-select"
          :class="{ active: (item.session_id || item.id) === activeId }"
          :aria-current="(item.session_id || item.id) === activeId ? 'true' : undefined"
          :aria-label="sessionTitle(item)"
          :disabled="loading"
          @click="$emit('select', item.session_id || item.id)"
        >
          <span class="session-title">{{ sessionTitle(item) }}</span>
          <span class="session-meta">
            <span class="session-status">{{ sessionStatus(item) }}</span>
            <span aria-hidden="true"> · </span>
            <span>{{ formatUpdatedAt(item.updated_at) }} · {{ item.message_count ?? 0 }} 条消息</span>
          </span>
        </button>
        <button
          type="button"
          class="session-delete"
          :aria-label="`删除会话 ${sessionTitle(item)}`"
          :disabled="loading"
          @click="$emit('remove', item.session_id || item.id)"
        >
          删除
        </button>
      </li>
    </ul>
  </aside>
</template>

<script setup>
const formatUpdatedAt = (value) => {
  if (!value) return '更新时间未知';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString('zh-CN', { hour12: false });
};

const sessionTitle = (item) => {
  const title = item?.title;
  if (typeof title === 'string' && title.trim()) return title.trim();
  return item?.session_id || item?.id || '未命名会话';
};

const sessionStatus = (item) => {
  const labels = {
    running: '进行中',
    completed: '已完成',
    failed: '执行失败',
    stopped: '已停止',
    throttled: '已限流',
    rejected: '已拒绝',
    unavailable: '执行投影不可用'
  };
  return labels[item?.latest_execution_state] || '未记录关闭结果';
};

defineProps({
  loading: {
    type: Boolean,
    default: false
  },
  sessions: {
    type: Array,
    default: () => []
  },
  activeId: {
    type: String,
    default: ''
  }
});

defineEmits(['select', 'remove', 'refresh', 'start']);
</script>

<style scoped>
.session-drawer {
  min-width: 0;
  padding: 20px 16px;
  border-right: 1px solid var(--color-rule);
  background: var(--color-paper-muted);
}

.drawer-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  margin-bottom: 16px;
}

.drawer-header h2,
.drawer-header p {
  margin: 4px 0 0;
}

.drawer-header h2 {
  color: var(--color-ink);
  font-family: var(--font-display);
  font-size: 18px;
  font-weight: 600;
  line-height: 1.35;
}

.drawer-header p,
.session-empty {
  font-size: 13px;
  line-height: 1.5;
  color: var(--color-ink-soft);
}

.drawer-refresh,
.session-delete,
.session-select,
.new-session {
  transition: color 180ms ease-out, background-color 180ms ease-out, border-color 180ms ease-out, transform 180ms ease-out;
}

.drawer-refresh,
.session-delete {
  border: 0;
  background: transparent;
  color: var(--color-ink-soft);
  font: inherit;
  font-size: 13px;
  cursor: pointer;
}

.drawer-refresh:hover,
.session-delete:hover {
  color: var(--color-copper-strong);
}

.drawer-refresh:active,
.session-delete:active,
.session-select:active,
.new-session:active {
  transform: translateY(1px);
}

.drawer-refresh:disabled,
.session-delete:disabled,
.session-select:disabled,
.new-session:disabled {
  cursor: not-allowed;
  opacity: 0.58;
}

.new-session {
  width: 100%;
  min-height: 40px;
  margin-bottom: 16px;
  border: 1px solid var(--color-copper);
  border-radius: var(--radius-control);
  background: var(--color-copper);
  color: var(--color-paper-raised);
  font: inherit;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
}

.new-session:hover:not(:disabled) {
  border-color: var(--color-copper-strong);
  background: var(--color-copper-strong);
}

.session-items {
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-height: calc(100vh - 196px);
  margin: 0;
  padding: 0;
  list-style: none;
  overflow-y: auto;
}

.session-row {
  display: flex;
  align-items: center;
  gap: 4px;
  border-bottom: 1px solid var(--color-rule);
}

.session-select {
  min-width: 0;
  flex: 1;
  padding: 12px 0;
  border: 0;
  background: transparent;
  color: var(--color-ink);
  text-align: left;
  cursor: pointer;
}

.session-select:hover:not(:disabled) {
  color: var(--color-copper-strong);
}

.session-select.active {
  color: var(--color-copper-strong);
}

.session-title {
  display: block;
  overflow: hidden;
  font-weight: 600;
  font-size: 14px;
  line-height: 1.45;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.session-meta {
  display: block;
  margin-top: 2px;
  color: var(--color-ink-soft);
  font-size: 11px;
  line-height: 1.5;
}
</style>
