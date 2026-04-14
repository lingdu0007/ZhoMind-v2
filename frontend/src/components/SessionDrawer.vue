<template>
  <transition name="fade-slide">
    <div v-if="visible" class="session-drawer card">
      <header class="drawer-header">
        <div>
          <strong>会话列表</strong>
          <p>最近 20 条会话，点击切换</p>
        </div>
        <div class="drawer-actions">
          <el-button text @click="$emit('refresh')">刷新</el-button>
          <el-button text @click="$emit('close')">收起</el-button>
        </div>
      </header>
      <div class="session-items">
        <div
          v-for="item in sessions"
          :key="item.session_id || item.id"
          class="session-row"
          :class="{ active: (item.session_id || item.id) === activeId }"
          @click="$emit('select', item.session_id || item.id)"
        >
          <div class="session-meta">
            <p>{{ item.session_id || item.id }}</p>
            <small>{{ item.updated_at || '--' }} · {{ item.message_count ?? 0 }} 条</small>
          </div>
          <el-button link type="danger" @click.stop="$emit('remove', item.session_id || item.id)">删除</el-button>
        </div>
      </div>
    </div>
  </transition>
</template>

<script setup>
defineProps({
  visible: {
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

defineEmits(['select', 'remove', 'refresh', 'close']);
</script>

<style scoped>
.session-drawer {
  padding: 20px;
  margin-bottom: 16px;
}

.drawer-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  margin-bottom: 12px;
}

.drawer-header p {
  margin: 4px 0 0;
  font-size: 13px;
  color: var(--text-muted);
}

.session-items {
  display: flex;
  flex-direction: column;
  gap: 12px;
  max-height: 280px;
  overflow-y: auto;
}

.session-row {
  border: 1px solid var(--line-soft);
  border-radius: 12px;
  padding: 12px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  cursor: pointer;
  transition: border-color 0.2s ease, background 0.2s ease;
}

.session-row:hover {
  border-color: var(--line-strong);
}

.session-row.active {
  border-color: var(--cta);
  background: #eef2ff;
}

.session-meta {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.session-meta p {
  margin: 0;
  font-weight: 600;
  font-size: 14px;
}

.session-meta small {
  color: var(--text-muted);
}

.fade-slide-enter-active,
.fade-slide-leave-active {
  transition: all 0.2s ease;
}

.fade-slide-enter-from,
.fade-slide-leave-to {
  opacity: 0;
  transform: translateY(-8px);
}
</style>
