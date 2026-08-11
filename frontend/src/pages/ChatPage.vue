<template>
  <section ref="chatSectionRef" class="conversation-workspace" :class="{ 'conversation-workspace--excerpt-open': selectedSource }">
    <SessionDrawer
      class="conversation-workspace__session-rail"
      :loading="sessionsLoading"
      :sessions="chatStore.sessions"
      :active-id="chatStore.activeSessionId"
      @select="openSession"
      @remove="removeSession"
      @refresh="loadSessions"
      @start="startNewSession"
    />

    <div class="conversation-workspace__content">
      <div class="chat-page">
        <div class="top-bar">
        <div>
          <h1>对话工作区</h1>
          <p class="subtitle" v-if="streamSubtitle">{{ streamSubtitle }}</p>
        </div>
        <div class="top-actions">
          <el-button class="btn-ghost" @click="startNewSession">新建会话</el-button>
          <el-button class="btn-ghost session-toggle" @click="toggleSessions">会话</el-button>
        </div>
        </div>

        <el-drawer v-model="sessionVisible" title="最近会话" direction="ltr" size="min(86vw, 320px)">
        <SessionDrawer
          :loading="sessionsLoading"
          :sessions="chatStore.sessions"
          :active-id="chatStore.activeSessionId"
          @select="openSession"
          @remove="removeSession"
          @refresh="loadSessions"
          @start="startNewSession"
        />
        </el-drawer>

        <ChatMessageList
          :messages="chatStore.messages"
          :retry-disabled="chatStore.loading"
          :show-diagnostics="authStore.isAdmin"
          @retry="retryAssistantMessage"
          @open-source="openSourceExcerpt"
        />

        <div class="composer card">
        <el-input
          v-model="input"
          type="textarea"
          :rows="3"
          placeholder="请输入需要检索的问题"
          aria-describedby="composer-feedback"
        />
        <p v-if="composerError" id="composer-feedback" class="composer-feedback" role="alert">{{ composerError }}</p>
        <div class="composer-actions">
          <el-button class="btn-ghost" :disabled="!chatStore.loading" @click="chatStore.stopStreaming">停止</el-button>
          <el-button type="primary" :loading="chatStore.loading" :disabled="!authStore.isLoggedIn || chatStore.loading" @click="onSend">
            发送
          </el-button>
        </div>
        </div>
      </div>

      <aside v-if="selectedSource" class="evidence-excerpt" role="complementary" aria-label="来源摘录" @keydown.esc.prevent="closeSourceExcerpt">
        <header class="evidence-excerpt__header">
          <div>
            <p>来源摘录</p>
            <h2>{{ sourceLabel }}</h2>
          </div>
          <button ref="excerptCloseRef" type="button" class="evidence-excerpt__close" aria-label="关闭来源摘录" @click="closeSourceExcerpt">关闭</button>
        </header>
        <p class="evidence-excerpt__id">{{ selectedSource.source_id }}</p>
        <div class="evidence-excerpt__content">{{ selectedSource.withdrawal_notice || selectedSource.excerpt || '未返回可展示的来源摘录。' }}</div>
      </aside>
    </div>
  </section>
</template>

<script setup>
import { computed, nextTick, onMounted, ref, watch } from 'vue';
import { ElMessage, ElMessageBox } from 'element-plus';
import ChatMessageList from '../components/ChatMessageList.vue';
import SessionDrawer from '../components/SessionDrawer.vue';
import { useChatStore } from '../store/chat';
import { useAuthStore } from '../store/auth';
import { getEvidenceSourceLabel } from '../app/evidence-summary';

const chatStore = useChatStore();
const authStore = useAuthStore();
const input = ref('');
const chatSectionRef = ref(null);
const sessionVisible = ref(false);
const composerError = ref('');
const sessionsLoading = ref(false);
const selectedSource = ref(null);
const sourceTrigger = ref(null);
const excerptCloseRef = ref(null);

const sourceLabel = computed(() => getEvidenceSourceLabel(selectedSource.value));

const streamSubtitle = computed(() => {
  if (chatStore.loading) return '流式生成中…';
  const assistantMessages = chatStore.messages.filter((item) => item.role === 'assistant');
  const lastAssistant = assistantMessages[assistantMessages.length - 1];
  if (!lastAssistant) return '';
  return lastAssistant.status || '';
});

const scrollToBottom = async () => {
  await nextTick();
  if (!chatSectionRef.value) return;
  chatSectionRef.value.scrollIntoView({ behavior: 'smooth', block: 'end' });
};

const loadSessions = async () => {
  if (!authStore.isLoggedIn) return;
  sessionsLoading.value = true;
  try {
    await chatStore.loadSessions();
  } catch (error) {
    ElMessage.error(error.message || '加载会话失败');
  } finally {
    sessionsLoading.value = false;
  }
};

const openSession = async (sessionId) => {
  sessionsLoading.value = true;
  try {
    await chatStore.loadSessionMessages(sessionId);
    sessionVisible.value = false;
  } catch (error) {
    ElMessage.error(error.message || '加载会话消息失败');
  } finally {
    sessionsLoading.value = false;
  }
};

const removeSession = async (sessionId) => {
  try {
    await ElMessageBox.confirm('确认删除该会话？', '删除会话', {
      type: 'warning',
      confirmButtonText: '删除',
      cancelButtonText: '取消'
    });
    sessionsLoading.value = true;
    await chatStore.deleteSession(sessionId);
    ElMessage.success('会话已删除');
  } catch (error) {
    if (error !== 'cancel') ElMessage.error(error.message || '删除会话失败');
  } finally {
    sessionsLoading.value = false;
  }
};

const toggleSessions = async () => {
  if (!authStore.isLoggedIn) {
    ElMessage.warning('请先登录');
    return;
  }
  sessionVisible.value = !sessionVisible.value;
  if (sessionVisible.value) {
    await loadSessions();
  }
};

const startNewSession = () => {
  chatStore.activeSessionId = '';
  chatStore.messages = [];
  sessionVisible.value = false;
};

const openSourceExcerpt = async ({ source, trigger }) => {
  selectedSource.value = source;
  sourceTrigger.value = trigger;
  await nextTick();
  excerptCloseRef.value?.focus();
};

const closeSourceExcerpt = async () => {
  selectedSource.value = null;
  await nextTick();
  sourceTrigger.value?.focus();
};

const onSend = async () => {
  if (!authStore.isLoggedIn) {
    ElMessage.warning('请先登录');
    return;
  }
  const question = input.value.trim();
  if (!question) {
    composerError.value = '请输入问题后再发送。';
    return;
  }
  composerError.value = '';
  input.value = '';
  await chatStore.sendMessage(question, { token: authStore.token });
};

const retryAssistantMessage = async (assistantIndex) => {
  if (chatStore.loading) return;
  const previousUserMessage = chatStore.messages
    .slice(0, assistantIndex)
    .reverse()
    .find((message) => message.role === 'user');
  if (!previousUserMessage?.content) return;
  composerError.value = '';
  await chatStore.sendMessage(previousUserMessage.content, { token: authStore.token });
};

watch(
  () => chatStore.streamTick,
  () => {
    scrollToBottom();
  }
);

onMounted(loadSessions);
</script>

<style scoped>
.conversation-workspace {
  display: grid;
  grid-template-columns: 264px minmax(0, 820px);
  justify-content: center;
  width: min(100%, 1084px);
  min-height: calc(100vh - 92px);
  border: 1px solid var(--color-rule);
  background: var(--color-paper-raised);
}

.conversation-workspace__session-rail {
  min-height: 100%;
}

.conversation-workspace--excerpt-open {
  grid-template-columns: 264px minmax(0, 932px);
  width: min(100%, 1196px);
}

.conversation-workspace__content {
  min-width: 0;
  display: grid;
  grid-template-columns: minmax(0, 1fr);
}

.conversation-workspace--excerpt-open .conversation-workspace__content {
  grid-template-columns: minmax(0, 1fr) minmax(280px, 320px);
}

.chat-page {
  min-width: 0;
  padding: 32px clamp(24px, 4vw, 48px) 40px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.top-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.session-toggle {
  display: none;
}

.composer {
  margin-top: 12px;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  position: sticky;
  bottom: 24px;
  backdrop-filter: blur(10px);
}

.composer-actions {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

.composer-feedback {
  margin: 0;
  color: var(--color-danger);
  font-size: 13px;
  line-height: 1.5;
}

.evidence-excerpt {
  min-width: 0;
  padding: 24px;
  border-left: 1px solid var(--color-rule);
  background: var(--color-paper-muted);
  animation: evidence-excerpt-enter 200ms ease-out both;
}

.evidence-excerpt__header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--color-rule);
}

.evidence-excerpt__header p,
.evidence-excerpt__header h2,
.evidence-excerpt__id,
.evidence-excerpt__content {
  margin: 0;
}

.evidence-excerpt__header p {
  color: var(--color-moss);
  font-size: 12px;
}

.evidence-excerpt__header h2 {
  margin-top: 4px;
  overflow-wrap: anywhere;
  font-family: var(--font-display);
  font-size: 18px;
  font-weight: 600;
  line-height: 1.4;
}

.evidence-excerpt__close {
  flex: 0 0 auto;
  min-height: 32px;
  padding: 4px 8px;
  border: 1px solid var(--line-strong);
  border-radius: var(--radius-control);
  background: transparent;
  color: var(--color-ink);
  cursor: pointer;
  font: inherit;
  font-size: 12px;
  transition: transform 180ms ease-out;
}

.evidence-excerpt__close:hover {
  border-color: var(--color-copper);
  background: var(--color-paper-raised);
  color: var(--color-copper-strong);
}

.evidence-excerpt__close:active {
  transform: translateY(1px);
}

.evidence-excerpt__id {
  margin-top: 16px;
  color: var(--color-ink-soft);
  font-family: var(--font-mono);
  font-size: 11px;
  line-height: 1.5;
  overflow-wrap: anywhere;
}

.evidence-excerpt__content {
  margin-top: 20px;
  color: var(--color-ink);
  font-size: 14px;
  line-height: 1.8;
  white-space: pre-wrap;
}

@keyframes evidence-excerpt-enter {
  from {
    opacity: 0;
    transform: translateX(12px);
  }

  to {
    opacity: 1;
    transform: translateX(0);
  }
}

@media (prefers-reduced-motion: reduce) {
  .evidence-excerpt {
    animation: none;
  }
}

@media (max-width: 1180px) {
  .conversation-workspace {
    display: block;
    max-width: 820px;
    margin: 0 auto;
  }

  .conversation-workspace__session-rail {
    display: none;
  }

  .session-toggle {
    display: inline-flex;
  }

  .conversation-workspace--excerpt-open .conversation-workspace__content {
    grid-template-columns: minmax(0, 1fr);
  }
}

@media (max-width: 760px) {
  .evidence-excerpt {
    border-top: 1px solid var(--color-rule);
    border-left: 0;
  }
}

@media (max-width: 640px) {
  .conversation-workspace {
    min-height: auto;
    border-right: 0;
    border-left: 0;
  }

  .chat-page {
    padding: 20px 16px 28px;
  }

  .top-bar {
    align-items: flex-start;
  }

  .top-actions {
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 8px;
  }

  .composer {
    bottom: 12px;
    padding: 16px;
  }
}
</style>
