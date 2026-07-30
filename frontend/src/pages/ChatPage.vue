<template>
  <section ref="chatSectionRef" class="conversation-workspace">
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

      <ChatMessageList :messages="chatStore.messages" :retry-disabled="chatStore.loading" @retry="retryAssistantMessage" />

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
  </section>
</template>

<script setup>
import { computed, nextTick, onMounted, ref, watch } from 'vue';
import { ElMessage, ElMessageBox } from 'element-plus';
import ChatMessageList from '../components/ChatMessageList.vue';
import SessionDrawer from '../components/SessionDrawer.vue';
import { useChatStore } from '../store/chat';
import { useAuthStore } from '../store/auth';

const chatStore = useChatStore();
const authStore = useAuthStore();
const input = ref('');
const chatSectionRef = ref(null);
const sessionVisible = ref(false);
const composerError = ref('');
const sessionsLoading = ref(false);

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
  min-height: calc(100vh - 92px);
  border: 1px solid var(--color-rule);
  background: var(--color-paper-raised);
}

.conversation-workspace__session-rail {
  min-height: 100%;
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
