<template>
  <section ref="chatSectionRef" class="chat-page">
    <div class="top-bar">
      <div>
        <h1>对话工作区</h1>
        <p class="subtitle" v-if="streamSubtitle">{{ streamSubtitle }}</p>
      </div>
      <div class="top-actions">
        <el-button class="btn-ghost" @click="startNewSession">新建会话</el-button>
        <el-button class="btn-ghost" @click="toggleSessions">会话</el-button>
      </div>
    </div>

    <SessionDrawer
      :visible="sessionVisible"
      :sessions="chatStore.sessions"
      :active-id="chatStore.activeSessionId"
      @select="openSession"
      @remove="removeSession"
      @refresh="loadSessions"
      @close="sessionVisible = false"
    />

    <ChatMessageList :messages="chatStore.messages" />

    <div class="composer card">
      <el-input
        v-model="input"
        type="textarea"
        :rows="3"
        placeholder="请输入需要检索的问题"
      />
      <div class="composer-actions">
        <el-button class="btn-ghost" :disabled="!chatStore.loading" @click="chatStore.stopStreaming">停止</el-button>
        <el-button type="primary" :loading="chatStore.loading" :disabled="!authStore.isLoggedIn" @click="onSend">
          发送
        </el-button>
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
  try {
    await chatStore.loadSessions();
  } catch (error) {
    ElMessage.error(error.message || '加载会话失败');
  }
};

const openSession = async (sessionId) => {
  try {
    await chatStore.loadSessionMessages(sessionId);
    sessionVisible.value = false;
  } catch (error) {
    ElMessage.error(error.message || '加载会话消息失败');
  }
};

const removeSession = async (sessionId) => {
  try {
    await ElMessageBox.confirm('确认删除该会话？', '提示', { type: 'warning' });
    await chatStore.deleteSession(sessionId);
    ElMessage.success('会话已删除');
  } catch (error) {
    if (error !== 'cancel') ElMessage.error(error.message || '删除会话失败');
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
};

const onSend = async () => {
  if (!authStore.isLoggedIn) {
    ElMessage.warning('请先登录');
    return;
  }
  const question = input.value;
  input.value = '';
  await chatStore.sendMessage(question, { token: authStore.token });
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
.chat-page {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.top-actions {
  display: flex;
  align-items: center;
  gap: 12px;
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
</style>
