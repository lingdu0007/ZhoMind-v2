<template>
  <section ref="chatSectionRef" class="chat-page">
    <div class="top-bar">
      <div>
        <h1>Chat · Knowledge Pulse</h1>
        <p class="subtitle" v-if="streamSubtitle">{{ streamSubtitle }}</p>
      </div>
      <div class="top-actions">
        <el-button class="btn-ghost" @click="startNewSession">新建会话</el-button>
        <el-button class="btn-ghost" @click="toggleSessions">会话</el-button>
        <template v-if="authStore.isLoggedIn">
          <span class="user-pill">{{ authStore.username }} · {{ authStore.role }}</span>
          <el-button text @click="logout">退出</el-button>
        </template>
        <el-button v-else type="primary" @click="authDialogVisible = true">登录 / 注册</el-button>
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
        placeholder="请输入问题，系统将执行 Agentic RAG 工作流"
      />
      <div class="composer-actions">
        <el-button class="btn-ghost" :disabled="!chatStore.loading" @click="chatStore.stopStreaming">停止</el-button>
        <el-button type="primary" :loading="chatStore.loading" :disabled="!authStore.isLoggedIn" @click="onSend">
          发送
        </el-button>
      </div>
    </div>

    <el-dialog v-model="authDialogVisible" title="登录 / 注册" width="480px">
      <el-form label-width="100px">
        <el-form-item label="模式">
          <el-radio-group v-model="authMode">
            <el-radio-button label="login">登录</el-radio-button>
            <el-radio-button label="register">注册</el-radio-button>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="用户名">
          <el-input v-model="authForm.username" />
        </el-form-item>
        <el-form-item label="密码">
          <el-input v-model="authForm.password" type="password" show-password />
        </el-form-item>
        <el-form-item v-if="authMode === 'register'" label="角色">
          <el-select v-model="authForm.role" placeholder="可选">
            <el-option label="user" value="user" />
            <el-option label="admin" value="admin" />
          </el-select>
        </el-form-item>
        <el-form-item v-if="authMode === 'register' && authForm.role === 'admin'" label="管理员码">
          <el-input v-model="authForm.admin_code" placeholder="仅注册管理员账号时需要" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="authDialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="authStore.loading" @click="submitAuth">
          {{ authMode === 'login' ? '登录' : '注册' }}
        </el-button>
      </template>
    </el-dialog>
  </section>
</template>

<script setup>
import { computed, nextTick, onMounted, reactive, ref, watch } from 'vue';
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

const authDialogVisible = ref(false);
const authMode = ref('login');
const authForm = reactive({
  username: '',
  password: '',
  role: 'user',
  admin_code: ''
});

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

const submitAuth = async () => {
  try {
    if (authMode.value === 'login') {
      await authStore.login({ username: authForm.username, password: authForm.password });
      ElMessage.success('登录成功');
    } else {
      const payload = {
        username: authForm.username,
        password: authForm.password,
        role: authForm.role
      };
      if (authForm.role === 'admin' && authForm.admin_code) {
        payload.admin_code = authForm.admin_code;
      }
      await authStore.register(payload);
      ElMessage.success('注册成功');
    }
    authDialogVisible.value = false;
    await loadSessions();
  } catch (error) {
    ElMessage.error(error.message || '认证失败');
  }
};

const logout = () => {
  authStore.clearAuth();
  chatStore.messages = [];
  chatStore.sessions = [];
  chatStore.activeSessionId = '';
};

const onSend = async () => {
  if (!authStore.isLoggedIn) {
    ElMessage.warning('请先登录');
    return;
  }
  const question = input.value;
  input.value = '';
  await chatStore.sendMessage(question);
};

watch(
  () => chatStore.streamTick,
  () => {
    scrollToBottom();
  }
);

onMounted(async () => {
  try {
    await authStore.refreshMe();
    await loadSessions();
  } catch {
    authStore.clearAuth();
  }
});
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
