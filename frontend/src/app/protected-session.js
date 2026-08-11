import { useAuthStore } from '../store/auth';
import { useChatStore } from '../store/chat';

export const clearProtectedSession = () => {
  useChatStore().clearWorkspaceState();
  useAuthStore().clearAuth();
};
