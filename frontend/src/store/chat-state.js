export const extractRejectReason = (step) => {
  if (!step || typeof step !== 'object') return '';
  if (step.step !== 'retrieve') return '';

  const detail = step.detail || {};
  const gateReason = detail.gate_reason || '';
  if (detail.gate_passed === false && String(gateReason).includes('reject')) {
    return String(gateReason);
  }
  return '';
};

export const formatStreamError = (error) => {
  if (!error) return '请求失败，请稍后重试';

  const code = error.code || '';
  if (code === 'AUTH_FORBIDDEN') return '无权限执行当前问答';
  if (code === 'AUTH_INVALID_TOKEN' || error.status === 401) return '登录状态已失效，请重新登录';

  return error.message || '请求失败，请稍后重试';
};

export const getDoneStatus = (assistantMsg) => {
  if (assistantMsg?.rejected) return '已拒答（证据不足）';
  return '';
};
