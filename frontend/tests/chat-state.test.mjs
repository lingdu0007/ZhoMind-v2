import test from 'node:test';
import assert from 'node:assert/strict';
import { extractRejectReason, formatStreamError, getDoneStatus, getProviderStatus } from '../src/store/chat-state.js';

test('extractRejectReason detects retrieve reject gates', () => {
  const reason = extractRejectReason({
    step: 'retrieve',
    detail: { gate_passed: false, gate_reason: 'low_reject' }
  });
  assert.equal(reason, 'low_reject');
});

test('extractRejectReason ignores non-reject steps', () => {
  assert.equal(
    extractRejectReason({
      step: 'retrieve',
      detail: { gate_passed: true, gate_reason: 'high_accept' }
    }),
    ''
  );
  assert.equal(extractRejectReason({ step: 'generate', detail: {} }), '');
});

test('formatStreamError formats known auth errors first', () => {
  assert.equal(formatStreamError({ code: 'AUTH_FORBIDDEN', message: 'Forbidden' }), '无权限执行当前问答');
  assert.equal(formatStreamError({ status: 401, message: 'Unauthorized' }), '登录状态已失效，请重新登录');
  assert.equal(formatStreamError({ message: 'boom' }), 'boom');
});

test('getDoneStatus returns reject state label when rejected', () => {
  assert.equal(getDoneStatus({ rejected: true }), '已拒答（证据不足）');
  assert.equal(getDoneStatus({ rejected: false }), '');
});

test('getProviderStatus returns direct provider label', () => {
  const status = getProviderStatus({ runtime: { final_provider: 'ark', fallback_hops: 0 } });
  assert.equal(status, '模型提供方：ark');
});

test('getProviderStatus returns fallback label with hops', () => {
  const status = getProviderStatus({ runtime: { final_provider: 'openai', fallback_hops: 1 } });
  assert.equal(status, '已切换到 openai（1 次回退）');
});

test('getProviderStatus returns empty when provider missing', () => {
  assert.equal(getProviderStatus({ runtime: {} }), '');
  assert.equal(getProviderStatus(null), '');
});

