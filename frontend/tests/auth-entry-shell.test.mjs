// Authentication Entry browser acceptance over a disposable real API.
// All roles are derived from the server identity endpoint; network traffic is
// real HTTP through the Vite proxy (see acceptance-env.mjs).
import assert from 'node:assert/strict';
import test from 'node:test';
import {
  createTeamInvitation,
  loginAdmin,
  registerKnowledgeUserViaApi,
  startWorkbench
} from './acceptance-env.mjs';

test('Authentication Entry registers a Knowledge User before entering the workbench shell', { timeout: 30000 }, async (t) => {
  const { page, baseUrl, api } = await startWorkbench(t, {});

  await page.goto(`${baseUrl}auth`);
  await page.getByRole('heading', { name: '身份验证' }).waitFor();
  assert.equal(await page.getByRole('navigation').count(), 0);

  await page.getByRole('tab', { name: '注册' }).click();
  assert.equal(await page.getByRole('radio', { name: '系统管理员' }).count(), 0);
  assert.equal(await page.getByLabel('团队邀请码').isVisible(), true);

  await page.getByLabel('用户名').fill('lin');
  await page.getByLabel('密码').fill('safe-password');
  await page.getByLabel('团队邀请码').fill(await createTeamInvitation(api));
  await page.getByRole('button', { name: '完成注册' }).click();

  await page.waitForURL(/\/chat$/);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();
  assert.equal(await page.getByRole('navigation').count(), 1);
});

test('a stored token refreshes the server role before rejecting an administrator route and logout clears it', { timeout: 30000 }, async (t) => {
  const { page, baseUrl, api } = await startWorkbench(t, {});
  const token = await registerKnowledgeUserViaApi(api, 'lin');

  await page.addInitScript(({ token: storedToken }) => {
    localStorage.setItem('access_token', storedToken);
    localStorage.setItem('username', 'forged-admin-name');
    localStorage.setItem('role', 'admin');
  }, { token });

  await page.goto(`${baseUrl}documents`);
  await page.waitForURL(/\/chat\?notice=admin-required$/);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();
  assert.equal(await page.getByText('知识用户').isVisible(), true);
  assert.equal(await page.getByText('系统管理员').count(), 0);
  assert.equal(await page.getByText('当前账户无权访问该工作区，已返回对话工作区。').isVisible(), true);

  await page.getByRole('button', { name: '退出登录' }).click();
  await page.waitForURL(/\/auth$/);
  const staleTokenResponse = await fetch(`${api.baseUrl}/auth/me`, {
    headers: { Authorization: `Bearer ${token}` }
  });
  assert.equal(staleTokenResponse.status, 401);
  assert.deepEqual(
    await page.evaluate(() => ({
      token: localStorage.getItem('access_token'),
      username: localStorage.getItem('username'),
      role: localStorage.getItem('role')
    })),
    { token: null, username: null, role: null }
  );
  assert.equal(await page.getByRole('navigation').count(), 0);
  assert.equal(await page.getByRole('heading', { name: '对话工作区' }).count(), 0);
});

test('a failed logout request never leaves a protected browser session visible', { timeout: 30000 }, async (t) => {
  const { page, baseUrl, api } = await startWorkbench(t, {});
  const token = await registerKnowledgeUserViaApi(api, 'logout-network-failure');

  await page.addInitScript(({ storedToken }) => {
    localStorage.setItem('access_token', storedToken);
  }, { storedToken: token });
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await page.route('**/auth/logout', (route) => route.abort('failed'));
  await page.getByRole('button', { name: '退出登录' }).click();

  await page.waitForURL(/\/auth$/);
  assert.equal(await page.getByRole('navigation').count(), 0);
  assert.deepEqual(
    await page.evaluate(() => ({
      token: localStorage.getItem('access_token'),
      username: localStorage.getItem('username'),
      role: localStorage.getItem('role')
    })),
    { token: null, username: null, role: null }
  );
  const serverSession = await fetch(`${api.baseUrl}/auth/me`, {
    headers: { Authorization: `Bearer ${token}` }
  });
  assert.equal(serverSession.status, 200);
});

test('a deactivated stale browser session is expelled when a protected request receives 401', { timeout: 30000 }, async (t) => {
  const { page, baseUrl, api } = await startWorkbench(t, {});
  const token = await registerKnowledgeUserViaApi(api, 'stale-browser-member');

  await page.addInitScript(({ storedToken }) => {
    localStorage.setItem('access_token', storedToken);
  }, { storedToken: token });
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  const administratorLogin = await fetch(`${api.baseUrl}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: 'operator', password: 'safe-password' })
  });
  assert.equal(administratorLogin.status, 200);
  const administratorToken = (await administratorLogin.json()).data.access_token;
  const deactivation = await fetch(`${api.baseUrl}/members/stale-browser-member/deactivate`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${administratorToken}` }
  });
  assert.equal(deactivation.status, 200);

  await page.goto(`${baseUrl}knowledge`);
  await page.waitForURL(/\/auth$/);
  assert.equal(await page.getByRole('navigation').count(), 0);
  assert.deepEqual(
    await page.evaluate(() => ({
      token: localStorage.getItem('access_token'),
      username: localStorage.getItem('username'),
      role: localStorage.getItem('role')
    })),
    { token: null, username: null, role: null }
  );
  const staleBearer = await fetch(`${api.baseUrl}/auth/me`, {
    headers: { Authorization: `Bearer ${token}` }
  });
  assert.equal(staleBearer.status, 401);
});

test('sign-in has no role selector and keeps form input after a specific authentication error', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, {});

  await page.goto(`${baseUrl}auth`);
  await page.getByRole('heading', { name: '身份验证' }).waitFor();
  assert.equal(await page.getByRole('radio', { name: '系统管理员' }).count(), 0);

  await page.getByLabel('用户名').fill('unknown-user');
  await page.getByLabel('密码').fill('incorrect-password');
  await page.getByRole('button', { name: '登录' }).click();

  await page.getByRole('alert').waitFor();
  assert.equal(await page.getByRole('alert').innerText(), '用户名或密码不正确，请核对后重试。');
  assert.equal(await page.getByLabel('用户名').inputValue(), 'unknown-user');
  assert.equal(await page.getByLabel('密码').inputValue(), 'incorrect-password');
  assert.match(page.url(), /\/auth$/);
});

test('sign-in derives the System Administrator role from the identity endpoint without unfinished navigation', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, {});
  await loginAdmin(page, baseUrl);

  await page.waitForURL(/\/chat$/);
  assert.equal(await page.getByText('系统管理员').isVisible(), true);
  assert.equal(await page.getByRole('link', { name: '对话工作区' }).count(), 1);
  assert.equal(await page.getByRole('link', { name: '文档库' }).count(), 1);
  assert.equal(await page.getByRole('link', { name: '构建任务' }).count(), 1);
  assert.equal(await page.getByRole('link', { name: '系统设置' }).count(), 1);
});

test('a System Administrator cannot discover or directly load System Settings when the server disables its lifecycle', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, {
    env: { SYSTEM_SETTINGS_DRAFT_ENABLED: 'false', SYSTEM_SETTINGS_APPLICATION_ENABLED: 'false' }
  });
  await loginAdmin(page, baseUrl);

  assert.equal(await page.getByRole('link', { name: '系统设置' }).count(), 0);

  await page.goto(`${baseUrl}config`);
  await page.waitForURL(/\/chat\?notice=settings-unavailable$/);
  assert.equal(await page.getByRole('heading', { name: '系统设置' }).count(), 0);
  assert.equal(await page.getByRole('link', { name: '系统设置' }).count(), 0);
});

test('an invalid stored session returns to Authentication Entry without protected state', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, {});

  await page.addInitScript(() => {
    localStorage.setItem('access_token', 'expired-token');
    localStorage.setItem('username', 'stale-user');
    localStorage.setItem('role', 'admin');
  });
  await page.goto(`${baseUrl}chat`);
  await page.waitForURL(/\/auth$/);
  await page.getByRole('heading', { name: '身份验证' }).waitFor();
  assert.equal(await page.getByRole('navigation').count(), 0);
  assert.deepEqual(
    await page.evaluate(() => ({
      token: localStorage.getItem('access_token'),
      username: localStorage.getItem('username'),
      role: localStorage.getItem('role')
    })),
    { token: null, username: null, role: null }
  );
});

test('built assets route an unauthenticated visitor to Authentication Entry', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, { built: true });

  await page.goto(`${baseUrl}chat`);
  await page.waitForURL(/\/auth$/);
  await page.getByRole('heading', { name: '身份验证' }).waitFor();
  assert.equal(await page.getByRole('navigation').count(), 0);
});
