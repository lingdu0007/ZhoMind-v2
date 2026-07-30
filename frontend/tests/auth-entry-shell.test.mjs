import assert from 'node:assert/strict';
import test from 'node:test';
import { chromium } from 'playwright';
import { createServer, preview } from 'vite';

const jsonResponse = (data, status = 200) => ({
  status,
  contentType: 'application/json',
  body: JSON.stringify({ data })
});

const startBrowserApp = async (t, apiHandler, { built = false } = {}) => {
  const testPort = 41000 + Math.floor(Math.random() * 1000);
  const server = built
    ? await preview({ preview: { host: '127.0.0.1', port: testPort, strictPort: true } })
    : await createServer({ server: { host: '127.0.0.1', port: testPort, strictPort: true } });
  if (!built) await server.listen();

  const browser = await chromium.launch({ headless: true });
  t.after(async () => {
    await browser.close();
    await server.close();
  });

  const page = await browser.newPage();
  page.setDefaultTimeout(5000);
  await page.route((url) => url.pathname.startsWith('/api/'), apiHandler);

  return {
    page,
    baseUrl: server.resolvedUrls.local[0]
  };
};

test('Authentication Entry registers a Knowledge User before entering the workbench shell', { timeout: 30000 }, async (t) => {
  const registerPayloads = [];
  const { page, baseUrl } = await startBrowserApp(t, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;

    if (path === '/api/auth/register') {
      const payload = request.postDataJSON();
      registerPayloads.push(payload);
      await route.fulfill(
        jsonResponse({
          access_token: 'registration-token',
          username: payload.username,
          role: 'user'
        })
      );
      return;
    }

    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'lin', role: 'user' }));
      return;
    }

    if (path === '/api/sessions') {
      await route.fulfill(jsonResponse({ sessions: [] }));
      return;
    }

    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await page.goto(`${baseUrl}auth`);
  await page.getByRole('heading', { name: '身份验证' }).waitFor();
  assert.equal(await page.getByRole('navigation').count(), 0);

  await page.getByRole('tab', { name: '注册' }).click();
  await page.getByRole('radio', { name: '系统管理员' }).check();
  assert.equal(await page.getByLabel('管理员邀请码').isVisible(), true);

  await page.getByRole('radio', { name: '知识用户' }).check();
  assert.equal(await page.getByLabel('管理员邀请码').isVisible(), false);

  await page.getByLabel('用户名').fill('lin');
  await page.getByLabel('密码').fill('safe-password');
  await page.getByRole('button', { name: '完成注册' }).click();

  await page.waitForURL(/\/chat$/);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();
  assert.equal(await page.getByRole('navigation').count(), 1);
  assert.deepEqual(registerPayloads, [{ username: 'lin', password: 'safe-password', role: 'user' }]);
});

test('a stored token refreshes the server role before rejecting an administrator route and logout clears it', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startBrowserApp(t, async (route) => {
    const path = new URL(route.request().url()).pathname;

    if (path === '/api/auth/me') {
      await new Promise((resolve) => setTimeout(resolve, 250));
      await route.fulfill(jsonResponse({ username: 'lin', role: 'user' }));
      return;
    }

    if (path === '/api/sessions') {
      await route.fulfill(jsonResponse({ sessions: [] }));
      return;
    }

    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await page.addInitScript(() => {
    localStorage.setItem('access_token', 'stored-user-token');
    localStorage.setItem('username', 'forged-admin-name');
    localStorage.setItem('role', 'admin');
  });

  await page.goto(`${baseUrl}documents`);
  assert.equal(await page.getByRole('navigation').count(), 0);
  await page.getByText('正在确认身份…').waitFor();
  await page.waitForURL(/\/chat\?notice=admin-required$/);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();
  assert.equal(await page.getByText('知识用户').isVisible(), true);
  assert.equal(await page.getByText('系统管理员').count(), 0);
  assert.equal(await page.getByText('当前账户无权访问该工作区，已返回对话工作区。').isVisible(), true);

  await page.getByRole('button', { name: '退出登录' }).click();
  await page.waitForURL(/\/auth$/);
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

test('sign-in has no role selector and keeps form input after a specific authentication error', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startBrowserApp(t, async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/auth/login') {
      await route.fulfill({
        status: 401,
        contentType: 'application/json',
        body: JSON.stringify({ code: 'AUTH_INVALID_CREDENTIALS', message: 'invalid username or password' })
      });
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

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
  const { page, baseUrl } = await startBrowserApp(t, async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/auth/login') {
      await route.fulfill(jsonResponse({ access_token: 'admin-token', username: 'operator', role: 'user' }));
      return;
    }
    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'operator', role: 'admin' }));
      return;
    }
    if (path === '/api/sessions') {
      await route.fulfill(jsonResponse({ sessions: [] }));
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await page.goto(`${baseUrl}auth`);
  await page.getByLabel('用户名').fill('operator');
  await page.getByLabel('密码').fill('safe-password');
  await page.getByRole('button', { name: '登录' }).click();

  await page.waitForURL(/\/chat$/);
  assert.equal(await page.getByText('系统管理员').isVisible(), true);
  assert.equal(await page.getByRole('link', { name: '对话工作区' }).count(), 1);
  assert.equal(await page.getByRole('link', { name: '文档库' }).count(), 0);
  assert.equal(await page.getByRole('link', { name: '构建任务' }).count(), 1);
  assert.equal(await page.getByRole('link', { name: '系统设置' }).count(), 0);
});

test('an invalid stored session returns to Authentication Entry without protected state', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startBrowserApp(t, async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/auth/me') {
      await route.fulfill({
        status: 401,
        contentType: 'application/json',
        body: JSON.stringify({ code: 'AUTH_INVALID_TOKEN', message: 'token expired' })
      });
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

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

test('a later unauthorized workspace request clears the active shell', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startBrowserApp(t, async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'lin', role: 'user' }));
      return;
    }
    if (path === '/api/sessions') {
      await route.fulfill({
        status: 401,
        contentType: 'application/json',
        body: JSON.stringify({ code: 'AUTH_INVALID_TOKEN', message: 'token expired' })
      });
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await page.addInitScript(() => localStorage.setItem('access_token', 'soon-to-expire-token'));
  await page.goto(`${baseUrl}chat`);
  await page.waitForURL(/\/auth$/);
  await page.getByRole('heading', { name: '身份验证' }).waitFor();
  assert.equal(await page.getByRole('navigation').count(), 0);
});

test('built assets route an unauthenticated visitor to Authentication Entry', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startBrowserApp(
    t,
    async (route) => route.fulfill(jsonResponse({ message: 'Unexpected request' }, 404)),
    { built: true }
  );

  await page.goto(`${baseUrl}chat`);
  await page.waitForURL(/\/auth$/);
  await page.getByRole('heading', { name: '身份验证' }).waitFor();
  assert.equal(await page.getByRole('navigation').count(), 0);
});
