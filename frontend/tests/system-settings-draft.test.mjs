import assert from 'node:assert/strict';
import test from 'node:test';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const jsonResponse = (data, status = 200) => ({
  status,
  contentType: 'application/json',
  body: JSON.stringify({ data })
});

const settingsDraft = {
  draft: {
    model_provider: 'ark',
    llm_model: 'Qwen/Qwen3-32B',
    embedding_model: 'BAAI/bge-m3',
    retrieval_strategy: 'migration',
    retrieval_top_k: 8,
    score_threshold: 0.3,
    milvus_uri: 'http://milvus.internal:19530',
    index_name: 'zhomind_docs',
    runtime_timeout_ms: 8000,
    provider_api_key: { configured: true }
  },
  saved_version: 2,
  active_version: 1,
  last_modified: { actor: 'operator', at: '2026-07-31T10:15:00Z' },
  application_state: 'saved',
  application: { version: 2, actor: 'operator', at: '2026-07-31T10:15:00Z', message: 'settings version is saved' }
};

const startSettingsWorkspace = async (t, apiHandler, viewport = { width: 1440, height: 900 }) => {
  const testPort = 43000 + Math.floor(Math.random() * 1000);
  const server = await createServer({ server: { host: '127.0.0.1', port: testPort, strictPort: true } });
  await server.listen();

  const browser = await chromium.launch({ headless: true });
  t.after(async () => {
    await browser.close();
    await server.close();
  });

  const page = await browser.newPage({ viewport });
  page.setDefaultTimeout(5000);
  await page.route((url) => url.pathname.startsWith('/api/'), apiHandler);
  return { page, baseUrl: server.resolvedUrls.local[0] };
};

test('System Administrator saves and applies a dirty draft without optimistically showing an active version', { timeout: 30000 }, async (t) => {
  const saves = [];
  const applications = [];
  let applicationStarted = false;
  const { page, baseUrl } = await startSettingsWorkspace(t, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'operator', role: 'admin' }));
      return;
    }
    if (path === '/api/settings/draft' && request.method() === 'GET') {
      await route.fulfill(
        jsonResponse(
          applicationStarted
            ? {
                ...settingsDraft,
                draft: { ...settingsDraft.draft, llm_model: 'Qwen/Qwen3-14B' },
                saved_version: 3,
                active_version: 3,
                application_state: 'active',
                application: { version: 3, actor: 'operator', at: '2026-07-31T10:20:01Z', message: 'settings version is active' }
              }
            : settingsDraft
        )
      );
      return;
    }
    if (path === '/api/settings/draft' && request.method() === 'PUT') {
      saves.push(request.postDataJSON());
      await route.fulfill(
        jsonResponse({
          ...settingsDraft,
          draft: { ...settingsDraft.draft, llm_model: 'Qwen/Qwen3-14B' },
          saved_version: 3,
          active_version: 1,
          last_modified: { actor: 'operator', at: '2026-07-31T10:20:00Z' },
          application_state: 'saved',
          application: { version: 3, actor: 'operator', at: '2026-07-31T10:20:00Z', message: 'settings version is saved' }
        })
      );
      return;
    }
    if (path === '/api/settings/apply' && request.method() === 'POST') {
      applications.push(request.postDataJSON());
      applicationStarted = true;
      await route.fulfill(
        jsonResponse({
          ...settingsDraft,
          draft: { ...settingsDraft.draft, llm_model: 'Qwen/Qwen3-14B' },
          saved_version: 3,
          active_version: 1,
          application_state: 'applying',
          application: { version: 3, actor: 'operator', at: '2026-07-31T10:20:00Z', message: 'settings version is applying' }
        })
      );
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await page.addInitScript(() => {
    localStorage.setItem('access_token', 'admin-token');
    localStorage.setItem('rag_config', JSON.stringify({ llm_model: 'forged-browser-value' }));
  });
  await page.goto(`${baseUrl}config`);

  await page.getByRole('heading', { name: '系统设置' }).waitFor();
  assert.equal(await page.getByRole('link', { name: '系统设置' }).isVisible(), true);
  assert.equal(await page.getByRole('heading', { name: '模型与提供方' }).isVisible(), true);
  assert.equal(await page.getByRole('heading', { name: '检索策略' }).isVisible(), true);
  assert.equal(await page.getByRole('heading', { name: '存储与索引' }).isVisible(), true);
  assert.equal(await page.getByRole('heading', { name: '安全与运行时' }).isVisible(), true);
  assert.equal(await page.getByLabel('语言模型').inputValue(), 'Qwen/Qwen3-32B');
  assert.equal(await page.getByText('Provider API 密钥已配置，内容已隐藏。').isVisible(), true);
  assert.equal(await page.getByText('已保存版本 2').isVisible(), true);
  assert.equal(await page.getByText('生效版本 1').isVisible(), true);
  assert.equal(await page.getByText('最后修改：operator').isVisible(), true);
  assert.equal(await page.getByText('保存并应用前，运行系统不会变化。').isVisible(), true);
  const geometry = await page.getByLabel('草稿状态').evaluate((bar) => ({
    pageWidth: document.documentElement.scrollWidth,
    viewportWidth: window.innerWidth,
    rightEdge: bar.getBoundingClientRect().right
  }));
  assert.ok(geometry.pageWidth <= geometry.viewportWidth);
  assert.ok(geometry.rightEdge <= geometry.viewportWidth);
  const [stateBarBounds, timeoutFieldBounds] = await Promise.all([
    page.getByLabel('草稿状态').evaluate((element) => element.getBoundingClientRect().toJSON()),
    page.getByLabel('运行时超时（毫秒）').evaluate((element) => element.getBoundingClientRect().toJSON())
  ]);
  const verticalGeometry = {
    stateTop: stateBarBounds.top,
    stateBottom: stateBarBounds.bottom,
    fieldTop: timeoutFieldBounds.top,
    fieldBottom: timeoutFieldBounds.bottom
  };
  assert.ok(verticalGeometry.fieldBottom <= verticalGeometry.stateTop || verticalGeometry.fieldTop >= verticalGeometry.stateBottom);

  await page.getByLabel('语言模型').fill('Qwen/Qwen3-14B');
  assert.equal(await page.getByText('存在未保存的草稿修改').isVisible(), true);
  assert.equal(await page.getByText('语言模型已修改').isVisible(), true);
  await page.getByRole('button', { name: '重置到已保存草稿' }).click();
  assert.equal(await page.getByLabel('语言模型').inputValue(), 'Qwen/Qwen3-32B');
  assert.equal(await page.getByText('存在未保存的草稿修改').count(), 0);

  await page.getByLabel('Provider API 密钥').fill('replacement-value');
  assert.equal(await page.getByText('Provider API 密钥已修改').isVisible(), true);
  await page.getByRole('button', { name: '重置到已保存草稿' }).click();
  assert.equal(await page.getByText('Provider API 密钥已修改').count(), 0);

  await page.getByLabel('语言模型').fill('Qwen/Qwen3-14B');
  await page.getByRole('button', { name: '保存并应用', exact: true }).click();
  await page.getByRole('button', { name: '正在应用版本 3。' }).waitFor();
  assert.equal(saves.length, 1);
  assert.equal(saves[0].llm_model, 'Qwen/Qwen3-14B');
  assert.equal(saves[0].provider_api_key, null);
  assert.deepEqual(applications, [{ version: 3 }]);
  assert.equal(await page.getByText('已保存版本 3').isVisible(), true);
  assert.equal(await page.getByText('生效版本 1').isVisible(), true);
  assert.equal(await page.getByRole('button', { name: '正在应用版本 3。' }).isDisabled(), true);
  assert.equal(await page.getByLabel('语言模型').isDisabled(), true);
  await page.getByText('生效版本 3').waitFor();
  assert.equal(await page.getByText('设置已生效。').isVisible(), true);
  await page.reload();
  await page.getByText('生效版本 3').waitFor();
  assert.equal(await page.getByText('设置已生效。').isVisible(), true);
  assert.equal(await page.getByText('存在未保存的草稿修改').count(), 0);
});

test('System Administrator sees a failed application after refresh and can retry the saved version', { timeout: 30000 }, async (t) => {
  const applications = [];
  let retry = false;
  const { page, baseUrl } = await startSettingsWorkspace(t, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'operator', role: 'admin' }));
      return;
    }
    if (path === '/api/settings/draft' && request.method() === 'GET') {
      await route.fulfill(
        jsonResponse(
          retry
            ? {
                ...settingsDraft,
                application_state: 'active',
                active_version: 2,
                application: { version: 2, actor: 'operator', at: '2026-07-31T10:25:00Z', message: 'settings version is active' }
              }
            : {
                ...settingsDraft,
                application_state: 'failed',
                application: { version: 2, actor: 'operator', at: '2026-07-31T10:24:00Z', message: 'runtime rejected the saved configuration' }
              }
        )
      );
      return;
    }
    if (path === '/api/settings/apply' && request.method() === 'POST') {
      applications.push(request.postDataJSON());
      retry = true;
      await route.fulfill(
        jsonResponse({
          ...settingsDraft,
          application_state: 'applying',
          application: { version: 2, actor: 'operator', at: '2026-07-31T10:25:00Z', message: 'settings version is applying' }
        })
      );
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await page.addInitScript(() => localStorage.setItem('access_token', 'admin-token'));
  await page.goto(`${baseUrl}config`);
  await page.getByText('应用失败：runtime rejected the saved configuration').waitFor();
  assert.equal(await page.getByText('生效版本 1').isVisible(), true);

  await page.getByRole('button', { name: '重试应用版本 2', exact: true }).click();
  await page.getByRole('button', { name: '正在应用版本 2。' }).waitFor();
  assert.deepEqual(applications, [{ version: 2 }]);
  await page.getByText('生效版本 2').waitFor();
  assert.equal(await page.getByText('设置已生效。').isVisible(), true);
});

test('settings draft validation keeps edits visible and exposes field-level feedback without showing a secret', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startSettingsWorkspace(t, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'operator', role: 'admin' }));
      return;
    }
    if (path === '/api/settings/draft' && request.method() === 'GET') {
      await route.fulfill(jsonResponse(settingsDraft));
      return;
    }
    if (path === '/api/settings/draft' && request.method() === 'PUT') {
      await route.fulfill({
        status: 400,
        contentType: 'application/json',
        body: JSON.stringify({
          code: 'VALIDATION_ERROR',
          message: 'system settings draft is invalid',
          detail: { fields: { llm_model: 'model identifier is unsafe or unsupported' } }
        })
      });
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await page.addInitScript(() => localStorage.setItem('access_token', 'admin-token'));
  await page.goto(`${baseUrl}config`);
  await page.getByRole('heading', { name: '系统设置' }).waitFor();
  await page.getByLabel('语言模型').fill('unsafe value');
  await page.getByRole('button', { name: '保存并应用', exact: true }).click();

  await page.getByRole('alert').waitFor();
  assert.equal(await page.getByRole('alert').innerText(), '草稿未保存，请修正标记字段后重试。');
  assert.equal(await page.getByText('model identifier is unsafe or unsupported').isVisible(), true);
  assert.equal(await page.getByLabel('语言模型').inputValue(), 'unsafe value');
  assert.equal(await page.getByText('存在未保存的草稿修改').isVisible(), true);
});

test('Knowledge User is redirected before the gated settings draft can load', { timeout: 30000 }, async (t) => {
  const requests = [];
  const { page, baseUrl } = await startSettingsWorkspace(t, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    requests.push(`${request.method()} ${path}`);
    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'knowledge-user', role: 'user' }));
      return;
    }
    if (path === '/api/sessions') {
      await route.fulfill(jsonResponse({ sessions: [] }));
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await page.addInitScript(() => localStorage.setItem('access_token', 'knowledge-user-token'));
  await page.goto(`${baseUrl}config`);

  await page.waitForURL(/\/chat\?notice=admin-required$/);
  assert.equal(await page.getByRole('heading', { name: '系统设置' }).count(), 0);
  assert.equal(await page.getByRole('link', { name: '系统设置' }).count(), 0);
  assert.equal(requests.some((request) => request.includes('/api/settings/draft')), false);
});

test('System Settings keeps its desktop-only boundary explicit on a mobile viewport', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startSettingsWorkspace(
    t,
    async (route) => {
      const path = new URL(route.request().url()).pathname;
      if (path === '/api/auth/me') {
        await route.fulfill(jsonResponse({ username: 'operator', role: 'admin' }));
        return;
      }
      if (path === '/api/settings/draft') {
        await route.fulfill(jsonResponse(settingsDraft));
        return;
      }
      await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
    },
    { width: 390, height: 844 }
  );

  await page.addInitScript(() => localStorage.setItem('access_token', 'admin-token'));
  await page.goto(`${baseUrl}config`);
  await page.getByRole('heading', { name: '系统设置' }).waitFor();

  assert.equal(await page.getByText('系统设置当前仅支持桌面工作区。').isVisible(), true);
  assert.equal(await page.getByRole('form').count(), 0);
  assert.equal(await page.getByLabel('草稿状态').count(), 0);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
});
