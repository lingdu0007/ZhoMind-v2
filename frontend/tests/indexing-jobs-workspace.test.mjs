import assert from 'node:assert/strict';
import test from 'node:test';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const jsonResponse = (data, status = 200) => ({
  status,
  contentType: 'application/json',
  body: JSON.stringify({ data })
});

const startIndexingJobsWorkspace = async (t, apiHandler, viewport = { width: 1440, height: 900 }) => {
  const testPort = 43000 + Math.floor(Math.random() * 1000);
  const server = await createServer({ server: { host: '127.0.0.1', port: testPort, strictPort: true } });
  await server.listen();

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport });
  page.setDefaultTimeout(5000);
  await page.route((url) => url.pathname.startsWith('/api/'), apiHandler);

  t.after(async () => {
    await browser.close();
    await server.close();
  });

  return { page, baseUrl: server.resolvedUrls.local[0] };
};

const activeJob = {
  job_id: 'job-active',
  document_id: 'doc-operations',
  status: 'running',
  stage: 'chunking',
  progress: 56,
  message: '正在生成可检索分块',
  updated_at: '2026-07-30T09:30:00Z'
};

const terminalJob = {
  job_id: 'job-finished',
  document_id: 'doc-handbook',
  status: 'succeeded',
  stage: 'completed',
  progress: 100,
  message: '索引已发布',
  updated_at: '2026-07-30T09:20:00Z'
};

const authenticateAdmin = (page) =>
  page.addInitScript(() => localStorage.setItem('access_token', 'administrator-token'));

test('System Administrator can open a separate Indexing Jobs workspace, filter lifecycle states, and cancel an eligible job', { timeout: 30000 }, async (t) => {
  const requests = [];
  const { page, baseUrl } = await startIndexingJobsWorkspace(t, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    requests.push(`${request.method()} ${path}`);

    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'operator', role: 'admin' }));
      return;
    }
    if (path === '/api/documents/jobs' && request.method() === 'GET') {
      await route.fulfill(jsonResponse({ items: [activeJob, terminalJob] }));
      return;
    }
    if (path === '/api/documents/jobs/job-active/cancel' && request.method() === 'POST') {
      await route.fulfill(jsonResponse({ ...activeJob, status: 'canceled', stage: 'failed', message: '任务已取消' }));
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await authenticateAdmin(page);
  await page.goto(`${baseUrl}jobs`);

  await page.getByRole('heading', { name: '构建任务' }).waitFor();
  assert.equal(await page.getByRole('link', { name: '构建任务' }).isVisible(), true);
  assert.equal(await page.getByText('文档库').count(), 0);
  assert.equal(await page.getByText('job-active').isVisible(), true);
  assert.equal(await page.getByText('doc-operations').isVisible(), true);
  assert.equal(await page.getByText('执行中 (running)').isVisible(), true);
  assert.equal(await page.getByText('分块中 (chunking)').isVisible(), true);
  assert.equal(await page.getByText('正在生成可检索分块').isVisible(), true);
  assert.equal(await page.getByRole('button', { name: '取消任务 job-active' }).isVisible(), true);
  assert.equal(await page.getByRole('button', { name: '取消任务 job-finished' }).count(), 0);

  const [tableBox, cancelBox] = await Promise.all([
    page.getByRole('table').boundingBox(),
    page.getByRole('button', { name: '取消任务 job-active' }).boundingBox()
  ]);
  assert.ok(tableBox.x + tableBox.width <= 1440);
  assert.ok(cancelBox.x >= tableBox.x);
  assert.ok(cancelBox.x + cancelBox.width <= tableBox.x + tableBox.width);

  await page.getByRole('radio', { name: '仅进行中' }).check();
  assert.equal(await page.getByText('job-active').isVisible(), true);
  assert.equal(await page.getByText('job-finished').count(), 0);

  await page.getByRole('radio', { name: '仅已结束' }).check();
  assert.equal(await page.getByText('job-active').count(), 0);
  assert.equal(await page.getByText('job-finished').isVisible(), true);
  assert.equal(await page.getByText('已成功 (succeeded)').isVisible(), true);

  await page.getByRole('radio', { name: '仅进行中' }).check();
  await page.getByRole('button', { name: '取消任务 job-active' }).click();
  await page.getByText('当前没有进行中的构建任务。').waitFor();
  await page.getByRole('radio', { name: '全部状态' }).check();
  await page.getByText('已取消 (canceled)').waitFor();
  assert.equal(await page.getByRole('button', { name: '取消任务 job-active' }).count(), 0);
  assert.equal(requests.includes('POST /api/documents/jobs/job-active/cancel'), true);
});

test('Knowledge User is redirected before the Indexing Jobs workspace or protected jobs endpoint can render', { timeout: 30000 }, async (t) => {
  const requests = [];
  const { page, baseUrl } = await startIndexingJobsWorkspace(t, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    requests.push(`${request.method()} ${path}`);

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

  await page.addInitScript(() => localStorage.setItem('access_token', 'knowledge-user-token'));
  await page.goto(`${baseUrl}jobs`);

  await page.waitForURL(/\/chat\?notice=admin-required$/);
  assert.equal(await page.getByRole('heading', { name: '构建任务' }).count(), 0);
  assert.equal(await page.getByRole('link', { name: '构建任务' }).count(), 0);
  assert.equal(requests.some((request) => request.includes('/api/documents/jobs')), false);
});

test('active jobs poll until terminal, then stop polling when the workspace is left', { timeout: 30000 }, async (t) => {
  let jobReads = 0;
  const { page, baseUrl } = await startIndexingJobsWorkspace(t, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;

    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'operator', role: 'admin' }));
      return;
    }
    if (path === '/api/documents/jobs' && request.method() === 'GET') {
      await route.fulfill(jsonResponse({ items: [activeJob] }));
      return;
    }
    if (path === '/api/documents/jobs/job-active' && request.method() === 'GET') {
      jobReads += 1;
      await route.fulfill(
        jsonResponse(
          jobReads === 1
            ? { ...activeJob, progress: 80 }
            : { ...activeJob, status: 'succeeded', stage: 'completed', progress: 100, message: '索引已发布' }
        )
      );
      return;
    }
    if (path === '/api/sessions') {
      await route.fulfill(jsonResponse({ sessions: [] }));
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await authenticateAdmin(page);
  await page.goto(`${baseUrl}jobs`);
  await page.getByRole('heading', { name: '构建任务' }).waitFor();
  await page.getByText('已成功 (succeeded)').waitFor({ timeout: 8000 });
  assert.equal(jobReads, 2);

  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();
  await page.waitForTimeout(450);
  assert.equal(jobReads, 2);
});

test('empty and refresh-failed lists preserve the stable desktop controls', { timeout: 30000 }, async (t) => {
  let listRequests = 0;
  const { page, baseUrl } = await startIndexingJobsWorkspace(t, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;

    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'operator', role: 'admin' }));
      return;
    }
    if (path === '/api/documents/jobs') {
      listRequests += 1;
      if (listRequests === 1) {
        await route.fulfill(jsonResponse({ items: [] }));
        return;
      }
      await route.fulfill(jsonResponse({ message: 'jobs service unavailable' }, 503));
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await authenticateAdmin(page);
  await page.goto(`${baseUrl}jobs`);
  await page.getByText('当前没有构建任务。').waitFor();
  assert.equal(await page.getByRole('table').isVisible(), true);
  assert.equal(await page.getByRole('group', { name: '任务状态筛选' }).isVisible(), true);

  await page.getByRole('button', { name: '刷新' }).click();
  await page.getByRole('alert').waitFor();
  assert.match(await page.getByRole('alert').innerText(), /加载构建任务失败，请重新加载。/);
  assert.equal(await page.getByRole('table').isVisible(), true);
  assert.equal(await page.getByRole('button', { name: '重新加载' }).isVisible(), true);
});

test('a backend-rejected cancellation retains the active job and explains the next action', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startIndexingJobsWorkspace(t, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;

    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'operator', role: 'admin' }));
      return;
    }
    if (path === '/api/documents/jobs' && request.method() === 'GET') {
      await route.fulfill(jsonResponse({ items: [activeJob] }));
      return;
    }
    if (path === '/api/documents/jobs/job-active/cancel' && request.method() === 'POST') {
      await route.fulfill(jsonResponse({ message: 'job is already claimed by a worker' }, 409));
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await authenticateAdmin(page);
  await page.goto(`${baseUrl}jobs`);
  await page.getByRole('button', { name: '取消任务 job-active' }).click();
  await page.getByRole('alert').waitFor();
  assert.equal(
    await page.getByRole('alert').innerText(),
    '取消任务 job-active 失败：请稍后重试。'
  );
  assert.equal(await page.getByRole('cell', { name: 'job-active', exact: true }).isVisible(), true);
  assert.equal(await page.getByRole('button', { name: '取消任务 job-active' }).isVisible(), true);
});

test('Indexing Jobs represents failed refresh and desktop-only mobile scope without rendering a dense table', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startIndexingJobsWorkspace(
    t,
    async (route) => {
      const request = route.request();
      const path = new URL(request.url()).pathname;

      if (path === '/api/auth/me') {
        await route.fulfill(jsonResponse({ username: 'operator', role: 'admin' }));
        return;
      }
      if (path === '/api/documents/jobs') {
        await route.fulfill(jsonResponse({ message: 'jobs service unavailable' }, 503));
        return;
      }
      await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
    },
    { width: 390, height: 844 }
  );

  await authenticateAdmin(page);
  await page.goto(`${baseUrl}jobs`);

  await page.getByRole('heading', { name: '构建任务' }).waitFor();
  assert.equal(await page.getByText('构建任务当前仅支持桌面工作区。').isVisible(), true);
  assert.equal(await page.getByRole('table').count(), 0);
  assert.equal(await page.getByRole('button', { name: '刷新' }).count(), 0);
  assert.equal(await page.getByRole('button', { name: '重新加载' }).count(), 0);
});
