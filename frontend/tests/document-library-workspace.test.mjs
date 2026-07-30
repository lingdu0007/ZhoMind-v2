import assert from 'node:assert/strict';
import test from 'node:test';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const jsonResponse = (data, status = 200) => ({
  status,
  contentType: 'application/json',
  body: JSON.stringify({ data })
});

const startDocumentLibrary = async (t, apiHandler, viewport = { width: 1440, height: 900 }) => {
  const testPort = 44000 + Math.floor(Math.random() * 1000);
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

const authenticateAdmin = (page) =>
  page.addInitScript(() => localStorage.setItem('access_token', 'administrator-token'));

const documents = [
  {
    document_id: 'doc-pending',
    filename: '待处理文档.txt',
    file_type: 'txt',
    file_size: 512,
    status: 'pending',
    chunk_count: 0,
    uploaded_at: '2026-07-30T09:00:00Z'
  },
  {
    document_id: 'doc-processing',
    filename: '处理中记录.md',
    file_type: 'md',
    file_size: 1536,
    status: 'processing',
    chunk_count: 0,
    uploaded_at: '2026-07-30T09:10:00Z'
  },
  {
    document_id: 'doc-ready',
    filename: '运行手册.pdf',
    file_type: 'pdf',
    file_size: 2 * 1024 * 1024,
    status: 'ready',
    chunk_count: 36,
    uploaded_at: '2026-07-30T09:20:00Z'
  },
  {
    document_id: 'doc-failed',
    filename: '失败导入.md',
    file_type: 'md',
    file_size: 784,
    status: 'failed',
    chunk_count: 0,
    uploaded_at: '2026-07-30T09:30:00Z'
  },
  {
    document_id: 'doc-deleting',
    filename: '删除中附件.txt',
    file_type: 'txt',
    file_size: 128,
    status: 'deleting',
    chunk_count: 4,
    uploaded_at: '2026-07-30T09:40:00Z'
  }
];

test('System Administrator can scan Document Library and filter the loaded inventory', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startDocumentLibrary(t, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;

    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'operator', role: 'admin' }));
      return;
    }
    if (path === '/api/documents' && request.method() === 'GET') {
      await route.fulfill(jsonResponse({ items: documents }));
      return;
    }
    if (path === '/api/documents/jobs' && request.method() === 'GET') {
      await route.fulfill(
        jsonResponse({
          items: [
            {
              job_id: 'job-ready',
              document_id: 'doc-ready',
              status: 'succeeded',
              stage: 'completed',
              progress: 100,
              message: '索引已发布',
              updated_at: '2026-07-30T09:25:00Z'
            }
          ]
        })
      );
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await authenticateAdmin(page);
  await page.goto(`${baseUrl}documents`);

  await page.getByRole('heading', { name: '文档库' }).waitFor();
  assert.equal(await page.getByRole('link', { name: '文档库' }).isVisible(), true);
  assert.equal(await page.getByRole('link', { name: '构建任务' }).isVisible(), true);
  assert.equal(await page.getByText('支持的格式：TXT、Markdown (.md)、PDF。').isVisible(), true);
  assert.equal(await page.getByText('待处理 (pending)').isVisible(), true);
  assert.equal(await page.getByText('处理中 (processing)').isVisible(), true);
  assert.equal(await page.getByText('可检索 (ready)').isVisible(), true);
  assert.equal(await page.getByText('构建失败 (failed)').isVisible(), true);
  assert.equal(await page.getByText('删除中 (deleting)').isVisible(), true);
  assert.equal(await page.getByText('36').isVisible(), true);
  assert.equal(await page.getByText('2.0 MB').isVisible(), true);
  assert.equal(await page.getByText('高级分块策略').count(), 0);
  assert.equal(await page.getByRole('button', { name: '查看文档 doc-ready 的构建任务' }).isVisible(), true);

  const [tableBox, refreshBox] = await Promise.all([
    page.getByRole('table').boundingBox(),
    page.getByRole('button', { name: '刷新' }).boundingBox()
  ]);
  assert.ok(tableBox.x + tableBox.width <= 1440);
  assert.ok(refreshBox.x >= tableBox.x);
  assert.ok(refreshBox.x + refreshBox.width <= 1440);

  await page.getByLabel('按状态筛选').selectOption('ready');
  assert.equal(await page.getByText('运行手册.pdf').isVisible(), true);
  assert.equal(await page.getByText('待处理文档.txt').count(), 0);

  await page.getByLabel('按文件名搜索').fill('运行手册');
  assert.equal(await page.getByText('运行手册.pdf').isVisible(), true);

  await page.getByLabel('按文件名搜索').fill('不存在的文件');
  assert.equal(await page.getByText('没有符合当前筛选条件的文档。').isVisible(), true);

  await page.getByLabel('按文件名搜索').fill('运行手册');
  await page.getByRole('button', { name: '查看文档 doc-ready 的构建任务' }).click();
  await page.waitForURL(/\/jobs\?document=doc-ready$/);
  assert.equal(await page.getByText('正在查看文档 doc-ready 的构建任务。').isVisible(), true);
  assert.equal(await page.getByText('job-ready').isVisible(), true);
});

test('Knowledge User is redirected before Document Library or its protected inventory can render', { timeout: 30000 }, async (t) => {
  const requests = [];
  const { page, baseUrl } = await startDocumentLibrary(t, async (route) => {
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
  await page.goto(`${baseUrl}documents`);

  await page.waitForURL(/\/chat\?notice=admin-required$/);
  assert.equal(await page.getByRole('heading', { name: '文档库' }).count(), 0);
  assert.equal(await page.getByRole('link', { name: '文档库' }).count(), 0);
  assert.equal(requests.some((request) => request.includes('/api/documents')), false);
});

test('initial upload accepts a supported file, relies on the general strategy contract, refreshes inventory, and hands off to its Indexing Job', { timeout: 30000 }, async (t) => {
  let listReads = 0;
  let uploadBody = '';
  let preciseJobReads = 0;
  const { page, baseUrl } = await startDocumentLibrary(t, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;

    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'operator', role: 'admin' }));
      return;
    }
    if (path === '/api/documents' && request.method() === 'GET') {
      listReads += 1;
      await route.fulfill(
        jsonResponse({
          items:
            listReads === 1
              ? []
              : [
                  {
                    document_id: 'doc-new',
                    filename: 'initial-notes.md',
                    file_type: 'md',
                    file_size: 7,
                    status: 'pending',
                    chunk_count: 0,
                    uploaded_at: '2026-07-30T10:00:00Z'
                  }
                ]
        })
      );
      return;
    }
    if (path === '/api/documents/upload' && request.method() === 'POST') {
      uploadBody = request.postDataBuffer().toString('utf8');
      await route.fulfill(jsonResponse({ document_id: 'doc-new', job_id: 'job-new' }));
      return;
    }
    if (path === '/api/documents/jobs' && request.method() === 'GET') {
      await route.fulfill(jsonResponse({ items: [] }));
      return;
    }
    if (path === '/api/documents/jobs/job-new' && request.method() === 'GET') {
      preciseJobReads += 1;
      await route.fulfill(
        jsonResponse({
          job_id: 'job-new',
          document_id: 'doc-new',
          status: 'queued',
          stage: 'queued',
          progress: 0,
          message: 'queued for build',
          updated_at: '2026-07-30T10:00:00Z'
        })
      );
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await authenticateAdmin(page);
  await page.goto(`${baseUrl}documents`);
  await page.getByRole('heading', { name: '文档库' }).waitFor();

  await page.getByLabel('选择文档').setInputFiles({
    name: 'initial-notes.md',
    mimeType: 'text/markdown',
    buffer: Buffer.from('# notes')
  });
  await page.getByRole('button', { name: '上传文档' }).click();

  await page.getByText('文档 ID：doc-new').waitFor();
  assert.equal(await page.getByText('构建任务 ID：job-new').isVisible(), true);
  assert.equal(await page.getByRole('cell', { name: 'initial-notes.md' }).isVisible(), true);
  assert.ok(uploadBody.includes('initial-notes.md'));
  assert.equal(uploadBody.includes('chunk_strategy'), false);
  assert.equal(await page.getByText('初始上传将自动使用通用分块策略。').isVisible(), true);

  await page.getByRole('button', { name: '查看构建任务 job-new' }).click();
  await page.waitForURL(/\/jobs\?job=job-new$/);
  assert.equal(await page.getByText('正在查看任务 job-new。').isVisible(), true);
  await page.getByRole('cell', { name: 'job-new', exact: true }).waitFor();
  assert.equal(preciseJobReads, 1);
});

test('unsupported files and backend upload validation failures retain a recoverable Document Library', { timeout: 30000 }, async (t) => {
  let uploadRequests = 0;
  const { page, baseUrl } = await startDocumentLibrary(t, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;

    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'operator', role: 'admin' }));
      return;
    }
    if (path === '/api/documents' && request.method() === 'GET') {
      await route.fulfill(jsonResponse({ items: [documents[2]] }));
      return;
    }
    if (path === '/api/documents/upload' && request.method() === 'POST') {
      uploadRequests += 1;
      await route.fulfill({
        status: 422,
        contentType: 'application/json',
        body: JSON.stringify({ code: 'VALIDATION_ERROR', message: 'backend validation details' })
      });
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await authenticateAdmin(page);
  await page.goto(`${baseUrl}documents`);
  await page.getByRole('heading', { name: '文档库' }).waitFor();

  await page.getByLabel('选择文档').setInputFiles({
    name: 'unsupported.docx',
    mimeType: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    buffer: Buffer.from('not supported')
  });
  await page.getByRole('alert').waitFor();
  assert.equal(await page.getByRole('alert').innerText(), '不支持的文件格式。仅支持 TXT、Markdown (.md) 和 PDF。');
  assert.equal(uploadRequests, 0);
  assert.equal(await page.getByText('运行手册.pdf').isVisible(), true);

  await page.getByLabel('选择文档').setInputFiles({
    name: 'duplicate.pdf',
    mimeType: 'application/pdf',
    buffer: Buffer.from('valid type')
  });
  await page.getByRole('button', { name: '上传文档' }).click();
  await page.getByRole('alert').waitFor();
  assert.equal(
    await page.getByRole('alert').innerText(),
    '上传被服务端拒绝，请检查文件名或内容后重试。'
  );
  assert.equal(uploadRequests, 1);
  assert.equal(await page.getByText('duplicate.pdf').count(), 1);
  assert.equal(await page.getByRole('button', { name: '上传文档' }).isVisible(), true);
  assert.equal(await page.getByText('运行手册.pdf').isVisible(), true);
});

test('empty and refresh-failed Document Library states preserve the desktop controls', { timeout: 30000 }, async (t) => {
  let listReads = 0;
  const { page, baseUrl } = await startDocumentLibrary(t, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;

    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'operator', role: 'admin' }));
      return;
    }
    if (path === '/api/documents' && request.method() === 'GET') {
      listReads += 1;
      if (listReads === 1) {
        await route.fulfill(jsonResponse({ items: [] }));
        return;
      }
      await route.fulfill({
        status: 503,
        contentType: 'application/json',
        body: JSON.stringify({ message: 'document service unavailable' })
      });
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await authenticateAdmin(page);
  await page.goto(`${baseUrl}documents`);
  await page.getByText('当前文档库为空。').waitFor();
  assert.equal(await page.getByRole('table').isVisible(), true);
  assert.equal(await page.getByLabel('按状态筛选').isVisible(), true);

  await page.getByRole('button', { name: '刷新' }).click();
  await page.getByRole('alert').waitFor();
  assert.match(await page.getByRole('alert').innerText(), /加载文档库失败，请重新加载。/);
  assert.equal(await page.getByRole('table').isVisible(), true);
  assert.equal(await page.getByRole('button', { name: '重新加载' }).isVisible(), true);
});

test('Document Library exposes a stable loading state before the inventory response resolves', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startDocumentLibrary(t, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;

    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'operator', role: 'admin' }));
      return;
    }
    if (path === '/api/documents' && request.method() === 'GET') {
      await new Promise((resolve) => setTimeout(resolve, 300));
      await route.fulfill(jsonResponse({ items: [] }));
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await authenticateAdmin(page);
  await page.goto(`${baseUrl}documents`);
  await page.getByRole('heading', { name: '文档库' }).waitFor();
  await page.getByText('正在加载文档库...').waitFor();
  assert.equal(await page.getByRole('table').isVisible(), true);
  assert.equal(await page.getByLabel('按状态筛选').isVisible(), true);
  await page.getByText('当前文档库为空。').waitFor();
});

test('Document Library reports desktop scope on mobile without fetching or rendering dense operations', { timeout: 30000 }, async (t) => {
  const requests = [];
  const { page, baseUrl } = await startDocumentLibrary(
    t,
    async (route) => {
      const request = route.request();
      const path = new URL(request.url()).pathname;
      requests.push(`${request.method()} ${path}`);

      if (path === '/api/auth/me') {
        await route.fulfill(jsonResponse({ username: 'operator', role: 'admin' }));
        return;
      }
      await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
    },
    { width: 390, height: 844 }
  );

  await authenticateAdmin(page);
  await page.goto(`${baseUrl}documents`);

  await page.getByRole('heading', { name: '文档库' }).waitFor();
  assert.equal(await page.getByText('文档库当前仅支持桌面工作区。').isVisible(), true);
  assert.equal(await page.getByRole('table').count(), 0);
  assert.equal(await page.getByLabel('选择文档').count(), 0);
  assert.equal(await page.getByRole('button', { name: '刷新' }).count(), 0);
  assert.equal(requests.some((request) => request.includes('/api/documents')), false);
});
