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

test('System Administrator can inspect paginated published chunks for a ready document', { timeout: 30000 }, async (t) => {
  const requestedPages = [];
  const readyDocument = {
    document_id: 'doc-ready',
    filename: 'incident-handbook.md',
    file_type: 'md',
    file_size: 2048,
    status: 'ready',
    chunk_count: 2,
    published_generation: 1,
    uploaded_at: '2026-07-30T10:00:00Z'
  };
  const { page, baseUrl } = await startDocumentLibrary(t, async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;

    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'operator', role: 'admin' }));
      return;
    }
    if (path === '/api/documents' && request.method() === 'GET') {
      await route.fulfill(jsonResponse({ items: [readyDocument] }));
      return;
    }
    if (path === '/api/documents/doc-ready/chunks' && request.method() === 'GET') {
      const requestedPage = Number(url.searchParams.get('page'));
      requestedPages.push(requestedPage);
      await new Promise((resolve) => setTimeout(resolve, 150));
      await route.fulfill(
        jsonResponse(
          requestedPage === 1
            ? {
                items: [
                  {
                    chunk_id: 'chunk-1',
                    document_id: 'doc-ready',
                    chunk_index: 0,
                    content: '发布代际只会在构建任务成功后切换。',
                    keywords: ['发布', '构建'],
                    generated_questions: ['何时切换已发布分块？'],
                    metadata: { source: 'incident-handbook', section: 'release' }
                  }
                ],
                pagination: { page: 1, page_size: 10, total: 11 }
              }
            : {
                items: [
                  {
                    chunk_id: 'chunk-2',
                    document_id: 'doc-ready',
                    chunk_index: 1,
                    content: '候选分块在发布前不参与检索。',
                    keywords: ['候选'],
                    generated_questions: [],
                    metadata: { source: 'incident-handbook', section: 'candidate' }
                  }
                ],
                pagination: { page: 2, page_size: 10, total: 11 }
              }
        )
      );
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await authenticateAdmin(page);
  await page.goto(`${baseUrl}documents`);
  await page.getByRole('heading', { name: '文档库' }).waitFor();

  await page.getByRole('button', { name: '查看文档 doc-ready 的已发布分块' }).click();
  await page.getByText('正在加载已发布分块...').waitFor();
  await page.getByRole('heading', { name: '已发布分块' }).waitFor();
  assert.equal(await page.getByText('文档 ID：doc-ready').isVisible(), true);
  await page.getByText('发布代际只会在构建任务成功后切换。').waitFor();
  assert.equal(await page.getByText('关键词：发布、构建').isVisible(), true);
  assert.equal(await page.getByText('生成问题：何时切换已发布分块？').isVisible(), true);
  assert.equal(await page.getByText('source', { exact: true }).isVisible(), true);
  assert.equal(await page.getByText('incident-handbook', { exact: true }).isVisible(), true);
  assert.equal(await page.getByText('第 1 / 2 页').isVisible(), true);

  await page.getByRole('button', { name: '下一页' }).click();
  await page.getByText('候选分块在发布前不参与检索。').waitFor();
  assert.equal(await page.getByText('section', { exact: true }).isVisible(), true);
  assert.equal(await page.getByText('candidate', { exact: true }).isVisible(), true);
  assert.deepEqual(requestedPages, [1, 2]);
});

test('chunk inspection keeps not-ready, empty, and failed responses recoverable', { timeout: 30000 }, async (t) => {
  let staleChunkReads = 0;
  const inspectionDocuments = [
    {
      document_id: 'doc-stale',
      filename: 'stale-status.md',
      file_type: 'md',
      file_size: 1024,
      status: 'ready',
      chunk_count: 1,
      published_generation: 1,
      uploaded_at: '2026-07-30T10:30:00Z'
    },
    {
      document_id: 'doc-read-failure',
      filename: 'read-failure.md',
      file_type: 'md',
      file_size: 1024,
      status: 'ready',
      chunk_count: 1,
      published_generation: 1,
      uploaded_at: '2026-07-30T10:31:00Z'
    },
    {
      document_id: 'doc-failed',
      filename: 'failed-build.md',
      file_type: 'md',
      file_size: 1024,
      status: 'failed',
      chunk_count: 0,
      published_generation: 1,
      uploaded_at: '2026-07-30T10:32:00Z'
    }
  ];
  const { page, baseUrl } = await startDocumentLibrary(t, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;

    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'operator', role: 'admin' }));
      return;
    }
    if (path === '/api/documents' && request.method() === 'GET') {
      await route.fulfill(jsonResponse({ items: inspectionDocuments }));
      return;
    }
    if (path === '/api/documents/doc-stale/chunks' && request.method() === 'GET') {
      staleChunkReads += 1;
      if (staleChunkReads === 1) {
        await route.fulfill(jsonResponse({ code: 'DOC_CHUNK_RESULT_NOT_READY' }, 409));
        return;
      }
      await route.fulfill(jsonResponse({ items: [], pagination: { page: 1, page_size: 10, total: 0 } }));
      return;
    }
    if (path === '/api/documents/doc-read-failure/chunks' && request.method() === 'GET') {
      await route.fulfill(jsonResponse({ message: 'chunk store unavailable' }, 503));
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await authenticateAdmin(page);
  await page.goto(`${baseUrl}documents`);
  await page.getByRole('heading', { name: '文档库' }).waitFor();
  assert.equal(await page.getByRole('button', { name: '查看文档 doc-failed 的已发布分块' }).count(), 0);

  await page.getByRole('button', { name: '查看文档 doc-stale 的已发布分块' }).click();
  await page.getByRole('alert').waitFor();
  assert.match(await page.getByRole('alert').innerText(), /当前文档尚未产生可查看的已发布分块，请等待构建完成后重试。/);
  await page.getByRole('button', { name: '重新检查' }).click();
  await page.getByText('当前已发布版本没有可展示的分块。').waitFor();

  await page.keyboard.press('Escape');
  await page.getByRole('button', { name: '查看文档 doc-read-failure 的已发布分块' }).click();
  await page.getByRole('alert').waitFor();
  assert.match(await page.getByRole('alert').innerText(), /加载已发布分块失败，请重新检查。/);
});

test('closing chunk inspection prevents an older response from replacing a newer document', { timeout: 30000 }, async (t) => {
  let firstChunkRequested;
  const firstChunkRequest = new Promise((resolve) => {
    firstChunkRequested = resolve;
  });
  let releaseFirstChunk;
  const firstChunkResponse = new Promise((resolve) => {
    releaseFirstChunk = resolve;
  });
  t.after(() => releaseFirstChunk());

  const documentsForRace = [
    {
      document_id: 'doc-first',
      filename: 'first.md',
      file_type: 'md',
      file_size: 1024,
      status: 'ready',
      chunk_count: 1,
      published_generation: 1,
      uploaded_at: '2026-07-30T10:45:00Z'
    },
    {
      document_id: 'doc-second',
      filename: 'second.md',
      file_type: 'md',
      file_size: 1024,
      status: 'ready',
      chunk_count: 1,
      published_generation: 1,
      uploaded_at: '2026-07-30T10:46:00Z'
    }
  ];
  const { page, baseUrl } = await startDocumentLibrary(t, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;

    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'operator', role: 'admin' }));
      return;
    }
    if (path === '/api/documents' && request.method() === 'GET') {
      await route.fulfill(jsonResponse({ items: documentsForRace }));
      return;
    }
    if (path === '/api/documents/doc-first/chunks' && request.method() === 'GET') {
      firstChunkRequested();
      await firstChunkResponse;
      await route.fulfill(
        jsonResponse({
          items: [{ chunk_id: 'chunk-first', chunk_index: 0, content: '旧响应内容', keywords: [], generated_questions: [], metadata: {} }],
          pagination: { page: 1, page_size: 10, total: 1 }
        })
      );
      return;
    }
    if (path === '/api/documents/doc-second/chunks' && request.method() === 'GET') {
      await route.fulfill(
        jsonResponse({
          items: [{ chunk_id: 'chunk-second', chunk_index: 0, content: '当前文档内容', keywords: [], generated_questions: [], metadata: {} }],
          pagination: { page: 1, page_size: 10, total: 1 }
        })
      );
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await authenticateAdmin(page);
  await page.goto(`${baseUrl}documents`);
  await page.getByRole('heading', { name: '文档库' }).waitFor();
  await page.getByRole('button', { name: '查看文档 doc-first 的已发布分块' }).click();
  await firstChunkRequest;
  await page.keyboard.press('Escape');
  await page.getByRole('heading', { name: '已发布分块' }).waitFor({ state: 'hidden' });

  await page.getByRole('button', { name: '查看文档 doc-second 的已发布分块' }).click();
  await page.getByText('当前文档内容').waitFor();
  const firstChunkSettled = page.waitForResponse(
    (response) => new URL(response.url()).pathname === '/api/documents/doc-first/chunks'
  );
  releaseFirstChunk();
  await firstChunkSettled;
  assert.equal(await page.getByText('当前文档内容').isVisible(), true);
  assert.equal(await page.getByText('旧响应内容').count(), 0);
});

test('Document Library retains a published generation through rebuild enqueue and processing failures', { timeout: 30000 }, async (t) => {
  let listReads = 0;
  let buildRequests = 0;
  const initialDocuments = [
    {
      document_id: 'doc-published',
      filename: 'published-handbook.md',
      file_type: 'md',
      file_size: 1024,
      status: 'ready',
      chunk_count: 12,
      published_generation: 3,
      uploaded_at: '2026-07-30T10:50:00Z'
    },
    {
      document_id: 'doc-unpublished',
      filename: 'unpublished.md',
      file_type: 'md',
      file_size: 512,
      status: 'ready',
      chunk_count: 0,
      published_generation: 0,
      uploaded_at: '2026-07-30T10:51:00Z'
    }
  ];
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
              ? initialDocuments
              : [
                  {
                    ...initialDocuments[0],
                    status: 'failed',
                    published_generation: 3
                  },
                  initialDocuments[1]
                ]
        })
      );
      return;
    }
    if (path === '/api/documents/doc-published/build' && request.method() === 'POST') {
      buildRequests += 1;
      if (buildRequests === 1) {
        await route.fulfill(jsonResponse({ message: 'queue unavailable' }, 503));
        return;
      }
      await route.fulfill(
        jsonResponse({
          job_id: 'job-rebuild',
          document_id: 'doc-published',
          status: 'queued',
          stage: 'queued',
          progress: 0,
          message: 'queued for rebuild',
          updated_at: '2026-07-30T10:52:00Z'
        })
      );
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await authenticateAdmin(page);
  await page.goto(`${baseUrl}documents`);
  await page.getByRole('heading', { name: '文档库' }).waitFor();
  assert.equal(await page.getByRole('button', { name: '查看文档 doc-unpublished 的已发布分块' }).count(), 0);

  await page.getByRole('button', { name: '重新构建文档 doc-published' }).click();
  await page.getByRole('button', { name: '创建重建任务' }).click();
  await page.getByRole('alert').waitFor();
  assert.equal(await page.getByRole('alert').innerText(), '创建重建任务失败，请稍后重试。');
  assert.equal(await page.getByRole('button', { name: '查看文档 doc-published 的已发布分块' }).isVisible(), true);
  await page.getByRole('button', { name: '取消', exact: true }).click();

  await page.getByRole('button', { name: '重新构建文档 doc-published' }).click();
  await page.getByRole('button', { name: '创建重建任务' }).click();
  await page.getByText('已创建重建任务 job-rebuild。').waitFor();
  await page.getByRole('button', { name: '刷新' }).click();
  await page.getByText('最近一次重建失败；当前已发布版本的 12 个分块仍可用于检索。候选分块未发布。').waitFor();
  assert.equal(await page.getByText('构建失败 (failed)').isVisible(), true);
  assert.equal(await page.getByRole('cell', { name: 'published-handbook.md' }).isVisible(), true);
});

test('System Administrator can rebuild with every supported strategy while the published generation remains available', { timeout: 30000 }, async (t) => {
  const buildRequests = [];
  const rebuildDocuments = ['general', 'paper', 'qa'].map((strategy) => ({
    document_id: `doc-${strategy}`,
    filename: `rebuild-${strategy}.md`,
    file_type: 'md',
    file_size: 1024,
    status: 'ready',
    chunk_count: 12,
    published_generation: 3,
    uploaded_at: '2026-07-30T11:00:00Z'
  }));
  const { page, baseUrl } = await startDocumentLibrary(t, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;

    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'operator', role: 'admin' }));
      return;
    }
    if (path === '/api/documents' && request.method() === 'GET') {
      await route.fulfill(jsonResponse({ items: rebuildDocuments }));
      return;
    }
    if (path.match(/^\/api\/documents\/doc-(general|paper|qa)\/build$/) && request.method() === 'POST') {
      const documentId = path.split('/')[3];
      const payload = request.postDataJSON();
      buildRequests.push({ documentId, payload });
      await route.fulfill(
        jsonResponse({
          job_id: `job-${payload.chunk_strategy}`,
          document_id: documentId,
          status: 'queued',
          stage: 'queued',
          progress: 0,
          message: 'queued for rebuild',
          updated_at: '2026-07-30T11:01:00Z'
        })
      );
      return;
    }
    if (path === '/api/documents/jobs' && request.method() === 'GET') {
      await route.fulfill(
        jsonResponse({
          items: [
            {
              job_id: 'job-qa',
              document_id: 'doc-qa',
              status: 'queued',
              stage: 'queued',
              progress: 0,
              message: 'queued for rebuild',
              updated_at: '2026-07-30T11:01:00Z'
            }
          ]
        })
      );
      return;
    }
    if (path === '/api/documents/jobs/job-qa' && request.method() === 'GET') {
      await route.fulfill(
        jsonResponse({
          job_id: 'job-qa',
          document_id: 'doc-qa',
          status: 'queued',
          stage: 'queued',
          progress: 0,
          message: 'queued for rebuild',
          updated_at: '2026-07-30T11:01:00Z'
        })
      );
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await authenticateAdmin(page);
  await page.goto(`${baseUrl}documents`);
  await page.getByRole('heading', { name: '文档库' }).waitFor();

  for (const strategy of ['general', 'paper', 'qa']) {
    await page.getByRole('button', { name: `重新构建文档 doc-${strategy}` }).click();
    await page.getByRole('heading', { name: '重新构建文档' }).waitFor();
    assert.deepEqual(
      await page.getByLabel('重建分块策略').locator('option').evaluateAll((options) => options.map((option) => option.value)),
      ['general', 'paper', 'qa']
    );
    await page.getByLabel('重建分块策略').selectOption(strategy);
    await page.getByRole('button', { name: '创建重建任务' }).click();
    await page.getByText(`已创建重建任务 job-${strategy}。`).waitFor();
  }

  assert.equal(await page.getByText('当前已发布版本的 12 个分块仍可用于检索；候选分块尚未发布。').count(), 3);
  assert.deepEqual(buildRequests, [
    { documentId: 'doc-general', payload: { chunk_strategy: 'general' } },
    { documentId: 'doc-paper', payload: { chunk_strategy: 'paper' } },
    { documentId: 'doc-qa', payload: { chunk_strategy: 'qa' } }
  ]);

  await page.getByRole('button', { name: '查看构建任务 job-qa' }).click();
  await page.waitForURL(/\/jobs\?job=job-qa$/);
  await page.getByText('正在查看任务 job-qa。').waitFor();
});

test('System Administrator confirms document deletion and keeps a document visible when deletion fails', { timeout: 30000 }, async (t) => {
  let releaseDeleteRequested;
  const releaseDeleteRequest = new Promise((resolve) => {
    releaseDeleteRequested = resolve;
  });
  let releaseDeleteCompleted;
  const releaseDeleteResponse = new Promise((resolve) => {
    releaseDeleteCompleted = resolve;
  });
  const deleteRequests = [];
  const removableDocuments = [
    {
      document_id: 'doc-release',
      filename: 'release-notes.md',
      file_type: 'md',
      file_size: 1024,
      status: 'ready',
      chunk_count: 8,
      uploaded_at: '2026-07-30T12:00:00Z'
    },
    {
      document_id: 'doc-protected',
      filename: 'protected-notes.md',
      file_type: 'md',
      file_size: 1024,
      status: 'failed',
      chunk_count: 0,
      uploaded_at: '2026-07-30T12:01:00Z'
    }
  ];
  const { page, baseUrl } = await startDocumentLibrary(t, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;

    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'operator', role: 'admin' }));
      return;
    }
    if (path === '/api/documents' && request.method() === 'GET') {
      await route.fulfill(jsonResponse({ items: removableDocuments }));
      return;
    }
    if (path === '/api/documents/release-notes.md' && request.method() === 'DELETE') {
      deleteRequests.push(`${request.method()} ${path}`);
      releaseDeleteRequested();
      await releaseDeleteResponse;
      await route.fulfill(jsonResponse({ success_ids: ['doc-release'], failed_items: [] }));
      return;
    }
    if (path === '/api/documents/protected-notes.md' && request.method() === 'DELETE') {
      deleteRequests.push(`${request.method()} ${path}`);
      await route.fulfill(jsonResponse({ message: 'document is being retained for investigation' }, 503));
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await authenticateAdmin(page);
  await page.goto(`${baseUrl}documents`);
  await page.getByRole('heading', { name: '文档库' }).waitFor();

  await page.getByRole('button', { name: '删除文档 doc-release' }).click();
  await page.getByRole('dialog').getByRole('button', { name: '删除', exact: true }).click();
  await releaseDeleteRequest;
  assert.equal(await page.getByRole('cell', { name: 'release-notes.md' }).isVisible(), true);
  releaseDeleteCompleted();
  await page.getByText('文档 release-notes.md 已由服务端确认删除。').waitFor();
  assert.equal(await page.getByRole('cell', { name: 'release-notes.md' }).count(), 0);

  await page.getByRole('button', { name: '删除文档 doc-protected' }).click();
  await page.getByRole('dialog').getByRole('button', { name: '删除', exact: true }).click();
  await page.getByRole('alert').waitFor();
  assert.equal(await page.getByRole('alert').innerText(), '删除文档 protected-notes.md 失败：请稍后重试。');
  assert.equal(await page.getByRole('cell', { name: 'protected-notes.md' }).isVisible(), true);
  assert.equal(await page.getByRole('button', { name: '删除文档 doc-protected' }).isVisible(), true);
  assert.deepEqual(deleteRequests, [
    'DELETE /api/documents/release-notes.md',
    'DELETE /api/documents/protected-notes.md'
  ]);
});

test('batch selection survives filtering and removes only records absent from a confirmed refresh', { timeout: 30000 }, async (t) => {
  let inventory = [documents[0], documents[2], documents[3]];
  const { page, baseUrl } = await startDocumentLibrary(t, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;

    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'operator', role: 'admin' }));
      return;
    }
    if (path === '/api/documents' && request.method() === 'GET') {
      await route.fulfill(jsonResponse({ items: inventory }));
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await authenticateAdmin(page);
  await page.goto(`${baseUrl}documents`);
  await page.getByRole('heading', { name: '文档库' }).waitFor();

  assert.equal(await page.getByRole('button', { name: '批量重新构建' }).isDisabled(), true);
  assert.equal(await page.getByRole('button', { name: '批量删除' }).isDisabled(), true);

  await page.getByLabel('选中文档 doc-ready').check();
  await page.getByText('已选择 1 个文档').waitFor();
  assert.equal(await page.getByRole('button', { name: '批量重新构建' }).isDisabled(), false);

  await page.getByLabel('按状态筛选').selectOption('ready');
  assert.equal(await page.getByLabel('选中文档 doc-ready').isChecked(), true);
  assert.equal(await page.getByText('已选择 1 个文档').isVisible(), true);

  await page.getByLabel('按状态筛选').selectOption('all');
  await page.getByLabel('选中文档 doc-failed').check();
  await page.getByText('已选择 2 个文档').waitFor();

  inventory = [documents[0], documents[3]];
  await page.getByRole('button', { name: '刷新' }).click();
  await page.getByText('已选择 1 个文档').waitFor();
  assert.equal(await page.getByLabel('选中文档 doc-ready').count(), 0);
  assert.equal(await page.getByLabel('选中文档 doc-failed').isChecked(), true);
});

test('a pending refresh blocks batch actions until stale document selection is reconciled', { timeout: 30000 }, async (t) => {
  let listReads = 0;
  let releaseRefresh;
  const refreshResponse = new Promise((resolve) => {
    releaseRefresh = resolve;
  });
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
        await route.fulfill(jsonResponse({ items: [documents[2]] }));
        return;
      }
      await refreshResponse;
      await route.fulfill(jsonResponse({ items: [] }));
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await authenticateAdmin(page);
  await page.goto(`${baseUrl}documents`);
  await page.getByRole('heading', { name: '文档库' }).waitFor();
  await page.getByLabel('选中文档 doc-ready').check();
  await page.getByRole('button', { name: '刷新' }).click();
  await page.getByText('正在刷新').waitFor();
  assert.equal(await page.getByRole('button', { name: '批量重新构建' }).isDisabled(), true);
  assert.equal(await page.getByRole('button', { name: '批量删除' }).isDisabled(), true);
  assert.equal(await page.getByLabel('选中文档 doc-ready').isDisabled(), true);

  releaseRefresh();
  await page.getByText('当前文档库为空。').waitFor();
  assert.equal(await page.getByText('已选择 0 个文档').isVisible(), true);
});

test('batch rebuild submits only supported strategies, keeps published generations available, and hands off to Indexing Jobs', { timeout: 30000 }, async (t) => {
  const rebuildRequests = [];
  const batchDocuments = [
    {
      document_id: 'doc-batch-first',
      filename: 'published-first.md',
      file_type: 'md',
      file_size: 1024,
      status: 'ready',
      chunk_count: 8,
      published_generation: 2,
      uploaded_at: '2026-07-30T13:00:00Z'
    },
    {
      document_id: 'doc-batch-second',
      filename: 'published-second.md',
      file_type: 'md',
      file_size: 2048,
      status: 'ready',
      chunk_count: 12,
      published_generation: 4,
      uploaded_at: '2026-07-30T13:01:00Z'
    }
  ];
  const { page, baseUrl } = await startDocumentLibrary(t, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;

    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'operator', role: 'admin' }));
      return;
    }
    if (path === '/api/documents' && request.method() === 'GET') {
      await route.fulfill(jsonResponse({ items: batchDocuments }));
      return;
    }
    if (path === '/api/documents/batch-build' && request.method() === 'POST') {
      const payload = request.postDataJSON();
      rebuildRequests.push(payload);
      await route.fulfill(
        jsonResponse({
          items: payload.document_ids.map((documentId) => ({
            job_id: `job-${documentId}`,
            document_id: documentId,
            status: 'queued',
            stage: 'queued',
            progress: 0,
            message: 'queued for rebuild',
            updated_at: '2026-07-30T13:02:00Z'
          }))
        })
      );
      return;
    }
    if (path === '/api/documents/jobs' && request.method() === 'GET') {
      await route.fulfill(jsonResponse({ items: [] }));
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await authenticateAdmin(page);
  await page.goto(`${baseUrl}documents`);
  await page.getByRole('heading', { name: '文档库' }).waitFor();
  const batchControlsBox = await page.getByLabel('批量文档操作').boundingBox();

  await page.getByLabel('选中文档 doc-batch-first').check();
  await page.getByLabel('选中文档 doc-batch-second').check();
  await page.getByRole('button', { name: '批量重新构建' }).click();
  await page.getByRole('heading', { name: '重新构建 2 个文档' }).waitFor();
  assert.deepEqual(
    await page.getByLabel('批量重建分块策略').locator('option').evaluateAll((options) => options.map((option) => option.value)),
    ['general', 'paper', 'qa']
  );
  assert.equal(
    await page.getByText('其中 2 个文档的当前已发布版本将在新任务运行时继续用于检索；候选分块尚未发布。').isVisible(),
    true
  );

  await page.getByLabel('批量重建分块策略').selectOption('qa');
  await page.getByRole('button', { name: '创建 2 个重建任务' }).click();
  await page.getByText('已为 2 个文档创建重建任务。').waitFor();
  assert.deepEqual(rebuildRequests, [
    { document_ids: ['doc-batch-first', 'doc-batch-second'], chunk_strategy: 'qa' }
  ]);
  assert.equal(await page.getByText('已创建重建任务 job-doc-batch-first。').isVisible(), true);
  assert.equal(await page.getByText('已创建重建任务 job-doc-batch-second。').isVisible(), true);
  assert.equal(await page.getByText('当前已发布版本的 8 个分块仍可用于检索；候选分块尚未发布。').isVisible(), true);
  assert.equal(await page.getByText('当前已发布版本的 12 个分块仍可用于检索；候选分块尚未发布。').isVisible(), true);

  const resultControlsBox = await page.getByLabel('批量文档操作').boundingBox();
  assert.equal(resultControlsBox.width, batchControlsBox.width);
  assert.equal(resultControlsBox.height, batchControlsBox.height);

  await page.getByRole('button', { name: '前往构建任务' }).click();
  await page.waitForURL(/\/jobs$/);
  await page.getByRole('heading', { name: '构建任务' }).waitFor();
});

test('batch deletion confirms the selected count, reports partial failures, and retains inventory on request failure', { timeout: 30000 }, async (t) => {
  let releaseFirstDeleteRequested;
  const firstDeleteRequested = new Promise((resolve) => {
    releaseFirstDeleteRequested = resolve;
  });
  let releaseFirstDeleteResponse;
  const firstDeleteResponse = new Promise((resolve) => {
    releaseFirstDeleteResponse = resolve;
  });
  const batchDeleteRequests = [];
  const batchDocuments = [
    {
      document_id: 'doc-batch-remove',
      filename: 'remove-after-confirmation.md',
      file_type: 'md',
      file_size: 1024,
      status: 'ready',
      chunk_count: 8,
      uploaded_at: '2026-07-30T14:00:00Z'
    },
    {
      document_id: 'doc-batch-retain',
      filename: 'retain-after-partial-failure.md',
      file_type: 'md',
      file_size: 1024,
      status: 'failed',
      chunk_count: 0,
      uploaded_at: '2026-07-30T14:01:00Z'
    },
    {
      document_id: 'doc-batch-request-failure',
      filename: 'retain-after-request-failure.md',
      file_type: 'md',
      file_size: 1024,
      status: 'ready',
      chunk_count: 2,
      uploaded_at: '2026-07-30T14:02:00Z'
    }
  ];
  const { page, baseUrl } = await startDocumentLibrary(t, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;

    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'operator', role: 'admin' }));
      return;
    }
    if (path === '/api/documents' && request.method() === 'GET') {
      await route.fulfill(jsonResponse({ items: batchDocuments }));
      return;
    }
    if (path === '/api/documents/batch-delete' && request.method() === 'POST') {
      batchDeleteRequests.push(request.postDataJSON());
      if (batchDeleteRequests.length === 1) {
        releaseFirstDeleteRequested();
        await firstDeleteResponse;
        await route.fulfill(
          jsonResponse({
            success_ids: ['doc-batch-remove'],
            failed_items: [{ document_id: 'doc-batch-retain', message: '文档正在保留以供调查。' }]
          })
        );
        return;
      }
      await route.fulfill(jsonResponse({ message: 'document service unavailable' }, 503));
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await authenticateAdmin(page);
  await page.goto(`${baseUrl}documents`);
  await page.getByRole('heading', { name: '文档库' }).waitFor();

  await page.getByLabel('选中文档 doc-batch-remove').check();
  await page.getByLabel('选中文档 doc-batch-retain').check();
  await page.getByRole('button', { name: '批量删除' }).click();
  await page.getByRole('dialog').getByText('确认删除已选择的 2 个文档？此操作不能撤销。').waitFor();
  await page.getByRole('dialog').getByRole('button', { name: '删除', exact: true }).click();
  await firstDeleteRequested;
  assert.equal(await page.getByRole('button', { name: '批量删除' }).isDisabled(), true);
  assert.equal(await page.getByRole('cell', { name: 'remove-after-confirmation.md' }).isVisible(), true);

  releaseFirstDeleteResponse();
  await page.getByText('已由服务端确认删除 1 个文档。').waitFor();
  assert.equal(await page.getByRole('cell', { name: 'remove-after-confirmation.md' }).count(), 0);
  assert.equal(await page.getByRole('cell', { name: 'retain-after-partial-failure.md' }).isVisible(), true);
  assert.equal(await page.getByText('文档 retain-after-partial-failure.md：文档正在保留以供调查。').isVisible(), true);
  assert.equal(await page.getByText('已选择 1 个文档').isVisible(), true);
  assert.equal(await page.getByLabel('选中文档 doc-batch-retain').isChecked(), true);

  await page.getByLabel('选中文档 doc-batch-request-failure').check();
  await page.getByRole('button', { name: '批量删除' }).click();
  await page.getByRole('dialog').getByRole('button', { name: '删除', exact: true }).click();
  await page.getByRole('alert').waitFor();
  assert.equal(await page.getByRole('alert').innerText(), '批量删除失败：请稍后重试。');
  assert.equal(await page.getByRole('cell', { name: 'retain-after-partial-failure.md' }).isVisible(), true);
  assert.equal(await page.getByRole('cell', { name: 'retain-after-request-failure.md' }).isVisible(), true);
  assert.equal(await page.getByText('已选择 2 个文档').isVisible(), true);
  assert.deepEqual(batchDeleteRequests, [
    { document_ids: ['doc-batch-remove', 'doc-batch-retain'] },
    { document_ids: ['doc-batch-retain', 'doc-batch-request-failure'] }
  ]);
});
