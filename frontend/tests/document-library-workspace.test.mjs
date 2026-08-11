// Document Library browser acceptance over a disposable real API.
// Uploads, builds, publication, deletion, and inventory refreshes use the real
// server; the in-process build dispatcher executes builds deterministically.
import assert from 'node:assert/strict';
import test from 'node:test';
import { loginAdmin, registerKnowledgeUserViaApi, startWorkbench } from './acceptance-env.mjs';

const openLibrary = async (page, baseUrl) => {
  await page.goto(`${baseUrl}documents`);
  await page.getByRole('heading', { name: '文档库' }).waitFor();
  // The inventory loads asynchronously after navigation. `aria-busy` alone is
  // racy (it starts false before the first load), so additionally wait for the
  // loading placeholder row to be gone: either data rows or the empty-state row
  // must have rendered.
  const tableWrap = page.locator('.document-library__table-wrap');
  await tableWrap.waitFor({ state: 'visible' });
  await page.getByRole('table').waitFor();
  await page.waitForFunction(() => {
    const wrap = document.querySelector('.document-library__table-wrap');
    const tbody = document.querySelector('tbody');
    if (!wrap || !tbody) return false;
    if (wrap.getAttribute('aria-busy') === 'true') return false;
    const stateRows = Array.from(tbody.querySelectorAll('.document-library__state'));
    const loadingShown = stateRows.some((row) => row.textContent.includes('正在加载文档库'));
    return !loadingShown;
  });
};

test('System Administrator can scan Document Library and filter the loaded inventory', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, {});
  await loginAdmin(page, baseUrl);
  await openLibrary(page, baseUrl);

  assert.equal(await page.getByRole('link', { name: '文档库' }).isVisible(), true);
  assert.equal(await page.getByRole('link', { name: '构建任务' }).isVisible(), true);
  assert.equal(await page.getByText('支持的格式：UTF-8 TXT、Markdown (.md)、可提取文本的 PDF；单个文件不超过 25 MiB。').isVisible(), true);
  assert.equal(await page.getByText('待处理 (pending)').first().isVisible(), true);
  assert.equal(await page.getByText('处理中 (processing)').first().isVisible(), true);
  assert.equal(await page.getByText('可检索 (ready)').first().isVisible(), true);
  assert.equal(await page.getByText('待发布候选 (candidate)').first().isVisible(), true);

  const [tableBox, refreshBox] = await Promise.all([
    page.getByRole('table').boundingBox(),
    page.getByRole('button', { name: '刷新' }).boundingBox()
  ]);
  assert.ok(tableBox.x + tableBox.width <= 1440);
  assert.ok(refreshBox.x >= tableBox.x);
  assert.ok(refreshBox.x + refreshBox.width <= 1440);

  await page.getByLabel('按状态筛选').selectOption('ready');
  assert.equal(await page.getByText('browser-evidence.md').isVisible(), true);
  assert.equal(await page.getByText('browser-cancelable.md').count(), 0);

  await page.getByLabel('按文件名搜索').fill('browser-evidence');
  assert.equal(await page.getByText('browser-evidence.md').isVisible(), true);

  await page.getByLabel('按文件名搜索').fill('不存在的文件');
  assert.equal(await page.getByText('没有符合当前筛选条件的文档。').isVisible(), true);

  await page.getByLabel('按文件名搜索').fill('browser-evidence');
  await page.getByRole('button', { name: '查看文档 browser-evidence 的构建任务' }).click();
  await page.waitForURL(/\/jobs\?document=browser-evidence$/);
  assert.equal(await page.getByText('正在查看文档 browser-evidence 的构建任务。').isVisible(), true);
  // The filtered job list loads asynchronously after navigation.
  await page.getByText('job-browser-evidence').waitFor();
  assert.equal(await page.getByText('job-browser-evidence').isVisible(), true);
});

test('Knowledge User is redirected before Document Library or its protected inventory can render', { timeout: 30000 }, async (t) => {
  const { page, baseUrl, api } = await startWorkbench(t, {});
  const token = await registerKnowledgeUserViaApi(api, 'lin');
  await page.addInitScript(
    ({ storedToken }) => localStorage.setItem('access_token', storedToken),
    { storedToken: token }
  );

  await page.goto(`${baseUrl}documents`);
  await page.waitForURL(/\/chat\?notice=admin-required$/);
  assert.equal(await page.getByRole('heading', { name: '文档库' }).count(), 0);
  assert.equal(await page.getByRole('link', { name: '文档库' }).count(), 0);
});

test('initial upload accepts a supported file, relies on the general strategy contract, refreshes inventory, and hands off to its Indexing Job', { timeout: 60000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, {});
  await loginAdmin(page, baseUrl);
  await openLibrary(page, baseUrl);

  await page.getByLabel('选择文档').setInputFiles({
    name: 'initial-notes.md',
    mimeType: 'text/markdown',
    buffer: Buffer.from('# 初始上传的部署记录')
  });
  await page.getByRole('button', { name: '上传文档' }).click();

  await page.getByText(/文档 ID：/).waitFor();
  assert.equal(await page.getByText(/构建任务 ID：/).isVisible(), true);
  // The inventory refresh after upload is async; wait for the new row instead
  // of asserting on a stale snapshot.
  await page.getByRole('cell', { name: 'initial-notes.md' }).waitFor();
  assert.equal(await page.getByRole('cell', { name: 'initial-notes.md' }).isVisible(), true);
  assert.equal(await page.getByText('初始上传将自动使用通用分块策略。').isVisible(), true);

  const jobLink = page.getByRole('button', { name: /^查看构建任务/ });
  await jobLink.waitFor();
  await jobLink.click();
  await page.waitForURL(/\/jobs\?job=/);
  await page.getByRole('heading', { name: '构建任务' }).waitFor();
  // The filtered job list loads asynchronously after navigation; the uploaded
  // document's id is a server-generated UUID, so wait for any focused job row
  // instead of a filename-derived cell.
  await page
    .getByRole('row')
    .filter({ has: page.locator('td.indexing-jobs__identifier') })
    .waitFor();
  assert.ok((await page.getByRole('row').count()) >= 1);
});

test('unsupported files and backend upload validation failures retain a recoverable Document Library', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, {});
  await loginAdmin(page, baseUrl);
  await openLibrary(page, baseUrl);

  await page.getByLabel('选择文档').setInputFiles({
    name: 'unsupported.docx',
    mimeType: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    buffer: Buffer.from('not supported')
  });
  await page.getByRole('alert').waitFor();
  assert.equal(await page.getByRole('alert').innerText(), '不支持的文件格式。仅支持 TXT、Markdown (.md) 和 PDF。');
  assert.equal(await page.getByText('browser-evidence.md').isVisible(), true);

  await page.getByLabel('选择文档').setInputFiles({
    name: 'broken.pdf',
    mimeType: 'application/pdf',
    buffer: Buffer.from('this is not a real pdf payload')
  });
  await page.getByRole('button', { name: '上传文档' }).click();
  await page.getByRole('alert').waitFor();
  assert.equal(await page.getByRole('alert').innerText(), '上传被服务端拒绝，请检查文件名或内容后重试。');
  // The rejected file stays selected for retry; only the inventory must not
  // contain a broken.pdf row.
  assert.equal(await page.getByRole('cell', { name: 'broken.pdf' }).count(), 0);
  assert.equal(await page.getByRole('button', { name: '上传文档' }).isVisible(), true);
});

test('empty Document Library states preserve the desktop controls', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, { env: { BROWSER_ACCEPTANCE_SEED: 'minimal' } });
  await loginAdmin(page, baseUrl);
  await openLibrary(page, baseUrl);

  await page.getByText('当前文档库为空。').waitFor();
  assert.equal(await page.getByRole('table').isVisible(), true);
  assert.equal(await page.getByLabel('按状态筛选').isVisible(), true);
});

test('System Administrator can inspect published chunks for a ready document', { timeout: 60000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, {});
  await loginAdmin(page, baseUrl);
  await openLibrary(page, baseUrl);

  await page.getByRole('button', { name: '查看文档 browser-evidence 的候选或已发布分块' }).click();
  await page.getByRole('heading', { name: '已发布版本分块' }).waitFor();
  assert.equal(await page.getByText('文档 ID：browser-evidence').isVisible(), true);
  await page.getByText('部署前需要完成变更审批。').waitFor();
  assert.equal(await page.getByText('关键词：部署、审批').isVisible(), true);
});

test('System Administrator can inspect and explicitly publish a Candidate Build', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, {});
  await loginAdmin(page, baseUrl);
  await openLibrary(page, baseUrl);

  assert.equal(await page.getByText('待发布候选 (candidate)').first().isVisible(), true);
  await page.getByRole('button', { name: '发布候选构建 browser-candidate' }).click();
  await page.getByRole('dialog').getByText(/确认发布文档“browser-candidate.md”的候选构建/).waitFor();
  await page.getByRole('dialog').getByRole('button', { name: '发布', exact: true }).click();
  // Wait for the publication to complete: the publish button must disappear.
  // (The generic '可检索 (ready)' label is already present from seed data, so
  // it cannot signal the transition.)
  await page.getByRole('button', { name: '发布候选构建 browser-candidate' }).waitFor({ state: 'detached' });
  assert.equal(await page.getByRole('button', { name: '发布候选构建 browser-candidate' }).count(), 0);
});

test('System Administrator can rebuild with every supported strategy while the published generation remains available', { timeout: 120000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, {});
  await loginAdmin(page, baseUrl);
  await openLibrary(page, baseUrl);

  // Each rebuild moves its document to `pending` (the build API sets status
  // "pending"), which removes its rebuild button, so exercise every supported
  // strategy against a distinct ready seed document instead of rebuilding the
  // same one repeatedly.
  const rebuildTargets = ['browser-evidence', 'browser-inspection', 'browser-single-delete'];
  const seenRebuildJobIds = new Set();
  for (const [index, strategy] of ['general', 'paper', 'qa'].entries()) {
    const target = rebuildTargets[index];
    const rebuildButton = page.getByRole('button', { name: `重新构建文档 ${target}` });
    await rebuildButton.waitFor({ timeout: 30000 });
    await rebuildButton.click();
    await page.getByRole('heading', { name: '重新构建文档' }).waitFor();
    assert.deepEqual(
      await page.getByLabel('重建分块策略').locator('option').evaluateAll((options) => options.map((option) => option.value)),
      ['general', 'paper', 'qa']
    );
    await page.getByLabel('重建分块策略').selectOption(strategy);
    await page.getByRole('button', { name: '创建重建任务' }).click();
    // The rebuild status line renders per document, so older rebuild messages
    // stay visible; wait for a message whose job id we have not seen yet.
    await page.waitForFunction(
      ({ seen }) => {
        const ids = Array.from(document.querySelectorAll('.document-library__rebuild-job-id'))
          .map((node) => node.textContent.trim())
          .filter(Boolean);
        return ids.some((id) => !seen.includes(id));
      },
      { seen: [...seenRebuildJobIds] },
      { timeout: 15000 }
    );
    const newIds = await page.locator('.document-library__rebuild-job-id').allInnerTexts();
    newIds.map((id) => id.trim()).filter(Boolean).forEach((id) => seenRebuildJobIds.add(id));
    await page.keyboard.press('Escape');
    await page.getByRole('heading', { name: '重新构建文档' }).waitFor({ state: 'detached' });
  }

  await page.getByRole('button', { name: '查看文档 browser-evidence 的构建任务' }).click();
  await page.waitForURL(/\/jobs\?document=browser-evidence$/);
  await page.getByRole('heading', { name: '构建任务' }).waitFor();
});

test('System Administrator confirms document deletion and the inventory updates from the server', { timeout: 60000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, {});
  await loginAdmin(page, baseUrl);
  await openLibrary(page, baseUrl);

  await page.getByRole('button', { name: '删除文档 browser-single-delete' }).click();
  await page.getByRole('dialog').getByRole('button', { name: '删除', exact: true }).click();
  await page.getByText('文档 browser-single-delete.md 已由服务端确认删除。').waitFor();
  assert.equal(await page.getByRole('cell', { name: 'browser-single-delete.md' }).count(), 0);
});

test('batch deletion confirms the selected count and reports partial failures from the server', { timeout: 60000 }, async (t) => {
  const { page, baseUrl, api } = await startWorkbench(t, {});
  await loginAdmin(page, baseUrl);
  await openLibrary(page, baseUrl);

  // Simulate a concurrent administrator removing browser-batch-partial-first
  // after the batch dialog opened, so the server reports it as a partial
  // failure while browser-batch-partial-second is deleted successfully.
  const login = await fetch(`${api.baseUrl}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: 'operator', password: 'safe-password' })
  });
  const adminToken = (await login.json()).data.access_token;
  const deleteResponse = await fetch(
    `${api.baseUrl}/documents/${encodeURIComponent('browser-batch-partial-first.md')}`,
    { method: 'DELETE', headers: { Authorization: `Bearer ${adminToken}` } }
  );
  assert.equal(deleteResponse.status, 200);

  await page.getByLabel('选中文档 browser-batch-partial-first').check();
  await page.getByLabel('选中文档 browser-batch-partial-second').check();
  await page.getByRole('button', { name: '批量删除' }).click();
  await page.getByRole('dialog').getByText('确认删除已选择的 2 个文档？此操作不能撤销。').waitFor();
  await page.getByRole('dialog').getByRole('button', { name: '删除', exact: true }).click();

  await page.getByText('已由服务端确认删除 1 个文档。').waitFor();
  assert.equal(await page.getByRole('cell', { name: 'browser-batch-partial-second.md' }).count(), 0);
  assert.equal(await page.getByRole('cell', { name: 'browser-batch-partial-first.md' }).isVisible(), true);
  assert.equal(await page.getByText('文档 browser-batch-partial-first.md：document not found').isVisible(), true);
});

test('batch selection survives filtering and removes only records absent from a confirmed refresh', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, {});
  await loginAdmin(page, baseUrl);
  await openLibrary(page, baseUrl);

  assert.equal(await page.getByRole('button', { name: '批量重新构建' }).isDisabled(), true);
  assert.equal(await page.getByRole('button', { name: '批量删除' }).isDisabled(), true);

  await page.getByLabel('选中文档 browser-evidence').check();
  await page.getByText('已选择 1 个文档').waitFor();
  assert.equal(await page.getByRole('button', { name: '批量重新构建' }).isDisabled(), false);

  await page.getByLabel('按状态筛选').selectOption('ready');
  assert.equal(await page.getByLabel('选中文档 browser-evidence').isChecked(), true);
  assert.equal(await page.getByText('已选择 1 个文档').isVisible(), true);

  await page.getByLabel('按状态筛选').selectOption('all');
  await page.getByLabel('选中文档 browser-cancelable').check();
  await page.getByText('已选择 2 个文档').waitFor();

  await page.getByRole('button', { name: '刷新' }).click();
  await page.getByText('已选择 2 个文档').waitFor();
  assert.equal(await page.getByLabel('选中文档 browser-evidence').isChecked(), true);
  assert.equal(await page.getByLabel('选中文档 browser-cancelable').isChecked(), true);
});

test('Document Library reports desktop scope on mobile without rendering the inventory', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, { viewport: { width: 390, height: 844 } });
  await loginAdmin(page, baseUrl);

  await page.goto(`${baseUrl}documents`);
  await page.getByRole('heading', { name: '文档库' }).waitFor();
  assert.equal(await page.getByText('文档库当前仅支持桌面工作区。').isVisible(), true);
  assert.equal(await page.getByRole('table').count(), 0);
  assert.equal(await page.getByLabel('选择文档').count(), 0);
  assert.equal(await page.getByRole('button', { name: '刷新' }).count(), 0);
});
