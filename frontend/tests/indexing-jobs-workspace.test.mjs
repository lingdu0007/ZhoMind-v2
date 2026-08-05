// Indexing Jobs workspace browser acceptance over a disposable real API.
// Job lifecycle states and cancellation come from the real server.
import assert from 'node:assert/strict';
import test from 'node:test';
import { loginAdmin, registerKnowledgeUserViaApi, startWorkbench } from './acceptance-env.mjs';

const openJobsWorkspace = async (page, baseUrl) => {
  await page.goto(`${baseUrl}jobs`);
  await page.getByRole('heading', { name: '构建任务' }).waitFor();
  // Jobs load asynchronously after navigation. `aria-busy` alone is racy (it
  // starts false before the first load), so additionally wait for the loading
  // placeholder row to be gone.
  await page.locator('.indexing-jobs__table-wrap').waitFor({ state: 'visible' });
  await page.getByRole('table').waitFor();
  await page.waitForFunction(() => {
    const wrap = document.querySelector('.indexing-jobs__table-wrap');
    const tbody = document.querySelector('tbody');
    if (!wrap || !tbody) return false;
    if (wrap.getAttribute('aria-busy') === 'true') return false;
    const stateRows = Array.from(tbody.querySelectorAll('.indexing-jobs__state'));
    const loadingShown = stateRows.some((row) => row.textContent.includes('正在加载构建任务'));
    return !loadingShown;
  });
};

test('System Administrator can open a separate Indexing Jobs workspace, filter lifecycle states, and cancel an eligible job', { timeout: 60000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, {});
  await loginAdmin(page, baseUrl);
  await openJobsWorkspace(page, baseUrl);

  assert.equal(await page.getByRole('link', { name: '构建任务' }).isVisible(), true);
  assert.equal(await page.getByRole('link', { name: '文档库' }).isVisible(), true);
  assert.equal(await page.getByText('job-browser-running').isVisible(), true);
  assert.equal(await page.getByText('browser-running', { exact: true }).isVisible(), true);
  assert.equal(await page.getByText('执行中 (running)').isVisible(), true);
  assert.equal(await page.getByText('分块中 (chunking)').isVisible(), true);
  assert.equal(await page.getByText('正在生成可检索分块').isVisible(), true);
  assert.equal(await page.getByRole('button', { name: '取消任务 job-browser-running' }).isVisible(), true);
  assert.equal(await page.getByRole('button', { name: '取消任务 job-browser-evidence' }).count(), 0);

  const [tableBox, cancelBox] = await Promise.all([
    page.getByRole('table').boundingBox(),
    page.getByRole('button', { name: '取消任务 job-browser-running' }).boundingBox()
  ]);
  assert.ok(tableBox.x + tableBox.width <= 1440);
  assert.ok(cancelBox.x >= tableBox.x);
  assert.ok(cancelBox.x + cancelBox.width <= tableBox.x + tableBox.width);

  await page.getByRole('radio', { name: '仅进行中' }).check();
  assert.equal(await page.getByText('job-browser-running').isVisible(), true);
  assert.equal(await page.getByText('job-browser-evidence').count(), 0);

  await page.getByRole('radio', { name: '仅已结束' }).check();
  assert.equal(await page.getByText('job-browser-running').count(), 0);
  assert.equal(await page.getByText('job-browser-evidence').isVisible(), true);
  assert.equal(await page.getByText('已成功 (succeeded)').first().isVisible(), true);

  await page.getByRole('radio', { name: '仅进行中' }).check();
  await page.getByRole('button', { name: '取消任务 job-browser-running' }).click();
  // Under the active filter the canceled job disappears immediately, so switch
  // to the terminal filter and wait for the confirmed canceled state.
  await page.getByRole('radio', { name: '仅已结束' }).check();
  await page.getByText('已取消 (canceled)').waitFor();
  assert.equal(await page.getByRole('button', { name: '取消任务 job-browser-running' }).count(), 0);
});

test('Knowledge User is redirected before the Indexing Jobs workspace or protected jobs endpoint can render', { timeout: 30000 }, async (t) => {
  const { page, baseUrl, api } = await startWorkbench(t, {});
  const token = await registerKnowledgeUserViaApi(api, 'lin');
  await page.addInitScript(
    ({ storedToken }) => localStorage.setItem('access_token', storedToken),
    { storedToken: token }
  );

  await page.goto(`${baseUrl}jobs`);
  await page.waitForURL(/\/chat\?notice=admin-required$/);
  assert.equal(await page.getByRole('heading', { name: '构建任务' }).count(), 0);
  assert.equal(await page.getByRole('link', { name: '构建任务' }).count(), 0);
});

test('active jobs poll until terminal, then stop polling when the workspace is left', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, {});
  await loginAdmin(page, baseUrl);
  await openJobsWorkspace(page, baseUrl);

  await page.getByText('执行中 (running)').waitFor();
  await page.getByRole('button', { name: '取消任务 job-browser-running' }).click();
  await page.getByText('已取消 (canceled)').waitFor({ timeout: 8000 });

  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();
  await page.waitForTimeout(450);
  assert.equal(await page.getByRole('heading', { name: '对话工作区' }).isVisible(), true);
});

test('empty lists preserve the stable desktop controls', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, { env: { BROWSER_ACCEPTANCE_SEED: 'minimal' } });
  await loginAdmin(page, baseUrl);
  await openJobsWorkspace(page, baseUrl);

  await page.getByText('当前没有构建任务。').waitFor();
  assert.equal(await page.getByRole('table').isVisible(), true);
  assert.equal(await page.getByRole('group', { name: '任务状态筛选' }).isVisible(), true);

  await page.getByRole('button', { name: '刷新' }).click();
  await page.getByText('当前没有构建任务。').waitFor();
  assert.equal(await page.getByRole('table').isVisible(), true);
});

test('Indexing Jobs keeps its desktop-only mobile scope explicit without rendering a dense table', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, { viewport: { width: 390, height: 844 } });
  await loginAdmin(page, baseUrl);

  await page.goto(`${baseUrl}jobs`);
  await page.getByRole('heading', { name: '构建任务' }).waitFor();
  assert.equal(await page.getByText('构建任务当前仅支持桌面工作区。').isVisible(), true);
  assert.equal(await page.getByRole('table').count(), 0);
  assert.equal(await page.getByRole('button', { name: '刷新' }).count(), 0);
  assert.equal(await page.getByRole('button', { name: '重新加载' }).count(), 0);
});
