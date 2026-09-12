import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { loginAdmin, registerKnowledgeUserViaApi, startWorkbench } from './acceptance-env.mjs';

const prepareQuestion = async (page) => {
  await page.getByPlaceholder('请输入需要检索的问题').fill('什么时候使用 deterministic workflow？');
  await page.getByRole('button', { name: '添加查询条件', exact: true }).click();
  await page.getByLabel('条件 1 字段', { exact: true }).fill('execution_path');
  await page.getByLabel('条件 1 值', { exact: true }).fill('known');
};

test('administrator operations exposes bounded admission without private content on desktop and narrow screens', async (t) => {
  const { page, baseUrl, api } = await startWorkbench(t);
  await registerKnowledgeUserViaApi(api, 'operations-member');
  await loginAdmin(page, baseUrl);
  await page.getByRole('link', { name: '运行状态', exact: true }).click();
  await page.getByRole('heading', { name: '运行状态', exact: true }).waitFor();
  await page.locator('[aria-label="请求准入"][aria-busy="false"]').waitFor();
  assert.equal(await page.getByTestId('admission-executing').innerText(), '0 / 2');
  assert.equal(await page.getByTestId('admission-queued').innerText(), '0 / 2');
  assert.equal(await page.getByPlaceholder('请输入需要检索的问题').count(), 0);
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 900 });
    await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    if (process.env.TICKET28_SCREENSHOT_DIR) {
      await mkdir(process.env.TICKET28_SCREENSHOT_DIR, { recursive: true });
      await page.screenshot({ path: join(process.env.TICKET28_SCREENSHOT_DIR, `operations-${width}.png`), fullPage: true });
    }
  }
  await page.getByRole('button', { name: '退出登录' }).click();
  await loginAdmin(page, baseUrl, { username: 'operations-member' });
  assert.equal(await page.getByRole('link', { name: '运行状态', exact: true }).count(), 0);
  await page.goto(`${baseUrl}operations`);
  await page.waitForURL(/\/chat\?notice=admin-required$/);
});

test('authenticated browser burst shows two executing, two queued and a retryable fifth request', async (t) => {
  const { page, baseUrl, api } = await startWorkbench(t, { env: { BROWSER_ACCEPTANCE_LLM_DELAY_MS: '6000' } });
  for (let index = 0; index < 4; index += 1) {
    await registerKnowledgeUserViaApi(api, `queue-browser-${index}`);
  }
  await loginAdmin(page, baseUrl);
  const token = await page.evaluate(() => localStorage.getItem('access_token'));
  const contexts = [];
  const errorCodes = [];
  t.after(async () => { await Promise.allSettled(contexts.map((context) => context.close())); });
  const pages = [];
  for (const member of [0, 1, 0, 2, 3]) {
    const context = await page.context().browser().newContext();
    contexts.push(context);
    const memberPage = await context.newPage();
    await memberPage.addInitScript(() => { Date.now = () => 1800000000000; });
    memberPage.on('response', async (response) => {
      if (!response.url().endsWith('/chat/stream')) return;
      try {
        const body = await response.text();
        for (const frame of body.split('\n\n')) {
          if (frame.startsWith('event: error\ndata: ')) {
            errorCodes.push(JSON.parse(frame.slice('event: error\ndata: '.length)).code);
          }
        }
      } catch {
        errorCodes.push('TEST_TRANSPORT_INCOMPLETE');
      }
    });
    memberPage.setDefaultTimeout(20000);
    await loginAdmin(memberPage, baseUrl, { username: `queue-browser-${member}` });
    pages.push(memberPage);
  }
  for (const target of pages) {
    await prepareQuestion(target);
  }
  const send = (target) => target.getByRole('button', { name: '发送', exact: true }).click();
  const counts = async (executing, queued) => {
    const deadline = Date.now() + 5000;
    let last;
    while (Date.now() < deadline) {
      const response = await fetch(`${api.baseUrl}/operations`, { headers: { Authorization: `Bearer ${token}` } });
      const { admission, events, failures } = (await response.json()).data;
      last = { admission, failures, states: events.map((event) => event.dimensions) };
      if (admission.executing === executing && admission.queued === queued) return;
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
    throw new Error(`expected admission state not observed: ${JSON.stringify({ ...last, errorCodes })}`);
  };
  await Promise.all(pages.slice(0, 2).map(send));
  await counts(2, 0);
  await Promise.all(pages.slice(2, 4).map(send));
  await counts(2, 2);
  for (const target of pages.slice(2, 4)) {
    await target.getByText(/Waiting in queue/).waitFor();
  }
  await send(pages[4]);
  await pages[4].locator('.answer-outcome__label').filter({ hasText: /^Throttled$/ }).waitFor();
  await pages[4].getByRole('button', { name: '重试', exact: true }).waitFor();
  for (const target of pages.slice(0, 4)) {
    await target.locator('.answer-outcome__label').filter({ hasText: /^Supported by published knowledge$/ }).waitFor();
  }
  await counts(0, 0);
});

test('queued browser request times out as failed and retains its retryable private history', async (t) => {
  const { page, baseUrl, api } = await startWorkbench(t, {
    env: { BROWSER_ACCEPTANCE_QUEUE_TIMEOUT_SECONDS: '0.5', BROWSER_ACCEPTANCE_LLM_DELAY_MS: '6000' }
  });
  await registerKnowledgeUserViaApi(api, 'timeout-owner');
  await loginAdmin(page, baseUrl, { username: 'timeout-owner' });
  const contexts = [];
  t.after(async () => { await Promise.allSettled(contexts.map((context) => context.close())); });
  const pages = [];
  for (let index = 0; index < 2; index += 1) {
    await registerKnowledgeUserViaApi(api, `timeout-busy-${index}`);
    const context = await page.context().browser().newContext();
    contexts.push(context);
    const target = await context.newPage();
    await loginAdmin(target, baseUrl, { username: `timeout-busy-${index}` });
    await prepareQuestion(target);
    pages.push(target);
  }
  await prepareQuestion(page);
  await Promise.all(pages.map((target) => target.getByRole('button', { name: '发送', exact: true }).click()));
  for (const target of pages) await target.getByRole('button', { name: '停止', exact: true }).waitFor();
  await page.getByRole('button', { name: '发送', exact: true }).click();
  await page.locator('.answer-outcome__label').filter({ hasText: /^Failed$/ }).waitFor();
  await page.getByRole('button', { name: '重试', exact: true }).waitFor();
  assert.equal(await page.getByText('Insufficient evidence', { exact: true }).count(), 0);
  const token = await page.evaluate(() => localStorage.getItem('access_token'));
  const sessions = await fetch(`${api.baseUrl}/sessions`, { headers: { Authorization: `Bearer ${token}` } });
  const sessionId = (await sessions.json()).data.sessions[0].id;
  const history = await fetch(`${api.baseUrl}/sessions/${sessionId}`, { headers: { Authorization: `Bearer ${token}` } });
  assert.ok((await history.json()).messages.every((item) => item.answer_execution.failure_code === 'CHAT_QUEUE_TIMEOUT'));
  for (const target of pages) {
    await target.locator('.answer-outcome__label').filter({ hasText: /^Supported by published knowledge$/ }).waitFor();
  }
});

test('provider timeout is generation unavailable while administrator sees only normalized route evidence', async (t) => {
  const { page, baseUrl, api } = await startWorkbench(t, { env: { BROWSER_ACCEPTANCE_PROVIDER_FAIL_ON_CALL: '1' } });
  await registerKnowledgeUserViaApi(api, 'provider-timeout-owner');
  await loginAdmin(page, baseUrl, { username: 'provider-timeout-owner' });
  await prepareQuestion(page);
  await page.getByRole('button', { name: '发送', exact: true }).click();
  await page.locator('.answer-outcome__label').filter({ hasText: /^Generation unavailable$/i }).waitFor();
  await page.getByRole('button', { name: '退出登录' }).click();
  await loginAdmin(page, baseUrl);
  await page.getByRole('link', { name: '运行状态', exact: true }).click();
  await page.locator('[aria-label="请求准入"][aria-busy="false"]').waitFor();
  await page.getByText('generation_provider', { exact: true }).waitFor();
  assert.equal(await page.getByText('什么时候使用 deterministic workflow？', { exact: true }).count(), 0);
});
