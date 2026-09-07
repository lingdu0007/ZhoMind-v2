import assert from 'node:assert/strict';
import test from 'node:test';
import { registerKnowledgeUserViaApi, startWorkbench } from './acceptance-env.mjs';


const sendQuestion = async (page, question) => {
  await page.getByPlaceholder('请输入需要检索的问题').fill(question);
  await page.getByRole('button', { name: '发送' }).click();
};


test('Knowledge User cannot submit feedback from an ineligible live answer without frozen evidence', { timeout: 30000 }, async (t) => {
  const { page, baseUrl, api } = await startWorkbench(t, {});
  const token = await registerKnowledgeUserViaApi(api, 'feedback-browser-user');
  await page.addInitScript(({ storedToken }) => localStorage.setItem('access_token', storedToken), { storedToken: token });
  await page.goto(`${baseUrl}chat`);

  await sendQuestion(page, '什么时候使用 deterministic workflow？');
  const answer = page.getByLabel('助手消息').last();
  await answer.getByRole('status').filter({ hasText: '证据不足' }).waitFor();
  assert.equal(await answer.getByLabel('证据摘要').getByText('0 个来源').isVisible(), true);
  assert.equal(await answer.getByLabel('知识反馈').count(), 0);

  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
});


test('feedback controls require a persisted Agent answer identity', { timeout: 30000 }, async (t) => {
  const { page, baseUrl, api } = await startWorkbench(t, {});
  const token = await registerKnowledgeUserViaApi(api, 'feedback-boundary-user');
  await page.addInitScript(({ storedToken }) => localStorage.setItem('access_token', storedToken), { storedToken: token });
  await page.goto(`${baseUrl}chat`);

  await sendQuestion(page, 'zzzz 完全不存在的内容 987654321');
  const answer = page.getByLabel('助手消息').last();
  await answer.getByRole('status').filter({ hasText: '证据不足' }).waitFor();
  assert.equal(await answer.getByLabel('知识反馈').count(), 0);
});
