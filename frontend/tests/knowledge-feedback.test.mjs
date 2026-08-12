import assert from 'node:assert/strict';
import test from 'node:test';
import { registerKnowledgeUserViaApi, startWorkbench } from './acceptance-env.mjs';


const sendQuestion = async (page, question) => {
  await page.getByPlaceholder('请输入需要检索的问题').fill(question);
  await page.getByRole('button', { name: '发送' }).click();
};


test('Knowledge User submits explicit answer feedback and administrator classifies a private-safe work item', { timeout: 40000 }, async (t) => {
  const { page, baseUrl, api } = await startWorkbench(t, {});
  const token = await registerKnowledgeUserViaApi(api, 'feedback-browser-user');
  await page.addInitScript(({ storedToken }) => localStorage.setItem('access_token', storedToken), { storedToken: token });
  await page.goto(`${baseUrl}chat`);

  const privateQuestion = '什么时候使用 deterministic workflow？';
  await sendQuestion(page, privateQuestion);
  const answer = page.getByLabel('助手消息').last();
  await answer.getByRole('status').filter({ hasText: '已完成' }).waitFor();
  const feedback = answer.getByLabel('知识反馈');
  await feedback.waitFor();
  assert.deepEqual(
    await feedback.getByRole('radio').allTextContents(),
    ['有帮助', '证据不足', '已过时', '超出范围']
  );
  await feedback.getByRole('radio', { name: '证据不足' }).click();
  await feedback.getByLabel('补充说明（可选）').fill('请明确 retry 状态码的适用版本。');
  await feedback.getByRole('button', { name: '提交反馈' }).click();
  await feedback.getByText('反馈已提交').waitFor();
  assert.equal(await feedback.getByText('保留 180 天').isVisible(), true);

  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
  await page.goto(`${baseUrl}reviews`);
  await page.waitForURL(/\/chat\?notice=admin-required$/);

  await page.getByRole('button', { name: '退出登录' }).click();
  await page.waitForURL(/\/auth$/);
  await page.getByLabel('用户名').fill('operator');
  await page.getByLabel('密码').fill('safe-password');
  await page.getByRole('button', { name: '登录' }).click();
  await page.waitForURL(/\/chat$/);
  await page.getByRole('link', { name: '知识复核' }).click();
  await page.getByRole('heading', { name: '知识复核' }).waitFor();

  const item = page.getByRole('article', { name: /synthetic-workflow-001/ });
  await item.waitFor();
  assert.equal(await item.getByText('证据不足').isVisible(), true);
  assert.equal(await item.getByText('请明确 retry 状态码的适用版本。').isVisible(), true);
  assert.equal(await page.getByText(privateQuestion).count(), 0);
  assert.equal(await page.getByText('已知路径应由 deterministic workflow 控制。').count(), 0);
  assert.equal(await page.getByText(/internal-|private-/).count(), 0);

  await item.getByLabel('质量分级').selectOption('p1');
  await item.getByLabel('处理状态').selectOption('reviewed');
  await item.getByRole('button', { name: '保存分类' }).click();
  await item.locator('header strong').filter({ hasText: '已复核' }).waitFor();
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
