import assert from 'node:assert/strict';
import test from 'node:test';
import { registerKnowledgeUserViaApi, startWorkbench } from './acceptance-env.mjs';


const sendQuestion = async (page, question) => {
  await page.getByPlaceholder('请输入需要检索的问题').fill(question);
  await page.getByRole('button', { name: '发送' }).click();
};

test('helpful feedback is only submitted after preview and explicit confirmation', { timeout: 30000 }, async (t) => {
  const { page, baseUrl, api } = await startWorkbench(t, {});
  const token = await registerKnowledgeUserViaApi(api, 'confirmed-helpful-user');
  await page.addInitScript((value) => localStorage.setItem('access_token', value), token);
  await page.goto(`${baseUrl}chat`);
  const conditions = page.locator('.composer').getByLabel('查询条件');
  await conditions.getByRole('button', { name: '添加查询条件' }).click();
  await conditions.getByLabel('条件 1 字段').fill('execution_path');
  await conditions.getByLabel('条件 1 值').fill('known');
  await sendQuestion(page, '什么时候使用 deterministic workflow？');
  const feedback = page.getByLabel('助手消息').last().getByLabel('知识反馈', { exact: true });
  const retained = async () => {
    const response = await fetch(`${api.baseUrl}/knowledge-feedback`, {
      headers: { Authorization: `Bearer ${token}` }
    });
    assert.equal(response.status, 200);
    return (await response.json()).data.items;
  };
  await feedback.getByRole('radio', { name: '有帮助' }).waitFor();
  for (const label of ['有帮助', '证据不足', '已过时', '超出范围']) {
    assert.equal(await feedback.getByRole('radio', { name: label }).count(), 1);
  }
  assert.deepEqual(await retained(), []);
  await feedback.getByRole('radio', { name: '有帮助' }).click();
  await feedback.getByRole('button', { name: '预览反馈' }).click();
  assert.deepEqual(await retained(), []);
  await feedback.getByRole('button', { name: '取消', exact: true }).click();
  assert.deepEqual(await retained(), []);
  await feedback.getByRole('radio', { name: '有帮助' }).click();
  await feedback.getByRole('button', { name: '预览反馈' }).click();
  await feedback.getByRole('button', { name: '确认提交' }).click();
  await feedback.getByText('反馈已提交', { exact: true }).waitFor();
  assert.equal((await retained()).length, 1);
});


test('Knowledge User previews and confirms a closed insufficient knowledge-gap report without a cited entry or transcript', { timeout: 30000 }, async (t) => {
  const { page, baseUrl, api } = await startWorkbench(t, {
    env: { BROWSER_ACCEPTANCE_SOURCE_LOST: '1' }
  });
  const token = await registerKnowledgeUserViaApi(api, 'feedback-browser-user');
  await page.addInitScript(({ storedToken }) => localStorage.setItem('access_token', storedToken), { storedToken: token });
  await page.goto(`${baseUrl}chat`);

  await sendQuestion(page, '什么时候使用 deterministic workflow？');
  const answer = page.getByLabel('助手消息').last();
  const insufficiency = answer.getByLabel('证据不足回复');
  await insufficiency.waitFor();
  assert.equal(
    await insufficiency.getByText('Insufficient Evidence Reply', { exact: true }).isVisible(),
    true
  );
  assert.equal(await answer.getByLabel('证据摘要').count(), 0);
  assert.equal(await answer.getByLabel('知识反馈').count(), 0);

  const gapReport = answer.getByLabel('知识缺口报告');
  await gapReport.getByRole('button', { name: '报告知识缺口' }).click();
  for (const label of ['有帮助', '证据不足', '已过时', '超出范围']) {
    assert.equal(await gapReport.getByRole('radio', { name: label }).count(), 1);
  }
  await gapReport.getByRole('radio', { name: '已过时' }).click();
  await gapReport.getByRole('button', { name: '预览缺口报告' }).click();
  await gapReport.getByLabel('缺口报告预览').getByText('已过时', { exact: true }).waitFor();
  await gapReport.getByRole('button', { name: '取消', exact: true }).click();
  await gapReport.getByRole('button', { name: '报告知识缺口' }).click();
  await gapReport.getByRole('radio', { name: '证据不足' }).click();
  await gapReport.getByRole('button', { name: '复制当前问题到共享说明' }).click();
  assert.equal(
    await gapReport.getByLabel('补充说明（可选）').inputValue(),
    '什么时候使用 deterministic workflow？'
  );
  await gapReport.getByLabel('补充说明（可选）').fill('缺少可复现的发布前验证证据。');
  await gapReport.getByRole('button', { name: '预览缺口报告' }).click();

  const preview = gapReport.getByLabel('缺口报告预览');
  await preview.waitFor();
  const previewText = await preview.innerText();
  assert.equal(previewText.includes('什么时候使用 deterministic workflow？'), false);
  assert.equal(previewText.includes('未检索到足够相关的知识片段'), false);

  const submittedRequest = page.waitForRequest('**/api/knowledge-feedback');
  const submittedResponse = page.waitForResponse('**/api/knowledge-feedback');
  await preview.getByRole('button', { name: '确认提交' }).click();
  const [request, response] = await Promise.all([submittedRequest, submittedResponse]);
  assert.equal(response.status(), 200);
  const payload = JSON.parse(request.postData() || '{}');
  assert.equal(typeof payload.answer_id, 'string');
  assert.equal(payload.label, 'insufficient_evidence');
  assert.equal('entry_id' in payload, false);
  assert.equal('question' in payload, false);
  assert.equal('answer' in payload, false);
  assert.equal(payload.note, '缺少可复现的发布前验证证据。');
  await gapReport.getByText('缺口报告已提交', { exact: true }).waitFor();

  const sessions = await fetch(`${api.baseUrl}/sessions`, {
    headers: { Authorization: `Bearer ${token}` }
  });
  assert.equal(sessions.status, 200);
  const sessionId = (await sessions.json()).data.sessions[0]?.session_id;
  assert.equal(typeof sessionId, 'string');
  const deletedSession = await fetch(`${api.baseUrl}/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
    headers: { Authorization: `Bearer ${token}` }
  });
  assert.equal(deletedSession.status, 200);

  await page.reload();
  const retainedFeedback = page.getByLabel('已保留反馈');
  await retainedFeedback.waitFor();
  await retainedFeedback.getByText('Insufficient Evidence Reply', { exact: true }).waitFor();
  assert.equal(await retainedFeedback.getByText('Insufficient Evidence Reply', { exact: true }).isVisible(), true);
  assert.equal((await retainedFeedback.innerText()).includes('什么时候使用 deterministic workflow？'), false);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForFunction(() => document.documentElement.scrollWidth <= window.innerWidth);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
  await retainedFeedback.getByRole('button', { name: '撤回反馈' }).click();
  await page.waitForFunction(() => document.querySelector('[aria-label="已保留反馈"]') === null);

});


test('evidence feedback controls remain absent from an insufficient answer', { timeout: 30000 }, async (t) => {
  const { page, baseUrl, api } = await startWorkbench(t, {});
  const token = await registerKnowledgeUserViaApi(api, 'feedback-boundary-user');
  await page.addInitScript(({ storedToken }) => localStorage.setItem('access_token', storedToken), { storedToken: token });
  await page.goto(`${baseUrl}chat`);

  await sendQuestion(page, 'zzzz 完全不存在的内容 987654321');
  const answer = page.getByLabel('助手消息').last();
  await answer.getByLabel('证据不足回复').waitFor();
  assert.equal(await answer.getByLabel('知识反馈').count(), 0);
  assert.equal(await answer.getByLabel('知识缺口报告').count(), 1);
});

test('a legacy retained feedback record remains visible and withdrawable without an inferred outcome', { timeout: 30000 }, async (t) => {
  const { page, baseUrl, api } = await startWorkbench(t, {});
  const token = await registerKnowledgeUserViaApi(api, 'legacy-feedback-user');
  let items = [
    {
      id: 'legacy-feedback-001',
      answer_id: 'legacy-answer-001',
      entry_id: 'legacy-entry-001',
      knowledge_edition: 'publication:v1',
      label: 'outdated',
      created_at: '2026-09-07T08:00:00Z',
      expires_at: '2027-03-06T08:00:00Z',
      retention_days: 180
    }
  ];
  await page.addInitScript(({ storedToken }) => localStorage.setItem('access_token', storedToken), { storedToken: token });
  await page.route('**/api/knowledge-feedback**', async (route) => {
    if (route.request().method() === 'GET') {
      await route.fulfill({ contentType: 'application/json', body: JSON.stringify({ data: { items } }) });
      return;
    }
    const id = route.request().url().split('/').at(-1);
    items = items.filter((item) => item.id !== id);
    await route.fulfill({ contentType: 'application/json', body: JSON.stringify({ data: { id, deleted: true } }) });
  });
  await page.goto(`${baseUrl}chat`);

  const retained = page.getByLabel('已保留反馈');
  await retained.getByText('历史反馈', { exact: true }).waitFor();
  assert.equal(await retained.getByText('已过时', { exact: true }).isVisible(), true);
  await retained.getByRole('button', { name: '撤回反馈' }).click();
  await page.waitForFunction(() => document.querySelector('[aria-label="已保留反馈"]') === null);
});
