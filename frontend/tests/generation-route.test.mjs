import assert from 'node:assert/strict';
import test from 'node:test';
import { loginAdmin, startWorkbench } from './acceptance-env.mjs';

test('administrator saves an inactive approved-route draft and credentials disappear after submission', { timeout: 30000 }, async (t) => {
  const { page, baseUrl, api } = await startWorkbench(t, {});
  await loginAdmin(page, baseUrl);
  await page.goto(`${baseUrl}config`);
  await page.getByRole('heading', { name: '批准生成路由' }).waitFor();
  await page.waitForFunction(() => document.querySelector('[data-testid="active-route"]')?.textContent.includes('provider_route:'));
  const previousActive = await page.getByTestId('active-route').innerText();
  const token = await page.evaluate(() => localStorage.getItem('access_token'));
  const headers = { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' };
  const previous = (await (await fetch(`${api.baseUrl}/settings/generation-route`, { headers })).json()).data.active;
  await page.getByLabel('路由模型 1').fill('route-model');
  await page.getByLabel('路由服务 URL 1').fill('https://provider.example.test/v1');
  await page.getByLabel('路由密钥 1').fill('route-test-placeholder');
  await page.getByRole('button', { name: '保存路由', exact: true }).click();
  await page.getByText('路由草稿已保存').waitFor();
  assert.equal(await page.getByLabel('路由密钥 1').inputValue(), '');
  assert.equal(await page.getByTestId('active-route').innerText(), previousActive);
  await page.reload();
  await page.getByRole('heading', { name: '批准生成路由' }).waitFor();
  await page.waitForFunction(() => document.querySelector('[aria-label="路由模型 1"]')?.value === 'route-model');
  assert.equal(await page.getByLabel('路由密钥 1').inputValue(), '');
  assert.equal(await page.getByRole('button', { name: /复制.*密钥/ }).count(), 0);
  await page.getByLabel('路由验收记录').fill('delivery_acceptance_record:missing');
  await page.getByRole('button', { name: '验证并激活路由' }).click();
  await page.getByRole('alert').filter({ hasText: '路由未激活' }).waitFor();
  assert.equal(await page.getByTestId('active-route').innerText(), previousActive);

  const draft = (await (await fetch(`${api.baseUrl}/settings/generation-route`, { headers })).json()).data.draft;
  const oldRecord = (await (await fetch(`${api.baseUrl}/acceptance/records/${previous.providers[0].validation_evidence.record_identity}`, { headers })).json()).data;
  const payload = Object.fromEntries([
    'stage', 'affected_scope', 'content_identities', 'product_identities', 'conditions',
    'assumptions', 'checks', 'known_limits', 'risks', 'evidence_links', 'reacceptance_triggers'
  ].map((key) => [key, oldRecord[key]]));
  const replacements = new Map([
    [previous.route_identity, draft.route_identity],
    [previous.providers[0].approval_identity, draft.providers[0].approval_identity]
  ]);
  payload.product_identities = payload.product_identities.map((id) => replacements.get(id) ?? id);
  payload.checks = payload.checks.map((check) => ({
    ...check, identity_dependencies: check.identity_dependencies.map((id) => replacements.get(id) ?? id)
  }));
  const created = await fetch(`${api.baseUrl}/acceptance/records`, { method: 'POST', headers, body: JSON.stringify(payload) });
  assert.equal(created.status, 200, await created.clone().text());
  const record = (await created.json()).data;
  const approved = await fetch(`${api.baseUrl}/acceptance/records/${record.record_id}/status`, {
    method: 'POST', headers, body: JSON.stringify({
      status: 'active', reason_code: 'checks_verified',
      verified_checks: record.checks.filter((check) => ['passed', 'carried_forward'].includes(check.result))
        .map(({ check_id, evidence_links }) => ({ check_id, evidence_links }))
    })
  });
  assert.equal(approved.status, 200, await approved.clone().text());
  await page.getByLabel('路由验收记录').fill(record.record_id);
  await page.getByRole('button', { name: '验证并激活路由' }).click();
  await page.getByText('批准路由已激活', { exact: true }).waitFor();
  assert.equal(await page.getByTestId('active-route').innerText(), draft.route_identity);
  await page.reload();
  await page.waitForFunction((id) => document.querySelector('[data-testid="active-route"]')?.textContent === id, draft.route_identity);
  await page.screenshot({ path: '/tmp/ticket22-route-desktop.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByLabel('路由模型 1').scrollIntoViewIfNeeded();
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  const fieldsFit = await page.locator('.generation-route input, .generation-route select, .generation-route button').evaluateAll(
    (elements) => elements.every((element) => {
      const bounds = element.getBoundingClientRect();
      return bounds.left >= 0 && bounds.right <= innerWidth;
    })
  );
  assert.equal(fieldsFit, true);
  await page.screenshot({ path: '/tmp/ticket22-route-mobile.png', fullPage: true });
});

test('cancelled provider attempts remain visible only as non-content administrator observations', { timeout: 30000 }, async (t) => {
  const { page, baseUrl, api } = await startWorkbench(t, { env: { BROWSER_ACCEPTANCE_LLM_DELAY_MS: '4000' } });
  page.setDefaultTimeout(6000);
  await loginAdmin(page, baseUrl);
  const token = await page.evaluate(() => localStorage.getItem('access_token'));
  const conditions = page.locator('.composer').getByLabel('查询条件');
  await conditions.getByRole('button', { name: '添加查询条件' }).click();
  await conditions.getByLabel('条件 1 字段').fill('execution_path');
  await conditions.getByLabel('条件 1 运算符').fill('equals');
  await conditions.getByLabel('条件 1 值').fill('known');
  await page.getByPlaceholder('请输入需要检索的问题').fill('什么时候使用 deterministic workflow？');
  await page.getByRole('button', { name: '发送' }).click();
  await page.getByRole('status').filter({ hasText: '正在生成回答' }).waitFor({ timeout: 6000 }).catch(async (error) => {
    await page.screenshot({ path: '/tmp/ticket22-cancellation-failure.png', fullPage: true });
    throw error;
  });
  await page.getByRole('button', { name: '停止' }).click();
  await page.getByLabel('助手消息').last().locator('.answer-outcome--terminal').getByText('Stopped', { exact: true }).waitFor();
  await page.waitForFunction(async ({ url, authorization }) => {
    const response = await fetch(url, { headers: { Authorization: authorization } });
    const data = (await response.json()).data;
    return data.generation.route_executions.some((event) =>
      event.route_reason === 'user_cancellation' && event.attempts[0]?.approval_identity);
  }, { url: `${baseUrl}api/operations`, authorization: `Bearer ${token}` }, { timeout: 4000 });
  const response = await fetch(`${api.baseUrl}/operations`, { headers: { Authorization: `Bearer ${token}` } });
  const text = await response.text();
  assert.equal(text.includes('什么时候使用 deterministic workflow'), false);
  assert.equal(text.includes('已知路径应由 deterministic workflow 控制'), false);
});
