import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { loginAdmin, registerKnowledgeUserViaApi, startWorkbench } from './acceptance-env.mjs';

const request = async (api, path, token, body) => {
  const response = await fetch(`${api.baseUrl}${path}`, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    ...(body === undefined ? {} : { body: JSON.stringify(body) })
  });
  assert.equal(response.status, 200, `${path} returned ${response.status}`);
  const payload = await response.json();
  return payload.data ?? payload;
};
const logout = async (page) => {
  await page.getByRole('button', { name: '退出登录' }).click();
  await page.getByRole('heading', { name: '身份验证', exact: true }).waitFor();
};
const capture = async (page, name) => {
  if (!process.env.MAINTENANCE_SCREENSHOT_DIR) return;
  await mkdir(process.env.MAINTENANCE_SCREENSHOT_DIR, { recursive: true });
  await page.screenshot({ path: join(process.env.MAINTENANCE_SCREENSHOT_DIR, name), fullPage: true });
};

const journey = async (t, { built, action }) => {
  const { page, baseUrl, api } = await startWorkbench(t, { built });
  const members = {};
  for (const role of ['editor', 'worker', 'owner', 'successor']) {
    members[role] = await registerKnowledgeUserViaApi(api, `roadmap-browser-${role}`);
  }
  const administrator = await request(api, '/auth/login', '', { username: 'operator', password: 'safe-password' });
  const assignment = await request(api, '/maintenance/assignments', administrator.access_token, { username: 'roadmap-browser-editor' });
  await request(api, `/maintenance/assignments/${assignment.id}/accept`, members.editor, {});
  await request(api, '/chat', members.worker, { session_id: 'roadmap-seed', message: 'uncovered roadmap decision' });
  const history = await request(api, '/sessions/roadmap-seed', members.worker);
  const answer = history.messages.findLast((message) => message.type === 'assistant');
  const signal = await request(api, '/knowledge-feedback', members.worker, {
    answer_id: answer.id, label: 'out_of_scope', note: 'private roadmap source note'
  });
  const item = await request(api, '/maintenance/items', members.editor, {
    classification: 'scope-roadmap', severity: 'p3', disposition: 'needs-reproduction',
    coverage_position: 'evidence_sufficiency_refusal_and_acceptance',
    work_owner_username: 'roadmap-browser-worker', signal_ids: [signal.id]
  });
  await request(api, `/maintenance/items/${item.id}/transition`, members.editor, { expected_revision: 1, state: 'triaged' });
  const deleted = await fetch(`${api.baseUrl}/knowledge-feedback/${signal.id}`, {
    method: 'DELETE', headers: { Authorization: `Bearer ${members.worker}` }
  });
  assert.equal(deleted.status, 200);

  await loginAdmin(page, baseUrl, { username: 'roadmap-browser-editor' });
  await page.getByRole('link', { name: '知识维护', exact: true }).click();
  const workScope = page.getByRole('region', { name: '具体影响范围', exact: true });
  await page.getByRole('region', { name: '维护项详情' }).waitFor();
  assert.equal(await workScope.count(), 1);
  assert.ok((await workScope.innerText()).includes(signal.query_condition_set_identity));
  assert.equal((await workScope.innerText()).includes('private roadmap source note'), false);
  await page.getByRole('tab', { name: '路线图候选', exact: true }).click();
  const qualification = page.getByRole('form', { name: '路线图资格确认' });
  await qualification.getByLabel('来源维护项', { exact: true }).selectOption(item.id);
  await qualification.getByLabel('候选负责人', { exact: true }).fill('roadmap-browser-owner');
  await qualification.getByLabel('期望目标', { exact: true }).selectOption('clarify_scope');
  await qualification.getByLabel('无法以有界工作闭合的原因', { exact: true }).selectOption('requires_separate_scope');
  await qualification.getByRole('button', { name: '确认路线图资格', exact: true }).click();
  await page.getByText('路线图候选已建立', { exact: true }).waitFor();
  const detail = page.getByRole('region', { name: '路线图候选详情' });
  await detail.getByText('已延期', { exact: true }).waitFor();
  assert.ok((await detail.getByRole('region', { name: '具体影响范围', exact: true }).innerText()).includes(signal.query_condition_set_identity));
  assert.equal((await detail.innerText()).includes('private roadmap source note'), false);
  assert.equal(await page.getByRole('form', { name: '月度路线图评审' }).count(), 0);
  await logout(page);

  await loginAdmin(page, baseUrl, { username: 'roadmap-browser-owner' });
  await page.getByRole('link', { name: '知识维护', exact: true }).click();
  await page.getByText('当前没有分配给你的维护工作。').waitFor();
  await page.getByRole('tab', { name: '路线图候选', exact: true }).click();
  const review = page.getByRole('form', { name: '月度路线图评审' });
  await review.getByLabel('评审处置', { exact: true }).selectOption('renew');
  await review.getByLabel('处置理由', { exact: true }).selectOption('still_outside_scope');
  await review.getByLabel('后续负责人', { exact: true }).fill('roadmap-browser-successor');
  await review.getByRole('button', { name: '记录月度评审', exact: true }).click();
  await page.getByText('路线图评审已记录', { exact: true }).waitFor();
  await page.getByText('当前没有可见的路线图候选。', { exact: true }).waitFor();
  await logout(page);

  await loginAdmin(page, baseUrl, { username: 'roadmap-browser-successor' });
  await page.getByRole('link', { name: '知识维护', exact: true }).click();
  await page.getByRole('tab', { name: '路线图候选', exact: true }).click();
  await review.getByLabel('评审处置', { exact: true }).selectOption(action);
  await review.getByLabel('处置理由', { exact: true }).selectOption(action === 'close' ? 'no_longer_needed' : 'separate_discovery_needed');
  await review.getByRole('button', { name: '记录月度评审', exact: true }).click();
  await detail.getByText(action === 'close' ? '已关闭' : '已建立探索图', { exact: true }).waitFor();
  await page.reload();
  await page.getByRole('tab', { name: '路线图候选', exact: true }).click();
  await detail.getByText(action === 'close' ? '已关闭' : '已建立探索图', { exact: true }).waitFor();
  assert.equal(await review.count(), 0);
  assert.ok((await detail.getByRole('region', { name: '具体影响范围', exact: true }).innerText()).includes(signal.query_condition_set_identity));
  if (action === 'start_wayfinder') {
    const artifact = page.getByRole('region', { name: 'Wayfinder map' });
    await artifact.locator('pre').getByText('Qualified as a separate effort;', { exact: false }).waitFor();
    const pending = page.waitForEvent('download');
    await artifact.getByRole('button', { name: '下载探索图', exact: true }).click();
    const download = await pending;
    assert.match(download.suggestedFilename(), /^wayfinder-[a-f0-9]{32}\.md$/);
    const markdown = await readFile(await download.path(), 'utf8');
    assert.match(markdown, /^# Wayfinder:/);
    assert.ok(markdown.includes('no product expansion is authorized'));
    assert.equal(markdown.includes('private roadmap source note'), false);
  }
  await capture(page, `roadmap-${action}-desktop.png`);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth);
  await capture(page, `roadmap-${action}-mobile.png`);
};

for (const scenario of [
  { built: false, action: 'close' }, { built: false, action: 'start_wayfinder' }, { built: true, action: 'start_wayfinder' }
]) {
  test(`roadmap owners renew, transfer and ${scenario.action} (${scenario.built ? 'built' : 'development'})`,
    { timeout: 120000 }, async (t) => journey(t, scenario));
}
