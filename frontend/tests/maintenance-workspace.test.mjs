import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { loginAdmin, registerKnowledgeUserViaApi, startWorkbench } from './acceptance-env.mjs';

const screenshot = async (page, name) => {
  const directory = process.env.MAINTENANCE_SCREENSHOT_DIR;
  if (!directory) return;
  await mkdir(directory, { recursive: true });
  await page.screenshot({ path: join(directory, `${name}.png`), fullPage: true });
};

const logout = async (page) => {
  await page.getByRole('button', { name: '退出登录' }).click();
  await page.getByRole('heading', { name: '身份验证', exact: true }).waitFor();
};

const maintenanceJourney = async (t, built) => {
  const { page, baseUrl, api } = await startWorkbench(t, { built });
  await registerKnowledgeUserViaApi(api, 'maintenance-browser-reporter');
  await registerKnowledgeUserViaApi(api, 'maintenance-browser-editor');
  await registerKnowledgeUserViaApi(api, 'maintenance-browser-worker');
  await loginAdmin(page, baseUrl, { username: 'maintenance-browser-reporter' });
  await page.getByPlaceholder('请输入需要检索的问题').fill('uncovered maintenance browser decision');
  await page.getByRole('button', { name: '发送', exact: true }).click();
  const feedback = page.getByLabel('助手消息').last().getByLabel('知识缺口报告');
  await feedback.getByRole('button', { name: '报告知识缺口' }).click();
  await feedback.getByRole('radio', { name: '超出范围' }).click();
  await feedback.getByLabel('补充说明（可选）').fill('browser private feedback detail');
  await feedback.getByRole('button', { name: '预览缺口报告' }).click();
  await feedback.getByRole('button', { name: '确认提交' }).click();
  await feedback.getByText('缺口报告已提交', { exact: true }).waitFor();
  await page.getByRole('link', { name: '知识维护', exact: true }).click();
  await page.getByRole('heading', { name: '知识维护', exact: true }).waitFor();
  await page.getByText('当前没有分配给你的维护工作。').waitFor();
  assert.equal(await page.getByRole('tab', { name: '反馈收件箱' }).count(), 0);
  assert.equal(await page.getByLabel('维护人用户名', { exact: true }).count(), 0);
  assert.equal((await page.locator('main').innerText()).includes('browser private feedback detail'), false);
  await logout(page);

  await loginAdmin(page, baseUrl);
  await page.getByRole('link', { name: '知识维护', exact: true }).click();
  await page.getByLabel('维护人用户名', { exact: true }).fill('maintenance-browser-editor');
  await page.getByRole('button', { name: '指派维护责任' }).click();
  await page.getByRole('status').getByText('维护责任已指派').waitFor();
  assert.equal((await page.locator('main').innerText()).includes('browser private feedback detail'), false);
  await page.setViewportSize({ width: 390, height: 844 });
  await screenshot(page, 'administrator-mobile');
  await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth);
  await page.setViewportSize({ width: 1440, height: 900 });
  await logout(page);

  await loginAdmin(page, baseUrl, { username: 'maintenance-browser-editor' });
  await page.getByRole('link', { name: '知识维护', exact: true }).click();
  await page.getByRole('button', { name: '接受维护责任' }).click();
  await page.getByRole('tab', { name: '反馈收件箱' }).click();
  await page.getByRole('checkbox', { name: '选择反馈' }).first().check();
  await page.getByLabel('维护分类', { exact: true }).selectOption('coverage-gap');
  await page.getByLabel('优先级', { exact: true }).selectOption('p3');
  await page.getByLabel('影响范围', { exact: true }).selectOption('evidence_sufficiency_refusal_and_acceptance');
  await page.getByLabel('Work Owner 用户名', { exact: true }).fill('maintenance-browser-worker');
  await page.getByRole('button', { name: '创建维护项' }).click();
  const detail = page.getByRole('region', { name: '维护项详情' });
  await detail.getByText('待分诊', { exact: true }).waitFor();
  assert.equal((await detail.innerText()).includes('browser private feedback detail'), false);
  await detail.getByRole('button', { name: '完成分诊' }).click();
  await detail.getByText('已分诊', { exact: true }).waitFor();
  await page.reload();
  await detail.getByText('已分诊', { exact: true }).waitFor();
  await screenshot(page, 'maintenance-desktop');
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await screenshot(page, 'maintenance-mobile');
  await page.setViewportSize({ width: 1440, height: 900 });
  await logout(page);

  await loginAdmin(page, baseUrl, { username: 'maintenance-browser-worker' });
  await page.getByPlaceholder('请输入需要检索的问题').fill('uncovered maintenance browser decision');
  await page.getByRole('button', { name: '发送', exact: true }).click();
  await page.getByLabel('助手消息').last().getByRole('button', { name: '报告知识缺口' }).waitFor();
  await page.getByRole('link', { name: '知识维护', exact: true }).click();
  const reproduction = page.getByRole('form', { name: '独立复现' });
  await reproduction.getByLabel('已保存的本人执行', { exact: true }).selectOption({ index: 1 });
  await reproduction.getByLabel('反馈目标', { exact: true }).selectOption({ index: 1 });
  await reproduction.getByLabel('预期结果', { exact: true }).selectOption('insufficient_evidence_reply');
  await reproduction.getByRole('checkbox', { name: '确认这是独立编写、可共享的非个人场景' }).check();
  await reproduction.getByRole('button', { name: '登记并运行复现' }).click();
  await page.getByText('复现场景已登记', { exact: true }).waitFor();
  assert.equal((await detail.innerText()).includes('browser private feedback detail'), false);
  assert.equal(await page.getByRole('tab', { name: '反馈收件箱' }).count(), 0);
  await logout(page);

  await loginAdmin(page, baseUrl, { username: 'maintenance-browser-editor' });
  await page.getByRole('link', { name: '知识维护', exact: true }).click();
  await detail.getByRole('button', { name: '确认诊断' }).click();
  await page.getByText('诊断已记录', { exact: true }).waitFor();
  await detail.getByRole('button', { name: '批准非个人发现' }).click();
  await page.getByText('非个人发现已批准', { exact: true }).waitFor();
  await detail.getByRole('button', { name: '开始处理' }).click();
  await detail.getByText('处理中', { exact: true }).waitFor();
  await logout(page);

  await loginAdmin(page, baseUrl, { username: 'maintenance-browser-worker' });
  await page.getByRole('link', { name: '知识维护', exact: true }).click();
  await detail.getByRole('button', { name: '重放已声明场景' }).waitFor();
  await page.waitForFunction(() => document.querySelector('[aria-label="维护证据"]')?.getAttribute('aria-busy') === 'false');
  const replayInput = detail.getByLabel('重放使用的本人执行', { exact: true });
  assert.equal(await replayInput.count(), 1);
  assert.equal(await detail.getByRole('button', { name: '重放已声明场景' }).isDisabled(), true);
  await replayInput.selectOption({ index: 1 });
  await detail.getByRole('button', { name: '重放已声明场景' }).click();
  await detail.getByText('重放通过', { exact: true }).waitFor();
  const originalFixture = await page.getByRole('region', { name: '认证复现结果' }).locator('dd').first().innerText();
  await reproduction.getByLabel('已保存的本人执行', { exact: true }).selectOption({ index: 1 });
  await reproduction.getByLabel('反馈目标', { exact: true }).selectOption({ index: 1 });
  await reproduction.getByLabel('预期结果', { exact: true }).selectOption('evidence_gated_answer');
  await reproduction.getByRole('checkbox', { name: '确认这是独立编写、可共享的非个人场景' }).check();
  await reproduction.getByRole('button', { name: '登记并运行复现' }).click();
  await page.getByText('复现场景已登记', { exact: true }).waitFor();
  await page.reload();
  await detail.getByText('重放通过', { exact: true }).waitFor();
  const approvedScenario = detail.getByLabel('已批准场景', { exact: true });
  assert.equal(await approvedScenario.count(), 1);
  assert.deepEqual(await approvedScenario.locator('option').evaluateAll((options) => options.map((option) => option.value)), [originalFixture]);
  await approvedScenario.selectOption(originalFixture);
  assert.equal(await replayInput.inputValue(), '');
  await replayInput.selectOption({ index: 1 });
  const selectedReplayAnswer = await replayInput.inputValue();
  const [repairResponse] = await Promise.all([
    page.waitForResponse((response) => response.request().method() === 'POST' && response.url().endsWith('/replays')),
    detail.getByRole('button', { name: '重放已声明场景' }).click()
  ]);
  assert.equal(repairResponse.request().postDataJSON().fixture_identity, originalFixture);
  assert.equal(repairResponse.request().postDataJSON().answer_id, selectedReplayAnswer);
  assert.equal(repairResponse.status(), 200);
  const repairedReplay = (await repairResponse.json()).data;
  await detail.getByText(repairedReplay.id, { exact: true }).waitFor();
  await detail.getByText('重放已完成', { exact: true }).waitFor();
  await detail.getByText('重放通过', { exact: true }).waitFor();
  await page.waitForFunction(() => document.querySelector('[aria-label="维护证据"]')?.getAttribute('aria-busy') === 'false');
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth);
  await screenshot(page, 'maintenance-replay-mobile');
  await page.setViewportSize({ width: 1440, height: 900 });
  await logout(page);

  await loginAdmin(page, baseUrl, { username: 'maintenance-browser-editor' });
  await page.getByRole('link', { name: '知识维护', exact: true }).click();
  await detail.getByRole('button', { name: '以边界查询解决' }).click();
  await detail.getByText('已解决', { exact: true }).waitFor();
  await detail.getByRole('button', { name: '关闭确认' }).click();
  await detail.getByText('已关闭确认', { exact: true }).waitFor();
  await page.reload();
  await detail.getByText('已关闭确认', { exact: true }).waitFor();
  await page.waitForFunction(() => document.querySelector('[aria-label="维护证据"]')?.getAttribute('aria-busy') === 'false');
  assert.equal(await detail.getByRole('button', { name: '开始处理' }).count(), 0);
  await screenshot(page, 'maintenance-closed-desktop');
};

for (const built of [false, true]) {
  test(`administrator assigns maintenance responsibility and the maintainer triages explicit feedback (${built ? 'built' : 'development'})`,
    { timeout: 120000 }, async (t) => maintenanceJourney(t, built));
}
