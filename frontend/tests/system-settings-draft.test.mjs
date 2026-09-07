// System Settings browser acceptance over a disposable real API.
// The draft lifecycle (save / apply / failed / active) uses the real settings
// endpoints; failures are seeded through the disposable API environment.
import assert from 'node:assert/strict';
import test from 'node:test';
import { loginAdmin, registerKnowledgeUserViaApi, startWorkbench } from './acceptance-env.mjs';

const openSettings = async (page, baseUrl) => {
  await page.goto(`${baseUrl}config`);
  await page.getByRole('heading', { name: '系统设置' }).waitFor();
  // The saved draft loads asynchronously; wait until the model field carries
  // the seeded value before assertions run.
  await page.waitForFunction(() => {
    const input = document.querySelector('input[aria-label="生成模型"]');
    return input && input.value !== '';
  });
};

test('System Administrator saves and applies a dirty draft without optimistically showing an active version', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, {});
  await loginAdmin(page, baseUrl);
  await openSettings(page, baseUrl);

  assert.equal(await page.getByRole('link', { name: '系统设置' }).isVisible(), true);
  assert.equal(await page.getByRole('heading', { name: '模型与提供方' }).isVisible(), true);
  assert.equal(await page.getByLabel('生成模型').inputValue(), 'Qwen/Qwen3-32B');
  assert.equal(await page.getByLabel('服务 URL', { exact: true }).inputValue(), 'https://provider.example.test/v1');
  assert.equal(await page.getByText('Provider API 密钥已配置，内容已隐藏。').isVisible(), true);
  assert.equal(await page.getByText('已保存版本 1').isVisible(), true);
  assert.equal(await page.getByText('尚无生效版本').isVisible(), true);
  assert.equal(await page.getByText('最后修改：browser-bootstrap').isVisible(), true);
  assert.equal(await page.getByText('保存并应用前，运行系统不会变化。').isVisible(), true);
  const geometry = await page.getByLabel('草稿状态').evaluate((bar) => ({
    pageWidth: document.documentElement.scrollWidth,
    viewportWidth: window.innerWidth,
    rightEdge: bar.getBoundingClientRect().right
  }));
  assert.ok(geometry.pageWidth <= geometry.viewportWidth);
  assert.ok(geometry.rightEdge <= geometry.viewportWidth);
  const [stateBarBounds, timeoutFieldBounds] = await Promise.all([
    page.getByLabel('草稿状态').evaluate((element) => element.getBoundingClientRect().toJSON()),
    page.getByLabel('服务 URL', { exact: true }).evaluate((element) => element.getBoundingClientRect().toJSON())
  ]);
  const verticalGeometry = {
    stateTop: stateBarBounds.top,
    stateBottom: stateBarBounds.bottom,
    fieldTop: timeoutFieldBounds.top,
    fieldBottom: timeoutFieldBounds.bottom
  };
  assert.ok(verticalGeometry.fieldBottom <= verticalGeometry.stateTop || verticalGeometry.fieldTop >= verticalGeometry.stateBottom);

  await page.getByLabel('生成模型').fill('Qwen/Qwen3-14B');
  assert.equal(await page.getByText('存在未保存的草稿修改').isVisible(), true);
  assert.equal(await page.getByText('模型已修改').isVisible(), true);
  await page.getByRole('button', { name: '重置到已保存草稿' }).click();
  assert.equal(await page.getByLabel('生成模型').inputValue(), 'Qwen/Qwen3-32B');
  assert.equal(await page.getByText('存在未保存的草稿修改').count(), 0);

  await page.getByLabel('Provider API 密钥').fill('replacement-value');
  assert.equal(await page.getByText('Provider API 密钥已修改').isVisible(), true);
  await page.getByRole('button', { name: '重置到已保存草稿' }).click();
  assert.equal(await page.getByText('Provider API 密钥已修改').count(), 0);

  await page.getByLabel('生成模型').fill('Qwen/Qwen3-14B');
  await page.getByRole('button', { name: '保存并应用', exact: true }).click();
  await page.getByRole('button', { name: '正在应用版本 2。' }).waitFor();
  assert.equal(await page.getByText('已保存版本 2').isVisible(), true);
  assert.equal(await page.getByText('尚无生效版本').isVisible(), true);
  assert.equal(await page.getByRole('button', { name: '正在应用版本 2。' }).isDisabled(), true);
  assert.equal(await page.getByLabel('生成模型').isDisabled(), true);
  await page.getByText('生效版本 2').waitFor({ timeout: 8000 });
  assert.equal(await page.getByText('设置已生效。').isVisible(), true);
  await page.reload();
  await page.getByText('生效版本 2').waitFor();
  assert.equal(await page.getByText('设置已生效。').isVisible(), true);
  assert.equal(await page.getByText('存在未保存的草稿修改').count(), 0);
});

test('System Administrator sees a failed application after refresh and can retry the saved version', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, { env: { BROWSER_ACCEPTANCE_SETTINGS_FAILED: '1' } });
  await loginAdmin(page, baseUrl);
  await openSettings(page, baseUrl);

  await page.getByText('应用失败：运行系统未接受该保存版本').waitFor();
  assert.equal(await page.getByText('尚无生效版本').isVisible(), true);

  await page.getByRole('button', { name: '重试应用版本 1', exact: true }).click();
  await page.getByRole('button', { name: '正在应用版本 1。' }).waitFor();
  await page.getByText('生效版本 1').waitFor({ timeout: 8000 });
  assert.equal(await page.getByText('设置已生效。').isVisible(), true);
});

test('settings draft validation keeps edits visible and exposes field-level feedback without showing a secret', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, {});
  await loginAdmin(page, baseUrl);
  await openSettings(page, baseUrl);

  await page.getByLabel('生成模型').fill('unsafe value');
  await page.getByRole('button', { name: '保存并应用', exact: true }).click();

  await page.getByRole('alert').waitFor();
  assert.equal(await page.getByRole('alert').innerText(), '草稿未保存，请修正标记字段后重试。');
  assert.equal(await page.getByText('model identifier is unsafe or unsupported').isVisible(), true);
  assert.equal(await page.getByLabel('生成模型').inputValue(), 'unsafe value');
  assert.equal(await page.getByText('存在未保存的草稿修改').isVisible(), true);
});

test('Knowledge User is redirected before the gated settings draft can load', { timeout: 30000 }, async (t) => {
  const { page, baseUrl, api } = await startWorkbench(t, {});
  const token = await registerKnowledgeUserViaApi(api, 'knowledge-user');
  await page.addInitScript(
    ({ storedToken }) => localStorage.setItem('access_token', storedToken),
    { storedToken: token }
  );

  await page.goto(`${baseUrl}config`);
  await page.waitForURL(/\/chat\?notice=admin-required$/);
  assert.equal(await page.getByRole('heading', { name: '系统设置' }).count(), 0);
  assert.equal(await page.getByRole('link', { name: '系统设置' }).count(), 0);
});

test('System Settings keeps its desktop-only boundary explicit on a mobile viewport', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, { viewport: { width: 390, height: 844 } });
  await loginAdmin(page, baseUrl);

  await page.goto(`${baseUrl}config`);
  await page.getByRole('heading', { name: '系统设置' }).waitFor();
  assert.equal(await page.getByText('旧版兼容配置仅支持桌面工作区。').isVisible(), true);
  assert.equal(await page.getByRole('form').count(), 0);
  assert.equal(await page.getByLabel('草稿状态').count(), 0);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
});

test('System Settings exposes only one non-secret generation provider configuration', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, {});
  await loginAdmin(page, baseUrl);
  await openSettings(page, baseUrl);

  await page.getByLabel('服务 URL', { exact: true }).waitFor();
  assert.equal(await page.getByLabel('模型提供方').inputValue(), 'ark');
  assert.equal(await page.getByLabel('生成模型').inputValue(), 'Qwen/Qwen3-32B');
  assert.equal(await page.getByLabel('嵌入模型').count(), 0);
  assert.equal(await page.getByText('Provider API 密钥已配置，内容已隐藏。').isVisible(), true);
});
