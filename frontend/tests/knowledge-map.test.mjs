import assert from 'node:assert/strict';
import test from 'node:test';
import { registerKnowledgeUserViaApi, startWorkbench } from './acceptance-env.mjs';


const startAsKnowledgeUser = async (t, { viewport = { width: 1440, height: 900 } } = {}) => {
  const workbench = await startWorkbench(t, { viewport });
  const token = await registerKnowledgeUserViaApi(workbench.api, 'map-user');
  await workbench.page.addInitScript(
    ({ storedToken }) => localStorage.setItem('access_token', storedToken),
    { storedToken: token }
  );
  return workbench;
};


test('Knowledge User browses the published Knowledge Map and starts a query from a published entry', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t);
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('link', { name: '知识地图' }).click();
  await page.getByRole('heading', { name: '知识地图' }).waitFor();
  await page.locator('section.knowledge-map[aria-busy="false"]').waitFor();

  assert.equal(await page.getByRole('heading', { name: 'Workflow 与 Agent' }).isVisible(), true);
  const entry = page.getByRole('article', { name: 'Prefer deterministic workflows' });
  assert.equal(await entry.getByText('已知路径应由 deterministic workflow 控制。').isVisible(), true);
  assert.equal(await entry.getByText('v1', { exact: true }).isVisible(), true);
  assert.equal(await entry.getByText('1 个公开来源', { exact: true }).isVisible(), true);
  const sourceLink = entry.getByRole('link', { name: 'Building effective agents' });
  assert.equal(await sourceLink.getAttribute('href'), 'https://www.anthropic.com/engineering/building-effective-agents');
  assert.equal(await page.getByText(/candidate private|internal-/).count(), 0);

  await entry.getByRole('button', { name: '基于此条提问' }).click();
  await page.waitForURL(/\/chat\?q=/);
  assert.equal(await page.getByPlaceholder('请输入需要检索的问题').inputValue(), '什么时候使用 deterministic workflow？');
});


test('Knowledge Map is auth-protected and remains readable on mobile', { timeout: 30000 }, async (t) => {
  const unauthenticated = await startWorkbench(t, { viewport: { width: 390, height: 844 } });
  await unauthenticated.page.goto(`${unauthenticated.baseUrl}knowledge`);
  await unauthenticated.page.waitForURL(/\/auth$/);

  const authenticated = await startAsKnowledgeUser(t, { viewport: { width: 390, height: 844 } });
  await authenticated.page.goto(`${authenticated.baseUrl}knowledge`);
  await authenticated.page.getByRole('heading', { name: '知识地图' }).waitFor();
  await authenticated.page.locator('section.knowledge-map[aria-busy="false"]').waitFor();
  assert.equal(
    await authenticated.page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    true
  );
});
