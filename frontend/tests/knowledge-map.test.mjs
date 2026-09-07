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

  assert.equal(await page.getByRole('heading', { name: 'Agent Orchestration' }).isVisible(), true);
  const entry = page.getByRole('article', { name: 'Prefer deterministic workflows' });
  assert.equal(await entry.getByText('已知路径应由 deterministic workflow 控制。').isVisible(), true);
  assert.equal(await entry.getByText('v1', { exact: true }).isVisible(), true);
  assert.equal(await entry.getByText('Claim-linked assurance', { exact: true }).isVisible(), true);
  assert.equal(await entry.getByText('1 个来源', { exact: true }).isVisible(), true);
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


test('Knowledge Map keeps a published controlled source discoverable without turning its locator into a public link', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t);
  await page.route('**/api/knowledge-map', async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        data: {
          total_entries: 1,
          themes: [
            {
              domain: 'workflow-vs-agent',
              label: 'Workflow 与 Agent',
              entries: [
                {
                  entry_id: 'controlled-workflow-001',
                  title: 'Controlled workflow source',
                  approved_summary: '仅在受控内部范围内使用已发布工作流证据。',
                  review_date: '2026-09-07',
                  applicable_versions: ['internal-pilot'],
                  publication_version: 'v1',
                  source_count: 1,
                  public_source_count: 0,
                  controlled_source_count: 1,
                  suggested_query: '受控来源工作流何时适用？',
                  sources: [
                    {
                      title: 'Reviewed internal workflow runbook',
                      authority: 'ZhoMind architecture group',
                      url: 'controlled://knowledge/reviewed-workflow-runbook',
                      version: '2026-09-07',
                      access_scope: 'controlled_internal'
                    }
                  ]
                }
              ]
            }
          ]
        }
      })
    });
  });

  await page.goto(`${baseUrl}knowledge`);
  const entry = page.getByRole('article', { name: 'Controlled workflow source' });
  await entry.getByText('Reviewed internal workflow runbook', { exact: true }).waitFor();
  assert.equal(
    await entry.getByText('controlled://knowledge/reviewed-workflow-runbook', { exact: true }).isVisible(),
    true
  );
  assert.equal(await entry.getByRole('link', { name: 'Reviewed internal workflow runbook' }).count(), 0);
  assert.equal(await entry.getByText('1 个来源', { exact: true }).isVisible(), true);
});
