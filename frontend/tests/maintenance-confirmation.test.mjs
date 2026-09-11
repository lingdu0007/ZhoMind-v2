import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { loginAdmin, registerKnowledgeUserViaApi, startWorkbench } from './acceptance-env.mjs';

const request = async (api, path, token, body) => {
  const response = await fetch(`${api.baseUrl}${path}`, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    ...(body === undefined ? {} : { body: JSON.stringify(body) })
  });
  const result = await response.json();
  assert.equal(response.status, 200, `${path}: ${JSON.stringify(result)}`);
  return result.data ?? result;
};

for (const built of [false, true]) {
  test(`confirmation closes only after its retained scenario is approved (${built ? 'built' : 'development'})`,
    { timeout: 120000 }, async (t) => {
      const { page, baseUrl, api } = await startWorkbench(t, {
        built, env: { BROWSER_ACCEPTANCE_TICKET24: '1', BROWSER_ACCEPTANCE_TICKET27_SOURCE: '1' }
      });
      const editor = await registerKnowledgeUserViaApi(api, 'confirmation-editor');
      const worker = await registerKnowledgeUserViaApi(api, 'confirmation-worker');
      const reporter = await registerKnowledgeUserViaApi(api, 'confirmation-reporter');
      const admin = await request(api, '/auth/login', '', { username: 'operator', password: 'safe-password' });
      const assignment = await request(api, '/maintenance/assignments', admin.access_token, { username: 'confirmation-editor' });
      await request(api, `/maintenance/assignments/${assignment.id}/accept`, editor, {});
      const scenarios = [];
      for (const suffix of ['alpha']) {
        const answer = async (token, session) => {
          await request(api, '/chat', token, {
            session_id: session,
            message: `Which Candidate publication contract applies to scenario ${suffix}? deployment=production`,
            query_conditions: [{ condition_id: 'production', field: 'deployment', operator: 'equals', value: 'production' }]
          });
          return (await request(api, `/sessions/${session}`, token)).messages.findLast((message) => message.type === 'assistant');
        };
        const reported = await answer(reporter, `confirmation-report-${suffix}`);
        const independent = await answer(worker, `confirmation-independent-${suffix}`);
        const signal = await request(api, '/knowledge-feedback', reporter, {
          answer_id: reported.id, entry_id: 'ticket24-browser-entry-001', label: 'helpful'
        });
        scenarios.push({ signal, independent });
      }
      const item = await request(api, '/maintenance/items', editor, {
        classification: 'confirmation', severity: 'p3', disposition: 'needs-reproduction',
        coverage_position: 'evidence_sufficiency_refusal_and_acceptance',
        work_owner_username: 'confirmation-worker', signal_ids: scenarios.map(({ signal }) => signal.id)
      });
      assert.equal(item.affected_scope.entry_versions.length, 1);
      const base = `/maintenance/items/${item.id}`;
      const mutate = async (endpoint, actor, body) => request(api, base + endpoint, actor, {
        expected_revision: (await request(api, base, editor)).revision, ...body
      });
      await mutate('/transition', editor, { state: 'triaged' });
      const approve = async (scenario) => {
        scenario.fixture = await mutate('/reproductions', worker, {
          answer_id: scenario.independent.id, signal_id: scenario.signal.id,
          expected_outcome: 'evidence_gated_answer', confirmed_synthetic_fixture: true,
          verified_observation: 'confirmation'
        });
        await mutate('/diagnosis', editor, { fixture_identity: scenario.fixture.id, observation: 'confirmation' });
        scenario.finding = await mutate('/findings', editor, { fixture_identity: scenario.fixture.id });
      };
      await loginAdmin(page, baseUrl, { username: 'confirmation-editor' });
      await page.getByRole('link', { name: '知识维护', exact: true }).click();
      const detail = page.getByRole('region', { name: '维护项详情' });
      await page.waitForFunction(() => document.querySelector('[aria-label="维护证据"]')?.getAttribute('aria-busy') === 'false');
      assert.equal(await detail.getByRole('button', { name: '关闭确认', exact: true }).count(), 0);
      await approve(scenarios[0]);
      await page.reload();
      await detail.getByRole('button', { name: '关闭确认', exact: true }).waitFor();
      await detail.getByRole('button', { name: '开始处理', exact: true }).click();
      await detail.getByRole('button', { name: '开始处理', exact: true }).waitFor({ state: 'hidden' });
      await detail.getByRole('button', { name: '关闭确认', exact: true }).click();
      await page.reload();
      await detail.getByText('已关闭确认', { exact: true }).waitFor();
      assert.equal(await detail.getByRole('region', { name: '修复后验证' }).count(), 0);
      const retained = await request(api, base, editor);
      assert.equal(retained.replay_identity ?? null, null);
      for (const scenario of scenarios) {
        assert.ok(retained.result_links.includes(scenario.finding.result_links[0]));
      }
      for (const [name, viewport] of [['desktop', { width: 1440, height: 900 }], ['mobile', { width: 390, height: 844 }]]) {
        await page.setViewportSize(viewport);
        await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth);
        if (process.env.MAINTENANCE_SCREENSHOT_DIR) {
          await mkdir(process.env.MAINTENANCE_SCREENSHOT_DIR, { recursive: true });
          await page.screenshot({
            path: join(process.env.MAINTENANCE_SCREENSHOT_DIR, `confirmation-closed-${name}.png`), fullPage: true
          });
        }
      }
    });
}
