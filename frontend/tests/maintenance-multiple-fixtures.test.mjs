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
  test(`maintainer closes complete multi-scenario source verification (${built ? 'built' : 'development'})`,
    { timeout: 120000 }, async (t) => {
      const { page, baseUrl, api } = await startWorkbench(t, {
        built, env: { BROWSER_ACCEPTANCE_TICKET24: '1', BROWSER_ACCEPTANCE_TICKET27_SOURCE: '1' }
      });
      const editor = await registerKnowledgeUserViaApi(api, 'multiple-editor');
      const worker = await registerKnowledgeUserViaApi(api, 'multiple-worker');
      const reporter = await registerKnowledgeUserViaApi(api, 'multiple-reporter');
      const admin = await request(api, '/auth/login', '', { username: 'operator', password: 'safe-password' });
      const owner = await request(api, '/auth/login', '', { username: 'ticket24-browser-maintainer', password: 'safe-password' });
      const assignment = await request(api, '/maintenance/assignments', admin.access_token, { username: 'multiple-editor' });
      await request(api, `/maintenance/assignments/${assignment.id}/accept`, editor, {});
      const sourcePath = '/editorial/entries/ticket24-browser-entry-001/sources/source-ticket24-browser-entry-001/availability';
      await request(api, sourcePath, owner.access_token, { availability: 'changed_or_unreachable_awaiting_review' });
      const scenarios = [];
      for (const suffix of ['alpha', 'beta']) {
        const question = `Which Candidate publication contract applies to scenario ${suffix}? deployment=production`;
        const answer = async (token, session) => {
          await request(api, '/chat', token, {
            session_id: session, message: question,
            query_conditions: [{ condition_id: 'production', field: 'deployment', operator: 'equals', value: 'production' }]
          });
          return (await request(api, `/sessions/${session}`, token)).messages.findLast((message) => message.type === 'assistant');
        };
        const reported = await answer(reporter, `multiple-report-${suffix}`);
        const independent = await answer(worker, `multiple-independent-${suffix}`);
        const signal = await request(api, '/knowledge-feedback', reporter, { answer_id: reported.id, label: 'outdated' });
        scenarios.push({ signal, independent });
      }
      const item = await request(api, '/maintenance/items', editor, {
        classification: 'source-freshness', severity: 'p2', disposition: 'needs-reproduction',
        coverage_position: 'evidence_sufficiency_refusal_and_acceptance',
        work_owner_username: 'multiple-worker', signal_ids: scenarios.map(({ signal }) => signal.id)
      });
      assert.equal(item.affected_scope.gap_contexts.length, 2);
      const base = `/maintenance/items/${item.id}`;
      const mutate = async (endpoint, actor, body) => request(api, base + endpoint, actor, {
        expected_revision: (await request(api, base, editor)).revision, ...body
      });
      await mutate('/transition', editor, { state: 'triaged' });
      for (const scenario of scenarios) {
        scenario.fixture = await mutate('/reproductions', worker, {
          answer_id: scenario.independent.id, signal_id: scenario.signal.id,
          expected_outcome: 'evidence_gated_answer', confirmed_synthetic_fixture: true,
          verified_observation: 'stale_source', entry_identity: 'entry:ticket24-browser-entry-001'
        });
        await mutate('/diagnosis', editor, { fixture_identity: scenario.fixture.id, observation: 'stale_source' });
        await mutate('/findings', editor, { fixture_identity: scenario.fixture.id });
      }
      await mutate('/transition', editor, { state: 'in_progress' });
      await request(api, sourcePath, owner.access_token, { availability: 'verified_usable' });
      const replay = async (scenario) => {
        scenario.replay = await mutate('/replays', worker, {
          fixture_identity: scenario.fixture.id, answer_id: scenario.independent.id
        });
        assert.equal(scenario.replay.passed, true);
      };
      await replay(scenarios[0]);
      await loginAdmin(page, baseUrl, { username: 'multiple-editor' });
      await page.getByRole('link', { name: '知识维护', exact: true }).click();
      const detail = page.getByRole('region', { name: '维护项详情' });
      await page.waitForFunction(() => document.querySelector('[aria-label="维护证据"]')?.getAttribute('aria-busy') === 'false');
      assert.equal(await detail.getByRole('button', { name: '确认来源修复', exact: true }).count(), 0);
      await replay(scenarios[1]);
      await page.reload();
      const scope = detail.getByRole('group', { name: '本次关闭验证场景' });
      await scope.waitFor();
      assert.equal(await scope.getByRole('checkbox').count(), 2);
      await scope.getByRole('checkbox').first().uncheck();
      assert.equal(await detail.getByRole('button', { name: '确认来源修复', exact: true }).count(), 0);
      await scope.getByRole('checkbox').first().check();
      if (process.env.MAINTENANCE_SCREENSHOT_DIR) {
        await mkdir(process.env.MAINTENANCE_SCREENSHOT_DIR, { recursive: true });
        for (const [name, viewport] of [['desktop', { width: 1440, height: 900 }], ['mobile', { width: 390, height: 844 }]]) {
          await page.setViewportSize(viewport);
          await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth);
          await page.screenshot({
            path: join(process.env.MAINTENANCE_SCREENSHOT_DIR, `multiple-fixtures-ready-${name}.png`), fullPage: true
          });
        }
      }
      await detail.getByRole('button', { name: '确认来源修复', exact: true }).click();
      await detail.getByRole('button', { name: '关闭确认', exact: true }).click();
      await page.reload();
      await detail.getByText('已关闭确认', { exact: true }).waitFor();
      const retained = await request(api, base, editor);
      for (const scenario of scenarios) {
        assert.ok(retained.result_links.some((link) => link.endsWith(scenario.fixture.id)));
        assert.ok(retained.result_links.some((link) => link.endsWith(scenario.replay.id)));
      }
      await page.setViewportSize({ width: 390, height: 844 });
      await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth);
      if (process.env.MAINTENANCE_SCREENSHOT_DIR) {
        await mkdir(process.env.MAINTENANCE_SCREENSHOT_DIR, { recursive: true });
        await page.screenshot({ path: join(process.env.MAINTENANCE_SCREENSHOT_DIR, 'multiple-fixtures-mobile.png'), fullPage: true });
      }
    });
}
