import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { loginAdmin, registerKnowledgeUserViaApi, startWorkbench } from './acceptance-env.mjs';
import { publicationAcceptance } from './maintenance-acceptance.mjs';

const request = async (api, path, token, body) => {
  const response = await fetch(`${api.baseUrl}${path}`, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    ...(body === undefined ? {} : { body: JSON.stringify(body) })
  });
  const result = await response.json();
  assert.equal(response.status, 200, `${path}: ${JSON.stringify(result)}`);
  return result.data ?? result;
};

for (const severity of ['p2', 'p0', 'p1']) {
for (const built of [false, true]) {
  test(`maintainer confirms a verified ${severity} source repair and reloads terminal closure (${built ? 'built' : 'development'})`,
    { timeout: 120000 }, async (t) => {
      const { page, baseUrl, api } = await startWorkbench(t, {
        built, env: { BROWSER_ACCEPTANCE_TICKET24: '1', BROWSER_ACCEPTANCE_TICKET27_SOURCE: '1' }
      });
      const editor = await registerKnowledgeUserViaApi(api, 'source-browser-editor');
      const worker = await registerKnowledgeUserViaApi(api, 'source-browser-worker');
      const reporter = await registerKnowledgeUserViaApi(api, 'source-browser-reporter');
      const admin = await request(api, '/auth/login', '', { username: 'operator', password: 'safe-password' });
      const owner = await request(api, '/auth/login', '', { username: 'ticket24-browser-maintainer', password: 'safe-password' });
      const assignment = await request(api, '/maintenance/assignments', admin.access_token, { username: 'source-browser-editor' });
      await request(api, `/maintenance/assignments/${assignment.id}/accept`, editor, {});
      const entry = 'ticket24-browser-entry-001';
      const source = `source-${entry}`;
      const sourcePath = `/editorial/entries/${entry}/sources/${source}/availability`;
      const highSeverity = severity !== 'p2';
      if (!highSeverity) await request(api, sourcePath, owner.access_token, { availability: 'changed_or_unreachable_awaiting_review' });
      const answer = async (token, session, outcome = 'insufficient_evidence_reply') => {
        const result = await request(api, '/chat', token, {
          session_id: session, message: 'Which Candidate publication contract applies? deployment=production',
          query_conditions: [{ condition_id: 'ticket24-production', field: 'deployment', operator: 'equals', value: 'production' }]
        });
        assert.equal(result.outcome, outcome);
        const history = await request(api, `/sessions/${session}`, token);
        return history.messages.findLast((message) => message.type === 'assistant');
      };
      const reported = await answer(reporter, 'source-browser-report', highSeverity ? 'evidence_gated_answer' : 'insufficient_evidence_reply');
      const signal = await request(api, '/knowledge-feedback', reporter, {
        answer_id: reported.id, label: 'outdated', ...(highSeverity ? { entry_id: entry } : {})
      });
      let containment;
      if (highSeverity) {
        containment = await publicationAcceptance(request, api, admin.access_token, entry);
        await request(api, sourcePath, owner.access_token, { availability: 'changed_or_unreachable_awaiting_review' });
        await request(api, `/acceptance/records/${containment.identity}/status`, admin.access_token, {
          status: 'suspended', reason_code: 'integrity_failure',
          status_failure: {
            check_id: 'check:entry-supported-query', reason: 'verified source integrity failure', failure_kind: 'entry_specific',
            blocking_scope: { scope: 'entry_version', identity: containment.publication },
            evidence_links: ['evidence://maintenance/source-integrity']
          }
        });
      }
      const independent = await answer(worker, 'source-browser-independent');
      const item = await request(api, '/maintenance/items', editor, {
        classification: 'source-freshness', severity, disposition: 'needs-reproduction',
        coverage_position: 'evidence_sufficiency_refusal_and_acceptance',
        work_owner_username: 'source-browser-worker', signal_ids: [signal.id],
        ...(containment ? { containment_record_identity: containment.identity } : {})
      });
      const base = `/maintenance/items/${item.id}`;
      await request(api, `${base}/transition`, editor, { expected_revision: 1, state: 'triaged' });
      const fixture = await request(api, `${base}/reproductions`, worker, {
        expected_revision: 2, answer_id: independent.id, signal_id: signal.id,
        expected_outcome: 'evidence_gated_answer', confirmed_synthetic_fixture: true,
        verified_observation: 'stale_source', entry_identity: `entry:${entry}`
      });
      await request(api, `${base}/diagnosis`, editor, {
        expected_revision: 3, fixture_identity: fixture.id, observation: 'stale_source'
      });
      await request(api, `${base}/findings`, editor, { expected_revision: 4, fixture_identity: fixture.id });
      await request(api, `${base}/transition`, editor, { expected_revision: 5, state: 'in_progress' });
      await request(api, sourcePath, owner.access_token, { availability: 'verified_usable' });
      const replay = await request(api, `${base}/replays`, worker, {
        expected_revision: 6, fixture_identity: fixture.id, answer_id: independent.id
      });
      assert.equal(replay.passed, true);
      const reacceptance = highSeverity ? await publicationAcceptance(request, api, admin.access_token, entry, replay.id) : null;
      await loginAdmin(page, baseUrl, { username: 'source-browser-editor' });
      await page.getByRole('link', { name: '知识维护', exact: true }).click();
      const detail = page.getByRole('region', { name: '维护项详情' });
      const resolve = detail.getByRole('button', { name: '确认来源修复', exact: true });
      if (reacceptance) {
        await page.waitForFunction(() => document.querySelector('[aria-label="维护证据"]')?.getAttribute('aria-busy') === 'false');
        const record = detail.getByLabel('重新验收记录标识', { exact: true });
        assert.equal(await record.count(), 1);
        assert.equal(await resolve.count(), 0);
        await record.fill(reacceptance.identity);
        if (process.env.MAINTENANCE_SCREENSHOT_DIR) {
          await mkdir(process.env.MAINTENANCE_SCREENSHOT_DIR, { recursive: true });
          await page.setViewportSize({ width: 390, height: 844 });
          await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth);
          await page.screenshot({
            path: join(process.env.MAINTENANCE_SCREENSHOT_DIR, `source-${severity}-reacceptance-mobile.png`), fullPage: true
          });
          await page.setViewportSize({ width: 1440, height: 900 });
          await page.evaluate(() => window.scrollTo(0, 0));
          await page.screenshot({
            path: join(process.env.MAINTENANCE_SCREENSHOT_DIR, `source-${severity}-reacceptance-desktop.png`), fullPage: true
          });
        }
      }
      await resolve.waitFor({ timeout: 7000 });
      await resolve.click();
      await detail.getByText('已解决', { exact: true }).waitFor();
      await detail.getByRole('button', { name: '关闭确认', exact: true }).click();
      await detail.getByText('已关闭确认', { exact: true }).waitFor();
      await page.reload();
      await detail.getByText('已关闭确认', { exact: true }).waitFor();
      await page.waitForFunction(() => document.querySelector('[aria-label="维护证据"]')?.getAttribute('aria-busy') === 'false');
      const retained = await request(api, base, editor);
      assert.ok(retained.result_links.some((link) => link.includes(`source:${source}:`)));
      if (process.env.MAINTENANCE_SCREENSHOT_DIR) {
        await mkdir(process.env.MAINTENANCE_SCREENSHOT_DIR, { recursive: true });
        await page.screenshot({ path: join(process.env.MAINTENANCE_SCREENSHOT_DIR, `source-${severity}-closure-desktop.png`), fullPage: true });
      }
      await page.setViewportSize({ width: 390, height: 844 });
      await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth);
      if (process.env.MAINTENANCE_SCREENSHOT_DIR) {
        await page.screenshot({ path: join(process.env.MAINTENANCE_SCREENSHOT_DIR, `source-${severity}-closure-mobile.png`), fullPage: true });
      }
    });
}
}
