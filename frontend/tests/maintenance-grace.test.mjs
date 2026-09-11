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
  for (const integrity of [false, true]) {
  test(`accepted entry maintainer explicitly records ${integrity ? 'integrity' : 'freshness'} review (${built ? 'built' : 'development'})`,
    { timeout: 120000 }, async (t) => {
      const { page, baseUrl, api } = await startWorkbench(t, {
        built, env: {
          BROWSER_ACCEPTANCE_TICKET24: '1', BROWSER_ACCEPTANCE_TICKET27_SOURCE: '1',
          BROWSER_ACCEPTANCE_TICKET27_GRACE: '1'
        }
      });
      const editor = await registerKnowledgeUserViaApi(api, 'grace-browser-editor');
      const reporter = await registerKnowledgeUserViaApi(api, 'grace-browser-reporter');
      const admin = await request(api, '/auth/login', '', { username: 'operator', password: 'safe-password' });
      const owner = await request(api, '/auth/login', '', { username: 'ticket24-browser-maintainer', password: 'safe-password' });
      const assignment = await request(api, '/maintenance/assignments', admin.access_token, { username: 'grace-browser-editor' });
      await request(api, `/maintenance/assignments/${assignment.id}/accept`, editor, {});
      const entry = 'ticket24-browser-entry-001';
      const question = {
        message: 'Which Candidate publication contract applies? deployment=production',
        query_conditions: [{ condition_id: 'ticket24-production', field: 'deployment', operator: 'equals', value: 'production' }]
      };
      const result = await request(api, '/chat', reporter, { ...question, session_id: 'grace-browser-report' });
      assert.equal(result.outcome, 'evidence_gated_answer');
      const history = await request(api, '/sessions/grace-browser-report', reporter);
      const answer = history.messages.findLast((message) => message.type === 'assistant');
      const signal = await request(api, '/knowledge-feedback', reporter, { answer_id: answer.id, entry_id: entry, label: 'outdated' });
      const item = await request(api, '/maintenance/items', editor, {
        classification: integrity ? 'content-integrity' : 'source-freshness', severity: 'p2', disposition: 'needs-reproduction',
        coverage_position: 'evidence_sufficiency_refusal_and_acceptance',
        work_owner_username: 'ticket24-browser-maintainer', signal_ids: [signal.id]
      });
      await request(api, `/maintenance/items/${item.id}/transition`, editor, { expected_revision: 1, state: 'triaged' });
      const submissions = [];
      page.on('request', (event) => {
        if (event.method() === 'POST' && event.url().endsWith(integrity ? '/integrity-review' : '/freshness-review')) submissions.push(event);
      });
      await loginAdmin(page, baseUrl, { username: 'ticket24-browser-maintainer' });
      await page.getByRole('link', { name: '知识维护', exact: true }).click();
      const form = page.getByRole('form', { name: integrity ? '内容完整性复核' : '来源时效复核' });
      const button = form.getByRole('button', { name: integrity ? '确认完整性缺陷' : '登记来源复核', exact: true });
      const notice = integrity ? '完整性复核已登记' : '来源复核已登记';
      await button.waitFor({ timeout: 7000 });
      assert.equal(submissions.length, 0);
      if (integrity) {
        assert.equal(await button.isDisabled(), true);
        await form.getByRole('combobox', { name: '缺陷类别' }).selectOption('known_contradiction');
        await form.getByRole('checkbox', { name: '确认已独立复核当前发布修订和来源' }).check();
        assert.equal(submissions.length, 0);
      }
      await button.click();
      await form.getByText(notice, { exact: true }).waitFor();
      const first = await request(api, `/editorial/entries/${entry}`, owner.access_token);
      const retainedKey = integrity ? first.integrity_review.event_id : first.freshness_review.started_at;
      assert.ok(retainedKey);
      assert.equal(first.answer_eligible, !integrity);
      await page.reload();
      await button.waitFor({ timeout: 7000 });
      if (integrity) {
        assert.equal(await button.isDisabled(), true);
        await form.getByRole('checkbox', { name: '确认已独立复核当前发布修订和来源' }).check();
      }
      await button.click();
      await form.getByText(notice, { exact: true }).waitFor();
      const repeated = await request(api, `/editorial/entries/${entry}`, owner.access_token);
      assert.equal(integrity ? repeated.integrity_review.event_id : repeated.freshness_review.started_at, retainedKey);
      assert.equal(submissions.length, 2);
      const replay = await request(api, '/chat', owner.access_token, { ...question, session_id: 'grace-browser-supported' });
      assert.equal(replay.outcome, integrity ? 'insufficient_evidence_reply' : 'evidence_gated_answer');
      for (const [name, viewport] of [
        ['desktop', { width: 1440, height: 900 }], ['mobile', { width: 390, height: 844 }]
      ]) {
        await page.setViewportSize(viewport);
        await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth);
        if (process.env.MAINTENANCE_SCREENSHOT_DIR) {
          await mkdir(process.env.MAINTENANCE_SCREENSHOT_DIR, { recursive: true });
          await page.screenshot({
            path: join(process.env.MAINTENANCE_SCREENSHOT_DIR, `${integrity ? 'integrity' : 'grace'}-review-${name}.png`), fullPage: true
          });
        }
      }
    });
  }
}
