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

const failureEnvs = {
  provider: 'BROWSER_ACCEPTANCE_PROVIDER_FAIL_ON_CALL',
  citation: 'BROWSER_ACCEPTANCE_CITATION_FAILURE_ON_CALL',
  retrieval: 'BROWSER_ACCEPTANCE_RETRIEVAL_MISS_ON_CALL',
  condition: 'BROWSER_ACCEPTANCE_CONDITION_LOSS_ON_CALL',
  product: 'BROWSER_ACCEPTANCE_PRODUCT_FAILURE_ON_CALL'
};
const observations = {
  provider: 'provider_failure', citation: 'citation_drift', retrieval: 'retrieval_miss',
  condition: 'condition_loss', product: 'product_failure'
};
for (const failure of ['provider', 'citation', 'retrieval', 'condition', 'product']) {
for (const severity of failure === 'provider' ? ['p2'] : ['p2', 'p0', 'p1']) {
for (const built of [false, true]) {
  test(`maintainer confirms ${severity} ${failure} recovery against the replayed approved route (${built ? 'built' : 'development'})`,
    { timeout: 120000 }, async (t) => {
      const { page, baseUrl, api } = await startWorkbench(t, {
        built, env: {
          BROWSER_ACCEPTANCE_TICKET24: '1',
          [failureEnvs[failure]]: '3'
        }
      });
      const editor = await registerKnowledgeUserViaApi(api, 'provider-browser-editor');
      const worker = await registerKnowledgeUserViaApi(api, 'provider-browser-worker');
      const reporter = await registerKnowledgeUserViaApi(api, 'provider-browser-reporter');
      const admin = await request(api, '/auth/login', '', { username: 'operator', password: 'safe-password' });
      const assignment = await request(api, '/maintenance/assignments', admin.access_token, { username: 'provider-browser-editor' });
      await request(api, `/maintenance/assignments/${assignment.id}/accept`, editor, {});
      const answer = async (token, session) => {
        const result = await request(api, '/chat', token, {
          session_id: session, message: 'Which Candidate publication contract applies? deployment=production',
          query_conditions: [{ condition_id: 'ticket24-production', field: 'deployment', operator: 'equals', value: 'production' }]
        });
        assert.equal(result.outcome, 'evidence_gated_answer');
        const history = await request(api, `/sessions/${session}`, token);
        return history.messages.findLast((message) => message.type === 'assistant');
      };
      const reported = await answer(reporter, 'provider-browser-report');
      const independent = await answer(worker, 'provider-browser-independent');
      const signal = await request(api, '/knowledge-feedback', reporter, {
        answer_id: reported.id, entry_id: 'ticket24-browser-entry-001', label: 'insufficient_evidence'
      });
      const highSeverity = severity !== 'p2';
      const entry = 'ticket24-browser-entry-001';
      const containment = highSeverity
        ? await publicationAcceptance(request, api, admin.access_token, entry, undefined, `entry:${entry}`) : null;
      if (containment) {
        await request(api, `/acceptance/records/${containment.identity}/status`, admin.access_token, {
          status: 'suspended', reason_code: 'integrity_failure',
          status_failure: {
            check_id: 'check:entry-supported-query', reason: 'verified execution integrity failure', failure_kind: 'entry_specific',
            blocking_scope: { scope: 'entry_version', identity: `entry:${entry}` },
            evidence_links: ['evidence://maintenance/execution-integrity']
          }
        });
      }
      const item = await request(api, '/maintenance/items', editor, {
        classification: ['provider', 'product'].includes(failure) ? 'product-privacy-operations' : 'retrieval-answer-behavior',
        severity, disposition: 'needs-reproduction',
        coverage_position: failure !== 'provider' ? 'evidence_sufficiency_refusal_and_acceptance' : 'provider_failure_and_observability',
        work_owner_username: 'provider-browser-worker', signal_ids: [signal.id],
        ...(containment ? { containment_record_identity: containment.identity } : {})
      });
      const base = `/maintenance/items/${item.id}`;
      await request(api, `${base}/administrator`, admin.access_token, { expected_revision: 1 });
      await request(api, `${base}/transition`, editor, { expected_revision: 2, state: 'triaged' });
      const observation = observations[failure];
      const fixture = await request(api, `${base}/reproductions`, worker, {
        expected_revision: 3, answer_id: independent.id, signal_id: signal.id,
        expected_outcome: 'evidence_gated_answer', confirmed_synthetic_fixture: true, verified_observation: observation,
        ...(failure !== 'provider' ? { reference_answer_id: independent.id } : {})
      });
      const expectedOutcome = ['condition', 'product'].includes(failure) ? null
        : failure === 'retrieval' ? 'insufficient_evidence_reply' : 'generation_unavailable';
      assert.equal(fixture.observed_outcome, expectedOutcome);
      if (failure === 'condition') {
        assert.equal(fixture.observed_state, 'failed');
        assert.equal(fixture.diagnosis_facts.verified_difference, true);
      }
      if (failure === 'citation') assert.equal(fixture.diagnosis_facts.verified_difference, true);
      if (failure === 'retrieval') {
        assert.equal(fixture.diagnosis_facts.frozen_evidence, false);
        assert.equal(fixture.diagnosis_facts.reference_supported, true);
      }
      await request(api, `${base}/diagnosis`, editor, {
        expected_revision: 4, fixture_identity: fixture.id, observation
      });
      await request(api, `${base}/findings`, editor, { expected_revision: 5, fixture_identity: fixture.id });
      await request(api, `${base}/transition`, editor, { expected_revision: 6, state: 'in_progress' });
      const replay = await request(api, `${base}/replays`, worker, {
        expected_revision: 7, fixture_identity: fixture.id, answer_id: independent.id
      });
      assert.equal(replay.passed, true);
      assert.ok(replay.generation_context.activation_event_id);
      const reacceptance = highSeverity
        ? await publicationAcceptance(request, api, admin.access_token, entry, replay.id, `entry:${entry}`) : null;
      const submitted = [];
      page.on('request', (event) => {
        if (event.method() === 'POST' && event.url().endsWith('/resolution')) submitted.push(event.postDataJSON());
      });
      await loginAdmin(page, baseUrl, { username: 'provider-browser-editor' });
      await page.getByRole('link', { name: '知识维护', exact: true }).click();
      const detail = page.getByRole('region', { name: '维护项详情' });
      const command = failure === 'product' ? '确认产品修复'
        : failure === 'provider' ? '确认 Provider 恢复' : '确认检索实验结果';
      const resolve = detail.getByRole('button', { name: command, exact: true });
      if (reacceptance) {
        await page.waitForFunction(() => document.querySelector('[aria-label="维护证据"]')?.getAttribute('aria-busy') === 'false');
        const record = detail.getByLabel('重新验收记录标识', { exact: true });
        assert.equal(await record.count(), 1);
        assert.equal(await resolve.count(), 0);
        await record.fill(containment.identity);
        await resolve.click();
        await detail.getByRole('alert').waitFor();
        assert.equal((await request(api, base, editor)).state, 'in_progress');
        await record.fill(reacceptance.identity);
        submitted.length = 0;
      }
      await resolve.waitFor({ timeout: 7000 });
      assert.equal(submitted.length, 0);
      await resolve.click();
      await detail.getByText('已解决', { exact: true }).waitFor();
      if (reacceptance) assert.deepEqual(submitted[0].artifact_identities, [fixture.id, replay.id, reacceptance.identity]);
      await detail.getByRole('button', { name: '关闭确认', exact: true }).click();
      await detail.getByText('已关闭确认', { exact: true }).waitFor();
      await page.reload();
      await detail.getByText('已关闭确认', { exact: true }).waitFor();
      await detail.getByText('重放通过', { exact: true }).waitFor();
      await detail.getByText(replay.generation_context.route_identity, { exact: true }).waitFor();
      const retained = await request(api, base, editor);
      assert.ok(retained.result_links.some((link) => link.includes(
        failure !== 'provider' ? replay.id : replay.generation_context.activation_event_id
      )));
      if (reacceptance) assert.ok(retained.result_links.some((link) => link.includes(`${reacceptance.identity}:`)));
      assert.equal((await detail.innerText()).includes('provider_api_key'), false);
      if (process.env.MAINTENANCE_SCREENSHOT_DIR) {
        await mkdir(process.env.MAINTENANCE_SCREENSHOT_DIR, { recursive: true });
        await page.screenshot({ path: join(process.env.MAINTENANCE_SCREENSHOT_DIR, `${severity}-${failure}-closure-desktop.png`), fullPage: true });
      }
      await page.setViewportSize({ width: 390, height: 844 });
      await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth);
      if (process.env.MAINTENANCE_SCREENSHOT_DIR) {
        await page.screenshot({ path: join(process.env.MAINTENANCE_SCREENSHOT_DIR, `${severity}-${failure}-closure-mobile.png`), fullPage: true });
      }
    });
}
}
}
