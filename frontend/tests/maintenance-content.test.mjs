import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import test from 'node:test';
import { loginAdmin, registerKnowledgeUserViaApi, startWorkbench } from './acceptance-env.mjs';
import { publicationAcceptance } from './maintenance-acceptance.mjs';

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
const ordered = (value) => Array.isArray(value) ? value.map(ordered)
  : value && typeof value === 'object' ? Object.fromEntries(Object.keys(value).sort().map((key) => [key, ordered(value[key])])) : value;
const digest = (value) => createHash('sha256').update(JSON.stringify(ordered(value))
  .replace(/[^\x00-\x7f]/g, (character) => `\\u${character.charCodeAt(0).toString(16).padStart(4, '0')}`)).digest('hex');
const manifest = (artifact) => {
  const item = { bundle_item_id: 'content-browser-repair-item', operation: 'replace', artifact_sha256: digest(artifact), artifact };
  item.bundle_item_sha256 = digest(item);
  const bundle = {
    schema: 'reviewed_release_bundle/v1', schema_version: 1, bundle_id: 'content-browser-repair',
    editorial_source_revision: artifact.revision_sha256, exported_at: '2026-09-11T00:00:00Z', items: [item]
  };
  return { ...bundle, bundle_sha256: digest(bundle) };
};
const captureState = async (page, state) => {
  for (const [name, viewport] of [['desktop', { width: 1440, height: 900 }], ['mobile', { width: 390, height: 844 }]]) {
    await page.setViewportSize(viewport);
    await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth);
    if (process.env.MAINTENANCE_SCREENSHOT_DIR) {
      await mkdir(process.env.MAINTENANCE_SCREENSHOT_DIR, { recursive: true });
      await page.screenshot({ path: join(process.env.MAINTENANCE_SCREENSHOT_DIR, `content-${state}-${name}.png`), fullPage: true });
    }
  }
};

for (const scenario of ['p2', 'p0', 'p1', 'coverage']) {
for (const built of [false, true]) {
  test(`maintainer closes a separately approved and published ${scenario} content repair (${built ? 'built' : 'development'})`,
    { timeout: 120000 }, async (t) => {
      const { page, baseUrl, api } = await startWorkbench(t, {
        built, env: {
          BROWSER_ACCEPTANCE_TICKET24: '1', BROWSER_ACCEPTANCE_TICKET27_SOURCE: '1',
          BROWSER_ACCEPTANCE_TICKET27_GRACE: '1', BROWSER_ACCEPTANCE_TICKET27_CONTENT: '1'
        }
      });
      const editor = await registerKnowledgeUserViaApi(api, 'content-browser-editor');
      const worker = await registerKnowledgeUserViaApi(api, 'content-browser-worker');
      const reporter = await registerKnowledgeUserViaApi(api, 'content-browser-reporter');
      const login = async (username) => (await request(api, '/auth/login', '', { username, password: 'safe-password' })).access_token;
      const admin = await login('operator');
      const owner = await login('ticket24-browser-maintainer');
      const reviewer = await login('ticket24-browser-reviewer');
      const assignment = await request(api, '/maintenance/assignments', admin, { username: 'content-browser-editor' });
      await request(api, `/maintenance/assignments/${assignment.id}/accept`, editor, {});
      const entry = 'ticket24-browser-entry-001';
      const editorial = `/editorial/entries/${entry}`;
      const coverage = scenario === 'coverage';
      const severity = coverage ? 'p2' : scenario;
      const observation = coverage ? 'coverage_gap' : 'wrong_content';
      if (coverage) await request(api, `${editorial}/sources/source-${entry}/availability`, owner, {
        availability: 'unavailable_for_new_evidence'
      });
      const answer = async (token, session) => {
        const result = await request(api, '/chat', token, {
          session_id: session, message: 'Which Candidate publication contract applies? deployment=production',
          query_conditions: [{ condition_id: 'ticket24-production', field: 'deployment', operator: 'equals', value: 'production' }]
        });
        assert.equal(result.outcome, coverage ? 'insufficient_evidence_reply' : 'evidence_gated_answer');
        const history = await request(api, `/sessions/${session}`, token);
        return history.messages.findLast((message) => message.type === 'assistant');
      };
      const reported = await answer(reporter, 'content-browser-report');
      const independent = await answer(worker, 'content-browser-independent');
      const signal = await request(api, '/knowledge-feedback', reporter, {
        answer_id: reported.id, ...(coverage ? {} : { entry_id: entry }), label: 'insufficient_evidence'
      });
      const highSeverity = severity !== 'p2';
      const containment = highSeverity
        ? await publicationAcceptance(request, api, admin, entry, undefined, `entry:${entry}`) : null;
      if (containment) {
        await request(api, `/acceptance/records/${containment.identity}/status`, admin, {
          status: 'suspended', reason_code: 'integrity_failure',
          status_failure: {
            check_id: 'check:entry-supported-query', reason: 'verified content integrity failure', failure_kind: 'entry_specific',
            blocking_scope: { scope: 'entry_version', identity: `entry:${entry}` },
            evidence_links: ['evidence://maintenance/content-integrity']
          }
        });
      }
      const item = await request(api, '/maintenance/items', editor, {
        classification: coverage ? 'coverage-gap' : 'content-integrity', severity, disposition: 'needs-reproduction',
        coverage_position: 'evidence_sufficiency_refusal_and_acceptance',
        work_owner_username: 'content-browser-worker', signal_ids: [signal.id],
        ...(containment ? { containment_record_identity: containment.identity } : {})
      });
      const base = `/maintenance/items/${item.id}`;
      await request(api, `${base}/transition`, editor, { expected_revision: 1, state: 'triaged' });
      const original = await request(api, editorial, owner);
      if (!coverage) await request(api, `${editorial}/integrity-review`, owner, {
        revision_identity: original.revision_identity, publication_identity: original.integrity_review.publication_identity,
        source_identity: original.integrity_review.source_identities[0], defect: 'integrity_defect', confirmed_independent_review: true
      });
      const fixture = await request(api, `${base}/reproductions`, worker, {
        expected_revision: 2, answer_id: independent.id, signal_id: signal.id,
        expected_outcome: 'evidence_gated_answer', confirmed_synthetic_fixture: true,
        verified_observation: observation, ...(coverage ? {} : { entry_identity: `entry:${entry}` })
      });
      await request(api, `${base}/diagnosis`, editor, { expected_revision: 3, fixture_identity: fixture.id, observation });
      await request(api, `${base}/findings`, editor, { expected_revision: 4, fixture_identity: fixture.id });
      await request(api, `${base}/transition`, editor, { expected_revision: 5, state: 'in_progress' });
      if (coverage) await request(api, `${editorial}/sources/source-${entry}/availability`, owner, {
        availability: 'verified_usable'
      });
      original.entry.body.recommendation_or_reviewed_branches += ' Independent repair review preserves explicit administrator publication confirmation.';
      await request(api, `${editorial}/revisions`, owner, { entry: original.entry, change_kind: 'material' });
      await request(api, `${editorial}/maintainer-acceptance`, owner, {});
      await request(api, `${editorial}/approve`, reviewer, {});
      const exported = await request(api, `${editorial}/export`, admin, {});
      const imported = await request(api, '/reviewed-release-bundles/import', admin, manifest(exported.artifact));
      const jobPath = `/reviewed-release-bundles/jobs/${imported.items[0].job_id}`;
      let job = await request(api, `${jobPath}/dispatch`, admin, {});
      const deadline = Date.now() + 20000;
      while (['queued', 'running'].includes(job.status) && Date.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, 100));
        job = await request(api, jobPath, admin);
      }
      assert.equal(job.status, 'candidate_ready', JSON.stringify(job));
      const candidatePath = `/reviewed-release-bundles/candidates/${job.candidate_id}`;
      await request(api, `${candidatePath}/inspection`, admin, {});
      await request(api, `${candidatePath}/acceptance`, admin, {});
      const eligibility = await request(api, `${candidatePath}/publication-eligibility`, admin);
      const publication = await request(api, '/reviewed-release-bundles/publication-batches', admin, {
        confirmation_id: 'content-browser-repair',
        selected_items: [{
          candidate_id: job.candidate_id, effect: eligibility.effect,
          current_published_knowledge_version: eligibility.current_published_knowledge_version.identity,
          inspection_record_identity: eligibility.inspection_record_identity, acceptance_record_identity: eligibility.acceptance_record_identity
        }]
      });
      assert.equal(publication.batch_complete, true);
      const replay = await request(api, `${base}/replays`, worker, {
        expected_revision: 6, fixture_identity: fixture.id, answer_id: independent.id
      });
      assert.equal(replay.passed, true);
      const reacceptance = highSeverity
        ? await publicationAcceptance(request, api, admin, entry, replay.id, `entry:${entry}`) : null;
      const submitted = [];
      page.on('request', (event) => {
        if (event.method() === 'POST' && event.url().endsWith('/resolution')) submitted.push(event.postDataJSON());
      });
      await loginAdmin(page, baseUrl, { username: 'content-browser-editor' });
      await page.getByRole('link', { name: '知识维护', exact: true }).click();
      const detail = page.getByRole('region', { name: '维护项详情' });
      const resolve = detail.getByRole('button', { name: coverage ? '确认覆盖补齐' : '确认内容修复', exact: true });
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
      const reproduction = detail.getByRole('region', { name: '认证复现结果' });
      assert.equal(await reproduction.getByText('复现时发布版本', { exact: true }).count(), 1);
      assert.equal(await reproduction.getByText('当前发布版本', { exact: true }).count(), 0);
      assert.equal(submitted.length, 0);
      const repairedPublication = coverage
        ? (await request(api, `/maintenance/replays/${replay.id}`, editor)).evidence_publications[0]
        : replay.publication_review;
      if (coverage) {
        const verification = detail.getByRole('region', { name: '修复后验证' });
        await verification.getByText(repairedPublication.publication_identity, { exact: true }).waitFor();
        await verification.getByText(repairedPublication.revision_identity, { exact: true }).waitFor();
      }
      await captureState(page, `${scenario}-repair-ready`);
      await resolve.click();
      await detail.getByText('已解决', { exact: true }).waitFor();
      assert.deepEqual(submitted[0].artifact_identities, [
        fixture.id, replay.id,
        ...(coverage ? [repairedPublication.publication_identity, repairedPublication.revision_identity]
          : [repairedPublication.revision_identity, repairedPublication.publication_identity]),
        ...(reacceptance ? [reacceptance.identity] : [])
      ]);
      await detail.getByRole('button', { name: '关闭确认', exact: true }).click();
      await detail.getByText('已关闭确认', { exact: true }).waitFor();
      await page.reload();
      await detail.getByText('已关闭确认', { exact: true }).waitFor();
      const retained = await request(api, base, editor);
      assert.ok(retained.result_links.includes(`evidence://maintenance/artifacts/${repairedPublication.publication_identity}`));
      if (reacceptance) assert.ok(retained.result_links.some((link) => link.includes(`${reacceptance.identity}:`)));
      await captureState(page, `${scenario}-closure`);
    });
}
}
