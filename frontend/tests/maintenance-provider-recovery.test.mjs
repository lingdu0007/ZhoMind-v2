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

const acceptRoute = async (api, token, original, replayIdentity) => {
  const payload = structuredClone(Object.fromEntries([
    'stage', 'affected_scope', 'content_identities', 'product_identities', 'conditions',
    'assumptions', 'checks', 'known_limits', 'risks', 'evidence_links', 'reacceptance_triggers'
  ].map((key) => [key, original[key]])));
  if (replayIdentity) {
    const link = `evidence://maintenance/artifacts/${replayIdentity}`;
    payload.evidence_links.push(link);
    payload.checks.find((check) => check.check_id === 'check:generation-privacy').evidence_links.push(link);
  }
  const record = await request(api, '/acceptance/records', token, payload);
  return request(api, `/acceptance/records/${record.record_id}/status`, token, {
    status: 'active', reason_code: 'checks_verified',
    verified_checks: record.checks.filter((check) => ['passed', 'carried_forward'].includes(check.result))
      .map(({ check_id, evidence_links }) => ({ check_id, evidence_links }))
  });
};

const ready = (page) => page.waitForFunction(() =>
  document.querySelector('[aria-label="维护证据"]')?.getAttribute('aria-busy') === 'false'
  && document.querySelector('.maintenance')?.getAttribute('aria-busy') === 'false');

for (const severity of ['p0', 'p1']) {
  for (const built of [false, true]) {
    test(`explicit ${severity} Provider authorization, replay and repair closure (${built ? 'built' : 'development'})`,
      { timeout: 180000 }, async (t) => {
        const { page, baseUrl, api } = await startWorkbench(t, {
          built, env: { BROWSER_ACCEPTANCE_TICKET24: '1', BROWSER_ACCEPTANCE_PROVIDER_FAIL_ON_CALL: '3' }
        });
        const editorName = 'recovery-browser-editor';
        const workerName = 'recovery-browser-worker';
        const editor = await registerKnowledgeUserViaApi(api, editorName);
        const worker = await registerKnowledgeUserViaApi(api, workerName);
        const reporter = await registerKnowledgeUserViaApi(api, 'recovery-browser-reporter');
        const admin = (await request(api, '/auth/login', '', { username: 'operator', password: 'safe-password' })).access_token;
        const assignment = await request(api, '/maintenance/assignments', admin, { username: editorName });
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
        const reported = await answer(reporter, 'recovery-browser-report');
        const independent = await answer(worker, 'recovery-browser-independent');
        const signal = await request(api, '/knowledge-feedback', reporter, {
          answer_id: reported.id, entry_id: 'ticket24-browser-entry-001', label: 'insufficient_evidence'
        });
        const route = (await request(api, '/settings/generation-route', admin)).active;
        const containment = route.providers[0].validation_evidence.record_identity;
        const original = await request(api, `/acceptance/records/${containment}`, admin);
        await request(api, `/acceptance/records/${containment}/status`, admin, {
          status: 'suspended', reason_code: 'integrity_failure',
          status_failure: {
            check_id: 'check:generation-privacy', reason: 'independently verified shared provider privacy failure',
            failure_kind: 'shared_privacy',
            blocking_scope: { scope: 'collection', identity: 'collection:production-rag-agent-engineering' },
            evidence_links: ['evidence://maintenance/provider-browser-containment']
          }
        });
        const item = await request(api, '/maintenance/items', editor, {
          classification: 'product-privacy-operations', severity, disposition: 'needs-reproduction',
          coverage_position: 'provider_failure_and_observability', work_owner_username: workerName,
          signal_ids: [signal.id], containment_record_identity: containment
        });
        const base = `/maintenance/items/${item.id}`;
        await request(api, `${base}/transition`, editor, { expected_revision: 1, state: 'triaged' });
        const admission = await acceptRoute(api, admin, original);
        const writes = [];
        page.on('request', (outgoing) => {
          if (outgoing.method() === 'POST' && decodeURIComponent(outgoing.url()).includes(base)) writes.push(outgoing);
        });
        let loggedIn = false;
        const enter = async (username) => {
          if (loggedIn) {
            await page.getByRole('button', { name: '退出登录' }).click();
            await page.getByRole('heading', { name: '身份验证' }).waitFor();
          }
          await loginAdmin(page, baseUrl, { username });
          loggedIn = true;
          await page.getByRole('link', { name: '知识维护', exact: true }).click();
          await ready(page);
        };
        const detail = page.getByRole('region', { name: '维护项详情' });
        const submit = async (button, suffix, status = 200) => {
          const pending = page.waitForResponse((response) =>
            response.request().method() === 'POST' && decodeURIComponent(response.url()).endsWith(`${base}${suffix}`));
          await button.click();
          const response = await pending;
          const result = await response.json();
          assert.equal(response.status(), status, JSON.stringify(result));
          await ready(page);
          return result.data;
        };
        const sign = async () => {
          await enter('operator');
          const form = detail.getByRole('form', { name: 'Provider 验证授权', exact: true });
          await form.waitFor({ timeout: 7000 });
          const count = writes.length;
          await form.getByLabel('验证路由标识', { exact: true }).fill(route.route_identity);
          await form.getByLabel('调用准入验收记录', { exact: true }).fill(admission.record_id);
          await form.getByLabel('授权查询条件集', { exact: true }).fill(independent.answer_execution.query_condition_set.identity);
          const command = form.getByRole('button', { name: '签发验证授权', exact: true });
          assert.equal(await command.isDisabled(), true);
          await form.getByLabel('确认仅授权所列非个人场景', { exact: true }).check();
          assert.equal(writes.length, count);
          const grant = await submit(command, '/provider-verification-authorizations');
          await detail.getByText(grant.id, { exact: true }).waitFor();
          assert.equal(await form.getByLabel('确认仅授权所列非个人场景', { exact: true }).isChecked(), false);
          assert.equal((await request(api, '/settings/generation-route', admin)).active, null);
          await snapshot(`authorization-${grant.item_revision}`);
          return grant.id;
        };
        const selectGrant = async (identity) => {
          await enter(workerName);
          const region = detail.getByRole('region', { name: 'Provider 验证授权', exact: true });
          await region.getByLabel('验证授权标识', { exact: true }).fill(identity);
          const count = writes.length;
          await region.getByRole('button', { name: '核验授权', exact: true }).click();
          await region.getByText('授权已核验', { exact: true }).waitFor();
          assert.equal(writes.length, count);
          await snapshot('authorization-selected');
        };
        const grant = await sign();
        await selectGrant(grant);
        await page.reload();
        await ready(page);
        const authorizationRegion = detail.getByRole('region', { name: 'Provider 验证授权', exact: true });
        assert.equal(await authorizationRegion.getByLabel('验证授权标识', { exact: true }).inputValue(), '');
        await authorizationRegion.getByLabel('验证授权标识', { exact: true }).fill(grant);
        await authorizationRegion.getByRole('button', { name: '核验授权', exact: true }).click();
        await authorizationRegion.getByText('授权已核验', { exact: true }).waitFor();
        const reproduction = detail.getByRole('form', { name: '独立复现', exact: true });
        await reproduction.getByLabel('已保存的本人执行', { exact: true }).selectOption(independent.id);
        await reproduction.getByLabel('反馈目标', { exact: true }).selectOption(signal.id);
        await reproduction.getByLabel('预期结果', { exact: true }).selectOption('evidence_gated_answer');
        await reproduction.getByLabel('验证观察', { exact: true }).selectOption('provider_failure');
        await reproduction.getByLabel('确认这是独立编写、可共享的非个人场景', { exact: true }).check();
        const fixture = await submit(
          reproduction.getByRole('button', { name: '登记并运行复现', exact: true }), '/reproductions'
        );
        assert.equal(fixture.observed_outcome, 'generation_unavailable');
        assert.equal(fixture.generation_context.authorization_identity, grant);
        await enter(editorName);
        await submit(detail.getByRole('button', { name: '确认诊断', exact: true }), '/diagnosis');
        await submit(detail.getByRole('button', { name: '批准非个人发现', exact: true }), '/findings');
        await submit(detail.getByRole('button', { name: '开始处理', exact: true }), '/transition');
        const deleted = await fetch(`${api.baseUrl}/knowledge-feedback/${signal.id}`, {
          method: 'DELETE', headers: { Authorization: `Bearer ${reporter}` }
        });
        assert.equal(deleted.status, 200);
        assert.equal((await request(api, base, worker)).signal_count, 0);
        await enter(workerName);
        await authorizationRegion.getByLabel('验证授权标识', { exact: true }).fill(grant);
        await authorizationRegion.getByRole('button', { name: '核验授权', exact: true }).click();
        await authorizationRegion.getByText('授权与当前维护修订不匹配。', { exact: true }).waitFor();
        assert.equal(await detail.getByRole('button', { name: '重放已声明场景', exact: true }).isDisabled(), true);
        const repairGrant = await sign();
        assert.notEqual(repairGrant, grant);
        const grantRecord = await request(api, `/maintenance/provider-verification-authorizations/${repairGrant}`, worker);
        assert.equal(grantRecord.approved_fixture_identity, fixture.id);
        await selectGrant(repairGrant);
        await detail.getByLabel('重放使用的本人执行', { exact: true }).selectOption(independent.id);
        const replay = await submit(
          detail.getByRole('button', { name: '重放已声明场景', exact: true }), '/replays'
        );
        assert.equal(replay.passed, true);
        assert.equal(replay.generation_context.authorization_identity, repairGrant);
        assert.equal((await request(api, '/settings/generation-route', admin)).active, null);
        const repairAcceptance = await acceptRoute(api, admin, original, replay.id);
        await request(api, '/settings/generation-route/activate', admin, {
          route_identity: route.route_identity, expected_active_identity: route.route_identity,
          acceptance_record_identity: repairAcceptance.record_id
        });
        await enter(editorName);
        const record = detail.getByLabel('重新验收记录标识', { exact: true });
        await record.waitFor();
        assert.equal(await detail.getByText('路由验收', { exact: true }).count(), 0);
        await detail.getByText('调用准入', { exact: true }).waitFor();
        await detail.getByText(repairGrant, { exact: true }).waitFor();
        const resolve = detail.getByRole('button', { name: '确认 Provider 恢复', exact: true });
        assert.equal(await resolve.count(), 0);
        await record.fill(admission.record_id);
        await submit(resolve, '/resolution', 409);
        await record.fill(repairAcceptance.record_id);
        async function snapshot(stage) {
          for (const [name, viewport] of [
            ['desktop', { width: 1440, height: 900 }], ['mobile', { width: 390, height: 844 }]
          ]) {
            await page.setViewportSize(viewport);
            await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth);
            if (process.env.MAINTENANCE_SCREENSHOT_DIR) {
              await mkdir(process.env.MAINTENANCE_SCREENSHOT_DIR, { recursive: true });
              await page.screenshot({
                path: join(process.env.MAINTENANCE_SCREENSHOT_DIR, `provider-${severity}-${stage}-${name}.png`), fullPage: true
              });
            }
          }
        }
        await snapshot('repair-ready');
        await submit(resolve, '/resolution');
        const resolutionBody = writes.findLast((outgoing) => outgoing.url().endsWith('/resolution')).postDataJSON();
        assert.deepEqual(resolutionBody.artifact_identities, [
          fixture.id, replay.id, route.route_identity, repairAcceptance.record_id
        ]);
        await submit(detail.getByRole('button', { name: '关闭确认', exact: true }), '/transition');
        await page.reload();
        await ready(page);
        await detail.getByText('已关闭确认', { exact: true }).waitFor();
        await detail.getByText('重放通过', { exact: true }).waitFor();
        const retained = await request(api, base, worker);
        assert.ok(retained.result_links.some((link) => link.includes(`${repairAcceptance.record_id}:`)));
        assert.equal((await detail.innerText()).includes('provider_api_key'), false);
        await snapshot('closure');
      });
  }
}
