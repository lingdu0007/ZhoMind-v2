// Reviewed Release Bundle workspace acceptance over a disposable real API.
// The UI can intake only immutable approved exports; it exposes no publication
// action and remains hidden from Knowledge Users.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { loginAdmin, registerKnowledgeUserViaApi, startWorkbench } from './acceptance-env.mjs';

const openReviewedBundlesWorkspace = async (page, baseUrl) => {
  await page.goto(`${baseUrl}reviewed-bundles`);
  await page.getByRole('heading', { name: 'Reviewed Release Bundles' }).waitFor();
  await page.locator('.reviewed-bundles__table-wrap').waitFor({ state: 'visible' });
  await page.getByText('尚未导入 Reviewed Release Bundle。').waitFor();
};

test('System Administrator can open immutable Reviewed Release Bundle intake without a publication control', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, { env: { BROWSER_ACCEPTANCE_SEED: 'minimal' } });
  await loginAdmin(page, baseUrl);
  await openReviewedBundlesWorkspace(page, baseUrl);

  assert.equal(await page.getByRole('link', { name: 'Reviewed Release Bundles' }).isVisible(), true);
  assert.equal(await page.getByRole('table').isVisible(), true);
  assert.equal(await page.getByRole('button', { name: '导入 Bundle' }).first().isVisible(), true);
  assert.equal(await page.getByRole('button', { name: '刷新' }).isVisible(), true);
  assert.equal(await page.getByRole('button', { name: /发布/ }).count(), 0);

  const [tableBox, importBox] = await Promise.all([
    page.getByRole('table').boundingBox(),
    page.getByRole('button', { name: '导入 Bundle' }).first().boundingBox()
  ]);
  assert.ok(tableBox.x + tableBox.width <= 1440);
  assert.ok(importBox.x + importBox.width <= 1440);
});

test('Knowledge User is redirected before Reviewed Release Bundle intake can render', { timeout: 30000 }, async (t) => {
  const { page, baseUrl, api } = await startWorkbench(t, {});
  const token = await registerKnowledgeUserViaApi(api, 'bundle-reader');
  await page.addInitScript(
    ({ storedToken }) => localStorage.setItem('access_token', storedToken),
    { storedToken: token }
  );

  await page.goto(`${baseUrl}reviewed-bundles`);
  await page.waitForURL(/\/chat\?notice=admin-required$/);
  assert.equal(await page.getByRole('heading', { name: 'Reviewed Release Bundles' }).count(), 0);
  assert.equal(await page.getByRole('link', { name: 'Reviewed Release Bundles' }).count(), 0);
});

test('Reviewed Release Bundle intake keeps its desktop-only boundary explicit on mobile', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, { viewport: { width: 390, height: 844 } });
  await loginAdmin(page, baseUrl);

  await page.goto(`${baseUrl}reviewed-bundles`);
  await page.getByRole('heading', { name: 'Reviewed Release Bundles' }).waitFor();
  assert.equal(await page.getByText('Reviewed Release Bundle 管理当前仅支持桌面工作区。').isVisible(), true);
  assert.equal(await page.getByRole('table').count(), 0);
  assert.equal(await page.getByRole('button', { name: '导入 Bundle' }).count(), 0);
  assert.equal(await page.getByRole('button', { name: '刷新' }).count(), 0);
});

test('Reviewed Release Bundle intake shows structured whole-bundle rejection reasons', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, { env: { BROWSER_ACCEPTANCE_SEED: 'minimal' } });
  await loginAdmin(page, baseUrl);
  await openReviewedBundlesWorkspace(page, baseUrl);

  await page.route('**/reviewed-release-bundles/import', async (route) => {
    await route.fulfill({
      status: 422,
      contentType: 'application/json',
      body: JSON.stringify({
        code: 'BUNDLE_INTEGRITY_REJECTED',
        message: 'bundle integrity validation failed',
        detail: {
          reasons: [
            {
              field: 'schema_version',
              code: 'invalid',
              message: 'schema_version must be a JSON integer'
            }
          ]
        }
      })
    });
  });

  await page.getByRole('button', { name: '导入 Bundle', exact: true }).first().click();
  await page.getByLabel('Bundle manifest').fill('{"schema":"reviewed_release_bundle/v1"}');
  await page.getByRole('button', { name: '导入 Bundle', exact: true }).last().click();

  await page.getByText('schema_version: schema_version must be a JSON integer').waitFor();
  assert.equal(await page.getByText('schema_version: schema_version must be a JSON integer').isVisible(), true);
});

test('Reviewed Release Bundle dispatch shows a server-returned enqueue failure instead of a success', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, { env: { BROWSER_ACCEPTANCE_SEED: 'minimal' } });
  const bundleId = 'bundle-ui-action-001';
  const jobId = 'job-ui-action-001';
  const bundle = {
    bundle_id: bundleId,
    state: 'processing',
    schema_version: 1,
    editorial_source_revision: 'a'.repeat(64),
    exported_at: '2026-09-06T12:00:00Z',
    bundle_sha256: 'b'.repeat(64),
    items: [
      {
        bundle_item_id: 'bundle_item:bundle-ui-action-item-001',
        entry_identity: 'entry:bundle-ui-action-entry-001',
        operation: 'create',
        state: 'admitted',
        artifact_sha256: 'c'.repeat(64),
        bundle_item_sha256: 'd'.repeat(64),
        allowed_next_action: 'dispatch_candidate_build',
        job_id: jobId
      }
    ]
  };
  const queuedJob = {
    job_id: jobId,
    bundle_id: `bundle:${bundleId}`,
    bundle_item_id: 'bundle_item:bundle-ui-action-item-001',
    entry_identity: 'entry:bundle-ui-action-entry-001',
    document_identity: 'document:candidate-bundle-ui-action-entry-001',
    requested_generation: 1,
    editorial_source_revision: 'a'.repeat(64),
    input_sha256: 'c'.repeat(64),
    chunk_strategy: {},
    embedding_configuration: { active: false },
    status: 'queued',
    stage: 'queued',
    progress: 0,
    attempt: 1,
    terminal_state: null,
    failure_reason: null,
    allowed_next_action: 'dispatch_candidate_build',
    candidate_id: null,
    derived_cleanup_pending: false,
    dispatched_at: null,
    started_at: null,
    heartbeat_at: null,
    lease_expires_at: null,
    completed_at: null,
    created_at: '2026-09-06T12:00:00Z',
    updated_at: '2026-09-06T12:00:00Z',
    events: []
  };
  const failedJob = {
    ...queuedJob,
    status: 'failed',
    terminal_state: 'failed',
    failure_reason: {
      code: 'CANDIDATE_ENQUEUE_FAILED',
      stage: 'queued',
      message: 'Candidate Build could not be returned to the queue'
    },
    allowed_next_action: 'retry_fixed_inputs',
    completed_at: '2026-09-06T12:00:01Z',
    updated_at: '2026-09-06T12:00:01Z'
  };

  await page.route('**/reviewed-release-bundles**', async (route) => {
    const request = route.request();
    const pathname = new URL(request.url()).pathname.replace(/\/+$/, '');
    if (request.method() === 'GET' && pathname.endsWith('/reviewed-release-bundles')) {
      await route.fulfill({ contentType: 'application/json', body: JSON.stringify({ data: { items: [bundle] } }) });
      return;
    }
    if (request.method() === 'GET' && pathname.endsWith(`/reviewed-release-bundles/${bundleId}`)) {
      await route.fulfill({ contentType: 'application/json', body: JSON.stringify({ data: bundle }) });
      return;
    }
    if (request.method() === 'GET' && pathname.endsWith(`/reviewed-release-bundles/jobs/${jobId}`)) {
      await route.fulfill({ contentType: 'application/json', body: JSON.stringify({ data: queuedJob }) });
      return;
    }
    if (request.method() === 'POST' && pathname.endsWith(`/reviewed-release-bundles/jobs/${jobId}/dispatch`)) {
      await route.fulfill({ contentType: 'application/json', body: JSON.stringify({ data: failedJob }) });
      return;
    }
    await route.fallback();
  });

  await loginAdmin(page, baseUrl);
  await page.goto(`${baseUrl}reviewed-bundles`);
  await page.getByRole('heading', { name: 'Reviewed Release Bundles' }).waitFor();
  await page.getByRole('button', { name: bundleId }).click();
  await page.getByRole('button', { name: `开始 Candidate Build ${jobId}` }).waitFor();
  await page.getByRole('button', { name: `开始 Candidate Build ${jobId}` }).click();

  await page.getByRole('alert').filter({ hasText: `Candidate Build ${jobId} 未能进入队列。` }).waitFor();
  assert.equal(
    await page
      .getByRole('alert')
      .filter({ hasText: `CANDIDATE_ENQUEUE_FAILED: Candidate Build could not be returned to the queue` })
      .isVisible(),
    true
  );
  assert.equal(await page.getByText(`Candidate Build ${jobId} 已进入队列。`, { exact: true }).count(), 0);
});

test('Reviewed Release Bundle acceptance runs inside the deterministic browser gate', async () => {
  const packageJson = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'));

  assert.equal(
    packageJson.scripts['test:reviewed-bundles'],
    'node --test --test-concurrency=1 tests/reviewed-bundles-workspace.test.mjs'
  );
  assert.match(packageJson.scripts['test:browser'], /npm run test:reviewed-bundles/);
});
