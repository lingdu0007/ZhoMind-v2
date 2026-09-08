// Reviewed Release Bundle workspace acceptance over a disposable real API.
// Intake, inspection, acceptance and explicit publication remain separate
// administrator actions and stay hidden from Knowledge Users.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { loginAdmin, registerKnowledgeUserViaApi, startWorkbench } from './acceptance-env.mjs';

const ticket24ReplacementBundle = 'ticket24-browser-replacement';
const ticket24FailureBundle = 'ticket24-browser-failure';

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

test('System Administrator inspects, accepts, selects, and explicitly confirms a Candidate publication batch', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, { env: { BROWSER_ACCEPTANCE_SEED: 'minimal' } });
  const bundleId = 'bundle-ticket24-browser-001';
  const jobId = 'job-ticket24-browser-001';
  const candidateId = 'candidate:job-ticket24-browser-001-attempt-1';
  const configurationIdentity = 'configuration:candidate-build-browser-001';
  let inspected = false;
  let accepted = false;
  const submittedBatches = [];
  const bundle = {
    bundle_id: bundleId,
    state: 'processing',
    schema_version: 1,
    editorial_source_revision: 'a'.repeat(64),
    exported_at: '2026-09-06T12:00:00Z',
    bundle_sha256: 'b'.repeat(64),
    items: [
      {
        bundle_item_id: 'bundle_item:ticket24-browser-item-001',
        entry_identity: 'entry:ticket24-browser-entry-001',
        operation: 'create',
        state: 'admitted',
        artifact_sha256: 'c'.repeat(64),
        bundle_item_sha256: 'd'.repeat(64),
        allowed_next_action: 'await_candidate_inspection',
        job_id: jobId
      }
    ]
  };
  const candidateDetail = () => ({
    candidate: {
      candidate_id: candidateId,
      entry_identity: 'entry:ticket24-browser-entry-001',
      document_identity: 'runtime-document:ticket24-browser-entry-001',
      bundle_id: `bundle:${bundleId}`,
      bundle_item_id: 'bundle_item:ticket24-browser-item-001',
      bundle_sha256: 'b'.repeat(64),
      bundle_item_sha256: 'd'.repeat(64),
      editorial_source_revision: 'a'.repeat(64),
      input_sha256: 'c'.repeat(64),
      frozen_input_sha256: 'e'.repeat(64),
      generation: 1,
      configuration_identity: configurationIdentity,
      configuration: {
        schema: 'candidate_embedding_configuration/v1',
        active: true,
        model: 'ticket24-embedding',
        dimension: 768
      },
      chunks: [
        {
          chunk_id: 'ticket24-browser-chunk-001',
          chunk_index: 0,
          content_sha256: 'f'.repeat(64),
          content: 'Only an explicitly confirmed Candidate may become published.',
          metadata: {
            entry_id: 'ticket24-browser-entry-001',
            entry_identity: 'entry:ticket24-browser-entry-001',
            editorial_revision_identity: 'editorial_revision:ticket24-browser-entry-001.r1',
            section_id: 'recommendation_or_reviewed_branches',
            section_title: 'Recommendation Or Reviewed Branches',
            chunk_strategy_id: 'section-aware-900-120',
            source_identities: ['source:ticket24-browser-source-001'],
            source_relationships: [
              {
                source_identity: 'source:ticket24-browser-source-001',
                availability: 'verified_usable',
                access_scope: 'public'
              }
            ],
            candidate_build: true
          }
        }
      ]
    },
    inspection: inspected
      ? {
          record_identity: 'event:ticket24-browser-inspection-001',
          candidate_id: candidateId,
          frozen_input_sha256: 'e'.repeat(64),
          configuration_identity: configurationIdentity,
          inspected_by: 'member:operator'
        }
      : null,
    replacement: {
      effect: 'create',
      current_published_knowledge_version: null,
      diff: {
        schema: 'candidate_replacement_diff/v1',
        added: [{ chunk_index: 0, candidate_content_sha256: 'f'.repeat(64) }],
        changed: [],
        removed: []
      }
    }
  });
  const candidateJob = () => ({
    job_id: jobId,
    bundle_id: `bundle:${bundleId}`,
    bundle_item_id: 'bundle_item:ticket24-browser-item-001',
    entry_identity: 'entry:ticket24-browser-entry-001',
    document_identity: 'runtime-document:ticket24-browser-entry-001',
    requested_generation: 1,
    editorial_source_revision: 'a'.repeat(64),
    input_sha256: 'c'.repeat(64),
    frozen_input_sha256: 'e'.repeat(64),
    chunk_strategy: { strategy_id: 'section-aware-900-120' },
    embedding_configuration: { active: false },
    status: 'candidate_ready',
    stage: 'indexing',
    progress: 100,
    attempt: 1,
    terminal_state: 'candidate_ready',
    failure_reason: null,
    allowed_next_action: accepted ? 'await_explicit_publication' : 'await_candidate_inspection',
    candidate_id: candidateId,
    derived_cleanup_pending: false,
    dispatched_at: null,
    started_at: null,
    heartbeat_at: null,
    lease_expires_at: null,
    completed_at: '2026-09-06T12:01:00Z',
    created_at: '2026-09-06T12:00:00Z',
    updated_at: '2026-09-06T12:01:00Z',
    events: []
  });
  const eligibility = () => ({
    candidate_id: candidateId,
    eligible: accepted,
    reasons: accepted ? [] : inspected ? ['CANDIDATE_ACCEPTANCE_REQUIRED'] : ['CANDIDATE_INSPECTION_REQUIRED'],
    effect: 'create',
    current_published_knowledge_version: null,
    configuration_identity: configurationIdentity,
    generation: 1,
    inspection_record_identity: 'event:ticket24-browser-inspection-001',
    acceptance_record_identity: 'event:ticket24-browser-acceptance-001'
  });

  await page.route('**/reviewed-release-bundles**', async (route) => {
    const request = route.request();
    const pathname = decodeURIComponent(new URL(request.url()).pathname.replace(/\/+$/, ''));
    if (request.method() === 'GET' && pathname.endsWith('/reviewed-release-bundles')) {
      await route.fulfill({ contentType: 'application/json', body: JSON.stringify({ data: { items: [bundle] } }) });
      return;
    }
    if (request.method() === 'GET' && pathname.endsWith(`/reviewed-release-bundles/${bundleId}`)) {
      await route.fulfill({ contentType: 'application/json', body: JSON.stringify({ data: bundle }) });
      return;
    }
    if (request.method() === 'GET' && pathname.endsWith(`/reviewed-release-bundles/jobs/${jobId}`)) {
      await route.fulfill({ contentType: 'application/json', body: JSON.stringify({ data: candidateJob() }) });
      return;
    }
    if (request.method() === 'GET' && pathname.endsWith(`/candidates/${candidateId}/inspection`)) {
      await route.fulfill({ contentType: 'application/json', body: JSON.stringify({ data: candidateDetail() }) });
      return;
    }
    if (request.method() === 'POST' && pathname.endsWith(`/candidates/${candidateId}/inspection`)) {
      inspected = true;
      await route.fulfill({ contentType: 'application/json', body: JSON.stringify({ data: candidateDetail() }) });
      return;
    }
    if (request.method() === 'POST' && pathname.endsWith(`/candidates/${candidateId}/acceptance`)) {
      accepted = true;
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({
          data: {
            record_identity: 'event:ticket24-browser-acceptance-001',
            candidate_id: candidateId,
            supported: { outcome: 'evidence_gated_answer' },
            boundary: { outcome: 'insufficient_evidence_reply' }
          }
        })
      });
      return;
    }
    if (request.method() === 'GET' && pathname.endsWith(`/candidates/${candidateId}/publication-eligibility`)) {
      await route.fulfill({ contentType: 'application/json', body: JSON.stringify({ data: eligibility() }) });
      return;
    }
    if (request.method() === 'POST' && pathname.endsWith('/reviewed-release-bundles/publication-batches')) {
      submittedBatches.push(request.postDataJSON());
      if (submittedBatches.length === 1) {
        await route.fulfill({
          status: 409,
          contentType: 'application/json',
          body: JSON.stringify({
            code: 'PUBLICATION_CONFIRMATION_IN_PROGRESS',
            message: 'the selected publication confirmation is already processing'
          })
        });
        return;
      }
      if (submittedBatches.length === 2) {
        await route.fulfill({
          status: 409,
          contentType: 'application/json',
          body: JSON.stringify({
            code: 'PUBLICATION_CONFIRMATION_POINTER_MISMATCH',
            message: 'the selected publication confirmation no longer binds the inspected replacement version'
          })
        });
        return;
      }
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({
          data: {
            batch_complete: true,
            published: [
              {
                candidate_id: candidateId,
                effect: 'create',
                publication_identity: 'published_knowledge_version:ticket24-browser-001'
              }
            ],
            failed: [],
            skipped: []
          }
        })
      });
      return;
    }
    await route.fallback();
  });

  await loginAdmin(page, baseUrl);
  await page.goto(`${baseUrl}reviewed-bundles`);
  await page.getByRole('button', { name: bundleId }).click();
  await page.getByRole('button', { name: `查看 Candidate ${candidateId}` }).click();
  await page.getByRole('heading', { name: candidateId }).waitFor();
  assert.equal(await page.getByText('创建新的 Published Knowledge Version').isVisible(), true);
  assert.equal(await page.getByText('Candidate chunks').isVisible(), true);
  assert.equal(await page.getByText(/"model": "ticket24-embedding"/).isVisible(), true);
  assert.equal(await page.getByText('section-aware-900-120', { exact: true }).isVisible(), true);
  assert.equal(await page.getByText('source:ticket24-browser-source-001', { exact: true }).isVisible(), true);

  await page.getByRole('button', { name: `记录 Candidate inspection ${candidateId}` }).click();
  await page.getByRole('button', { name: `执行 Candidate 验收 ${candidateId}` }).waitFor();
  await page.getByRole('button', { name: `执行 Candidate 验收 ${candidateId}` }).click();
  await page.getByText('可发布').waitFor();

  await page.locator('.reviewed-bundles__candidate-panel input[type="checkbox"]').check();
  await page.getByRole('button', { name: '发布已选择 1 项' }).click();
  await page.getByRole('heading', { name: '确认发布 Candidate' }).waitFor();
  await page.locator('.reviewed-bundles__publication-dialog input[type="checkbox"]').check();
  await page.getByRole('button', { name: '确认发布' }).click();
  await page.getByRole('button', { name: '发布已选择 1 项' }).waitFor();
  await page.locator('.reviewed-bundles__publication-dialog').getByRole('button', { name: '取消' }).click();
  await page.getByRole('button', { name: '发布已选择 1 项' }).click();
  await page.getByRole('heading', { name: '确认发布 Candidate' }).waitFor();
  await page.locator('.reviewed-bundles__publication-dialog input[type="checkbox"]').check();
  await page.getByRole('button', { name: '确认发布' }).click();
  await page.getByRole('button', { name: '发布已选择 1 项' }).waitFor();
  await page.locator('.reviewed-bundles__publication-dialog').getByRole('button', { name: '取消' }).click();
  await page.getByRole('button', { name: '发布已选择 1 项' }).click();
  await page.getByRole('heading', { name: '确认发布 Candidate' }).waitFor();
  await page.locator('.reviewed-bundles__publication-dialog input[type="checkbox"]').check();
  await page.getByRole('button', { name: '确认发布' }).click();
  await page.getByText('已发布 1 项，失败 0 项，跳过 0 项。').waitFor();

  assert.equal(submittedBatches.length, 3);
  assert.deepEqual(submittedBatches[0].selected_items, [
    {
      candidate_id: candidateId,
      effect: 'create',
      current_published_knowledge_version: null,
      inspection_record_identity: 'event:ticket24-browser-inspection-001',
      acceptance_record_identity: 'event:ticket24-browser-acceptance-001'
    }
  ]);
  assert.deepEqual(submittedBatches[1].selected_items, submittedBatches[0].selected_items);
  assert.match(submittedBatches[0].confirmation_id, /^candidate-publication-/);
  assert.equal(submittedBatches[1].confirmation_id, submittedBatches[0].confirmation_id);
  assert.deepEqual(submittedBatches[2].selected_items, submittedBatches[0].selected_items);
  assert.notEqual(submittedBatches[2].confirmation_id, submittedBatches[0].confirmation_id);
});

test('Candidate selection never displays stale publication eligibility from an earlier request', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, { env: { BROWSER_ACCEPTANCE_SEED: 'minimal' } });
  const bundleId = 'bundle-ticket24-stale-eligibility';
  const firstJobId = 'job-ticket24-stale-eligibility-a';
  const secondJobId = 'job-ticket24-stale-eligibility-b';
  const firstCandidateId = 'candidate:job-ticket24-stale-eligibility-a-attempt-1';
  const secondCandidateId = 'candidate:job-ticket24-stale-eligibility-b-attempt-1';
  let signalFirstEligibilityRequest;
  let releaseFirstEligibility;
  let signalFirstEligibilityFulfilled;
  const firstEligibilityRequested = new Promise((resolve) => {
    signalFirstEligibilityRequest = resolve;
  });
  const firstEligibilityReleased = new Promise((resolve) => {
    releaseFirstEligibility = resolve;
  });
  const firstEligibilityFulfilled = new Promise((resolve) => {
    signalFirstEligibilityFulfilled = resolve;
  });
  const bundle = {
    bundle_id: bundleId,
    state: 'processing',
    schema_version: 1,
    editorial_source_revision: 'a'.repeat(64),
    exported_at: '2026-09-08T12:00:00Z',
    bundle_sha256: 'b'.repeat(64),
    items: [
      {
        bundle_item_id: 'bundle_item:ticket24-stale-eligibility-a',
        entry_identity: 'entry:ticket24-stale-eligibility-a',
        operation: 'create',
        state: 'admitted',
        artifact_sha256: 'c'.repeat(64),
        bundle_item_sha256: 'd'.repeat(64),
        allowed_next_action: 'await_explicit_publication',
        job_id: firstJobId
      },
      {
        bundle_item_id: 'bundle_item:ticket24-stale-eligibility-b',
        entry_identity: 'entry:ticket24-stale-eligibility-b',
        operation: 'create',
        state: 'admitted',
        artifact_sha256: 'e'.repeat(64),
        bundle_item_sha256: 'f'.repeat(64),
        allowed_next_action: 'await_candidate_inspection',
        job_id: secondJobId
      }
    ]
  };
  const candidateJob = (jobId, candidateId, entryId) => ({
    job_id: jobId,
    bundle_id: `bundle:${bundleId}`,
    bundle_item_id: `bundle_item:${entryId}`,
    entry_identity: `entry:${entryId}`,
    document_identity: `runtime-document:${entryId}`,
    requested_generation: 1,
    editorial_source_revision: 'a'.repeat(64),
    input_sha256: 'b'.repeat(64),
    frozen_input_sha256: 'c'.repeat(64),
    chunk_strategy: { strategy_id: 'section-aware-900-120' },
    embedding_configuration: { active: false },
    status: 'candidate_ready',
    stage: 'indexing',
    progress: 100,
    attempt: 1,
    terminal_state: 'candidate_ready',
    failure_reason: null,
    allowed_next_action: 'await_candidate_inspection',
    candidate_id: candidateId,
    derived_cleanup_pending: false,
    dispatched_at: null,
    started_at: null,
    heartbeat_at: null,
    lease_expires_at: null,
    completed_at: '2026-09-08T12:01:00Z',
    created_at: '2026-09-08T12:00:00Z',
    updated_at: '2026-09-08T12:01:00Z',
    events: []
  });
  const candidateDetail = (candidateId, entryId) => ({
    candidate: {
      candidate_id: candidateId,
      entry_identity: `entry:${entryId}`,
      document_identity: `runtime-document:${entryId}`,
      bundle_id: `bundle:${bundleId}`,
      bundle_item_id: `bundle_item:${entryId}`,
      bundle_sha256: 'b'.repeat(64),
      bundle_item_sha256: 'c'.repeat(64),
      editorial_source_revision: 'a'.repeat(64),
      input_sha256: 'b'.repeat(64),
      frozen_input_sha256: 'c'.repeat(64),
      generation: 1,
      configuration_identity: `configuration:${entryId}`,
      configuration: { active: false },
      chunks: [
        {
          chunk_id: `${entryId}-chunk-001`,
          chunk_index: 0,
          content_sha256: 'd'.repeat(64),
          content: `Candidate ${entryId} content`,
          metadata: {
            section_id: 'recommendation_or_reviewed_branches',
            section_title: 'Recommendation Or Reviewed Branches',
            chunk_strategy_id: 'section-aware-900-120',
            source_identities: ['source:ticket24-stale-source-001'],
            source_relationships: [
              {
                source_identity: 'source:ticket24-stale-source-001',
                availability: 'verified_usable',
                access_scope: 'public'
              }
            ]
          }
        }
      ]
    },
    inspection: null,
    replacement: {
      effect: 'create',
      current_published_knowledge_version: null,
      diff: {
        schema: 'candidate_replacement_diff/v1',
        added: [],
        changed: [],
        removed: []
      }
    }
  });
  const publicationEligibility = (candidateId, eligible) => ({
    candidate_id: candidateId,
    eligible,
    reasons: eligible ? [] : ['CANDIDATE_ACCEPTANCE_REQUIRED'],
    effect: 'create',
    current_published_knowledge_version: null,
    configuration_identity: `configuration:${candidateId}`,
    generation: 1,
    inspection_record_identity: 'event:ticket24-stale-inspection-001',
    acceptance_record_identity: eligible ? 'event:ticket24-stale-acceptance-001' : null
  });

  await page.route('**/reviewed-release-bundles**', async (route) => {
    const request = route.request();
    const pathname = decodeURIComponent(new URL(request.url()).pathname.replace(/\/+$/, ''));
    if (request.method() === 'GET' && pathname.endsWith('/reviewed-release-bundles')) {
      await route.fulfill({ contentType: 'application/json', body: JSON.stringify({ data: { items: [bundle] } }) });
      return;
    }
    if (request.method() === 'GET' && pathname.endsWith(`/reviewed-release-bundles/${bundleId}`)) {
      await route.fulfill({ contentType: 'application/json', body: JSON.stringify({ data: bundle }) });
      return;
    }
    if (request.method() === 'GET' && pathname.endsWith(`/reviewed-release-bundles/jobs/${firstJobId}`)) {
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({ data: candidateJob(firstJobId, firstCandidateId, 'ticket24-stale-eligibility-a') })
      });
      return;
    }
    if (request.method() === 'GET' && pathname.endsWith(`/reviewed-release-bundles/jobs/${secondJobId}`)) {
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({ data: candidateJob(secondJobId, secondCandidateId, 'ticket24-stale-eligibility-b') })
      });
      return;
    }
    if (request.method() === 'GET' && pathname.endsWith(`/candidates/${firstCandidateId}/inspection`)) {
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({ data: candidateDetail(firstCandidateId, 'ticket24-stale-eligibility-a') })
      });
      return;
    }
    if (request.method() === 'GET' && pathname.endsWith(`/candidates/${secondCandidateId}/inspection`)) {
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({ data: candidateDetail(secondCandidateId, 'ticket24-stale-eligibility-b') })
      });
      return;
    }
    if (request.method() === 'GET' && pathname.endsWith(`/candidates/${firstCandidateId}/publication-eligibility`)) {
      signalFirstEligibilityRequest();
      await firstEligibilityReleased;
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({ data: publicationEligibility(firstCandidateId, true) })
      });
      signalFirstEligibilityFulfilled();
      return;
    }
    if (request.method() === 'GET' && pathname.endsWith(`/candidates/${secondCandidateId}/publication-eligibility`)) {
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({ data: publicationEligibility(secondCandidateId, false) })
      });
      return;
    }
    await route.fallback();
  });

  await loginAdmin(page, baseUrl);
  await page.goto(`${baseUrl}reviewed-bundles`);
  await page.getByRole('button', { name: bundleId }).click();
  await page.getByRole('button', { name: `查看 Candidate ${firstCandidateId}` }).click();
  await firstEligibilityRequested;
  await page.getByRole('button', { name: `查看 Candidate ${secondCandidateId}` }).click();
  await page.getByRole('heading', { name: secondCandidateId }).waitFor();
  await page.getByText('尚未满足', { exact: true }).waitFor();

  releaseFirstEligibility();
  await firstEligibilityFulfilled;
  await page.waitForFunction((candidateId) => {
    const panel = document.querySelector('.reviewed-bundles__candidate-panel');
    return (
      panel?.querySelector('h2')?.textContent === candidateId &&
      panel.textContent?.includes('尚未满足') &&
      panel.querySelector('input[type="checkbox"]') === null
    );
  }, secondCandidateId);

  assert.equal(await page.getByRole('heading', { name: secondCandidateId }).isVisible(), true);
  assert.equal(await page.getByText('尚未满足', { exact: true }).isVisible(), true);
  assert.equal(await page.locator('.reviewed-bundles__candidate-panel input[type="checkbox"]').count(), 0);
});

test('System Administrator completes Ticket 24 inspection and isolated publication over the real disposable API', { timeout: 120000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, {
    env: {
      BROWSER_ACCEPTANCE_SEED: 'minimal',
      BROWSER_ACCEPTANCE_TICKET24: '1'
    }
  });
  await loginAdmin(page, baseUrl);
  await page.goto(`${baseUrl}reviewed-bundles`);
  await page.getByRole('heading', { name: 'Reviewed Release Bundles' }).waitFor();

  const ticket24Bundles = await page.evaluate(async () => {
    const response = await fetch('/api/reviewed-release-bundles', {
      headers: { Authorization: `Bearer ${localStorage.getItem('access_token')}` }
    });
    const payload = await response.json();
    return payload.data?.items || [];
  });
  const replacementBundle = ticket24Bundles.find((bundle) => bundle.bundle_id === ticket24ReplacementBundle);
  const failureBundle = ticket24Bundles.find((bundle) => bundle.bundle_id === ticket24FailureBundle);
  assert.ok(
    replacementBundle,
    `real fixture returns the replacement bundle; received ${JSON.stringify(ticket24Bundles)}`
  );
  assert.ok(
    failureBundle,
    `real fixture returns the failure-isolation bundle; received ${JSON.stringify(ticket24Bundles)}`
  );

  await page.getByRole('button', { name: replacementBundle.bundle_id }).click();
  await page.getByRole('button', { name: /^查看 Candidate / }).click();
  const replacementPanel = page.locator('.reviewed-bundles__candidate-panel');
  await page.getByText('替换当前 Published Knowledge Version', { exact: true }).waitFor();
  await page.getByText('Candidate chunks', { exact: true }).waitFor();
  await replacementPanel.getByText('section-aware-900-120', { exact: true }).first().waitFor();
  await replacementPanel.locator('.reviewed-bundles__replacement-content')
    .getByText(/material revision must be separately reviewed before it can replace the current pointer/i)
    .waitFor();

  const replacementCandidate = await replacementPanel.getByRole('heading').innerText();
  await replacementPanel.getByRole('button', { name: `记录 Candidate inspection ${replacementCandidate}` }).click();
  await replacementPanel.getByRole('button', { name: `执行 Candidate 验收 ${replacementCandidate}` }).click();
  await replacementPanel.getByText('可发布', { exact: true }).waitFor();
  const acceptance = replacementPanel.getByRole('region', { name: 'Candidate 验收记录' });
  await acceptance.waitFor();
  await acceptance.getByText('evidence_gated_answer', { exact: true }).waitFor();
  await acceptance.getByText('insufficient_evidence_reply', { exact: true }).waitFor();
  assert.equal(await acceptance.getByText('recommendation_or_reviewed_branches', { exact: true }).isVisible(), true);
  assert.equal(await acceptance.getByText('S1', { exact: true }).isVisible(), true);
  await replacementPanel.getByText('Frozen input SHA-256', { exact: true }).waitFor();
  const retainedAcceptance = await acceptance.innerText();
  await replacementPanel.getByRole('button', { name: `刷新 Candidate ${replacementCandidate}` }).click();
  await acceptance.waitFor();
  assert.equal(await acceptance.innerText(), retainedAcceptance);
  await replacementPanel.locator('input[type="checkbox"]').check();

  await page.getByRole('button', { name: failureBundle.bundle_id }).click();
  await page.getByRole('button', { name: /^查看 Candidate / }).click();
  const failurePanel = page.locator('.reviewed-bundles__candidate-panel');
  const failingCandidate = await failurePanel.getByRole('heading').innerText();
  await failurePanel.getByRole('button', { name: `记录 Candidate inspection ${failingCandidate}` }).click();
  await failurePanel.getByRole('button', { name: `执行 Candidate 验收 ${failingCandidate}` }).click();
  await failurePanel.getByText('可发布', { exact: true }).waitFor();
  await failurePanel.locator('input[type="checkbox"]').check();

  await page.getByRole('button', { name: '发布已选择 2 项' }).click();
  await page.getByRole('heading', { name: '确认发布 Candidate' }).waitFor();
  await page.locator('.reviewed-bundles__publication-dialog input[type="checkbox"]').check();
  await page.getByRole('button', { name: '确认发布' }).click();
  await page.getByText('已发布 1 项，失败 1 项，跳过 0 项。').waitFor();
  assert.equal(
    await page.getByText('PUBLICATION_ITEM_RETRYABLE_FAILURE', { exact: true }).isVisible(),
    true
  );
  assert.equal(await page.getByText(/批次发布完成/).count(), 0);
  await page.getByText('批次未全部发布：1 项已发布，1 项失败，0 项跳过。', { exact: true }).waitFor();

  const legacyBypass = await page.evaluate(async () => {
    const response = await fetch('/api/documents/runtime-document:ticket24-browser-entry-001/publish', {
      method: 'POST',
      headers: { Authorization: `Bearer ${localStorage.getItem('access_token')}` }
    });
    return { status: response.status, payload: await response.json() };
  });
  assert.equal(legacyBypass.status, 410);
  assert.equal(legacyBypass.payload.code, 'LEGACY_PUBLICATION_BYPASS_REJECTED');

  await page.goto(`${baseUrl}chat`);
  await page.getByPlaceholder('请输入需要检索的问题').fill(
    'Which Candidate publication contract applies? deployment=production'
  );
  await page.getByRole('button', { name: '发送' }).click();
  const answer = page.getByLabel('助手消息').last();
  await answer.getByLabel('已支持的知识回答').waitFor();
  await answer.getByRole('button', { name: '打开引用 S1' }).first().click();
  const source = page.getByRole('complementary', { name: '来源摘录' });
  await source.getByText('Ticket 24 candidate publication authority', { exact: true }).waitFor();
  assert.equal(await source.getByText(/candidate:|ticket24-browser-failure/).count(), 0);
});

test('Reviewed Release Bundle acceptance runs inside the deterministic browser gate', async () => {
  const packageJson = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'));

  assert.equal(
    packageJson.scripts['test:reviewed-bundles'],
    'node --test --test-concurrency=1 tests/reviewed-bundles-workspace.test.mjs'
  );
  assert.match(packageJson.scripts['test:browser'], /npm run test:reviewed-bundles/);
});
