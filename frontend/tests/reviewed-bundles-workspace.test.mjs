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

test('Reviewed Release Bundle controls follow the server-provided allowed action', async () => {
  const page = await readFile(new URL('../src/pages/ReviewedBundlesPage.vue', import.meta.url), 'utf8');

  assert.match(
    page,
    /const canRetry = \(item\) =>\s*\['retry_fixed_inputs', 'reconcile_derived_data_then_retry'\]\.includes\(jobFor\(item\)\?\.allowed_next_action\);/
  );
  assert.match(page, /const canCancel = \(item\) => jobFor\(item\)\?\.allowed_next_action === 'cancel_or_await_candidate_build';/);
  assert.match(page, /completed_with_rejections: '已完成，存在拒绝项'/);
  assert.match(page, /const bundleStateTone = \(state\) => \{[\s\S]*completed_with_rejections.*return 'danger';/);
  assert.match(page, /:class="`reviewed-bundles__status--\$\{bundleStateTone\(selectedBundle\.state\)\}`"/);
  assert.match(page, /const refreshSelectedBundle = async \(\) => \{/);
  assert.match(page, /jobs\.forEach\(mergeJob\);\s*await refreshSelectedBundle\(\);/);
  assert.match(page, /const structuredReasonMessage = \(detail\) => \{/);
  assert.match(page, /const applyActionResult = \(job, successMessage, failureFallback\) => \{/);
  assert.match(page, /if \(job\?\.status === 'failed'\) \{/);
});

test('Reviewed Release Bundle acceptance runs inside the deterministic browser gate', async () => {
  const packageJson = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'));

  assert.equal(
    packageJson.scripts['test:reviewed-bundles'],
    'node --test --test-concurrency=1 tests/reviewed-bundles-workspace.test.mjs'
  );
  assert.match(packageJson.scripts['test:browser'], /npm run test:reviewed-bundles/);
});
