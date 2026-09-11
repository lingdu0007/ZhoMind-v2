import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { loginAdmin, registerKnowledgeUserViaApi, startWorkbench } from './acceptance-env.mjs';

const request = async (api, path, token, body) => {
  const response = await fetch(`${api.baseUrl}${path}`, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    ...(body === undefined ? {} : { body: JSON.stringify(body) })
  });
  assert.equal(response.status, 200, `${path} returned ${response.status}`);
  const result = await response.json();
  return result.data ?? result;
};

for (const built of [false, true]) {
  test(`maintainer reviews current knowledge health and records weekly, monthly and quarterly evidence (${built ? 'built' : 'development'})`,
    { timeout: 120000 }, async (t) => {
      const { page, baseUrl, api } = await startWorkbench(t, { built, env: { BROWSER_ACCEPTANCE_TICKET24: '1' } });
      const token = await registerKnowledgeUserViaApi(api, 'cadence-browser-maintainer');
      const admin = await request(api, '/auth/login', '', { username: 'operator', password: 'safe-password' });
      await request(api, '/maintenance/assignments', admin.access_token, { username: 'cadence-browser-maintainer' });
      await loginAdmin(page, baseUrl, { username: 'cadence-browser-maintainer' });
      await page.getByRole('link', { name: '知识维护', exact: true }).click();
      await page.getByRole('button', { name: '接受维护责任' }).click();
      await page.getByRole('tab', { name: '例行评审', exact: true }).click();
      const health = page.getByRole('table', { name: '知识健康快照' });
      await health.getByText('存在后续编辑版本', { exact: true }).waitFor();
      const context = await request(api, '/maintenance/review-context', token);
      assert.ok(context.knowledge_health.published_entries.some((entry) => entry.published_revision_identity !== entry.current_revision_identity));
      const sample = context.sample_options[0];
      assert.ok(sample.verified_at);
      assert.ok(sample.verified_checks.length);
      const review = page.getByRole('form', { name: '记录例行评审' });
      const history = page.getByRole('table', { name: '例行评审记录' });
      for (const [period, label] of [['weekly', '每周'], ['monthly', '每月'], ['quarterly', '每季度']]) {
        await review.getByLabel('评审周期', { exact: true }).selectOption(period);
        if (period === 'quarterly') {
          assert.equal(await review.getByRole('button', { name: '记录例行评审', exact: true }).isDisabled(), true);
          await review.getByText(sample.verified_by, { exact: true }).first().waitFor();
          await review.getByText(sample.verified_checks[0].check_id, { exact: true }).first().waitFor();
          await review.getByRole('checkbox', { name: /^抽样验收 / }).first().check();
        }
        await review.getByRole('checkbox', { name: '确认本次评审', exact: true }).check();
        await review.getByRole('button', { name: '记录例行评审', exact: true }).click();
        await history.getByText(label, { exact: true }).waitFor();
      }
      await page.reload();
      await page.getByRole('tab', { name: '例行评审', exact: true }).click();
      await history.getByText('每季度', { exact: true }).waitFor();
      assert.equal(await history.locator('tbody tr').count(), 3);
      await history.getByRole('button', { name: /查看评审/ }).last().click();
      const snapshot = page.getByRole('region', { name: '已记录评审快照' });
      await snapshot.getByText('已绑定抽样验收', { exact: true }).waitFor();
      await snapshot.getByText(sample.verified_at, { exact: true }).waitFor();
      await snapshot.getByText(sample.verified_by, { exact: true }).waitFor();
      for (const check of sample.verified_checks) {
        await snapshot.getByText(check.check_id, { exact: true }).waitFor();
        for (const link of check.evidence_links) {
          await snapshot.getByText(link, { exact: true }).first().waitFor();
        }
      }
      assert.equal((await snapshot.innerText()).includes('provider_api_key'), false);
      if (process.env.MAINTENANCE_SCREENSHOT_DIR) {
        await mkdir(process.env.MAINTENANCE_SCREENSHOT_DIR, { recursive: true });
        await page.screenshot({ path: join(process.env.MAINTENANCE_SCREENSHOT_DIR, 'cadence-desktop.png'), fullPage: true });
      }
      await page.setViewportSize({ width: 390, height: 844 });
      await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth);
      if (process.env.MAINTENANCE_SCREENSHOT_DIR) {
        await page.screenshot({ path: join(process.env.MAINTENANCE_SCREENSHOT_DIR, 'cadence-mobile.png'), fullPage: true });
      }
    });
}
