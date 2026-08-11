import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';

const siteAddress = process.env.DEPLOY_CADDY_SITE_ADDRESS;
assert.match(siteAddress || '', /^[A-Za-z0-9.-]+\.[A-Za-z0-9.-]+$/);

const repositoryRoot = resolve(import.meta.dirname, '../..');
const require = createRequire(resolve(repositoryRoot, 'frontend/package.json'));
const { chromium } = require('@playwright/test');
const baseUrl = `https://${siteAddress}`;

const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  page.setDefaultTimeout(10000);

  await page.goto(`${baseUrl}/documents`, { waitUntil: 'networkidle' });
  await page.waitForURL(/\/auth$/);
  await page.getByRole('heading', { name: '身份验证' }).waitFor();
  await page.getByRole('tab', { name: '注册' }).click();
  const invitationCode = page.getByLabel('团队邀请码');
  await invitationCode.waitFor();
  assert.equal(await invitationCode.evaluate((input) => input.required), true);
  assert.equal(await page.getByText('文档库', { exact: true }).count(), 0);
  printf('public browser invitation gate and unauthenticated route guard passed\n');
} finally {
  await browser.close();
}

function printf(message) {
  process.stdout.write(message);
}
