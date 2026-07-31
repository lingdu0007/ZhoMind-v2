import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';

const siteAddress = process.env.DEPLOY_CADDY_SITE_ADDRESS;
assert.match(siteAddress || '', /^[A-Za-z0-9.-]+\.[A-Za-z0-9.-]+$/);

const repositoryRoot = resolve(import.meta.dirname, '../..');
const require = createRequire(resolve(repositoryRoot, 'frontend/package.json'));
const { chromium } = require('@playwright/test');
const baseUrl = `https://${siteAddress}`;
const username = `deploy-browser-${Date.now()}`;
const password = 'deployment-browser-password';

const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  page.setDefaultTimeout(10000);

  await page.goto(`${baseUrl}/auth`, { waitUntil: 'networkidle' });
  await page.getByRole('heading', { name: '身份验证' }).waitFor();
  await page.getByRole('tab', { name: '注册' }).click();
  await page.getByLabel('用户名').fill(username);
  await page.getByLabel('密码').fill(password);
  await page.getByRole('button', { name: '完成注册' }).click();
  await page.waitForURL(/\/chat$/);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();
  assert.equal(await page.getByText('文档库', { exact: true }).count(), 0);
  printf('public browser registration and Knowledge User route guard passed\n');
} finally {
  await browser.close();
}

function printf(message) {
  process.stdout.write(message);
}
