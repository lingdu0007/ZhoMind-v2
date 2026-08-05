import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import net from 'node:net';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import test from 'node:test';
import { chromium } from 'playwright';
import { createServer, preview } from 'vite';

const artifactRoot = resolve(
  process.env.ZHOMIND_BROWSER_ARTIFACT_DIR || join(process.cwd(), 'test-artifacts', 'production-acceptance')
);
await mkdir(artifactRoot, { recursive: true });
const artifactRunDirectory = await mkdtemp(join(artifactRoot, 'run-'));
let artifactSequence = 0;

const captureWorkspaceArtifacts = async (page, label, payload) => {
  const artifactName = `${String(++artifactSequence).padStart(3, '0')}-${label}`;
  const screenshotPath = join(artifactRunDirectory, `${artifactName}.png`);
  const geometryPath = join(artifactRunDirectory, `${artifactName}.json`);
  const screenshot = await page.screenshot({ path: screenshotPath, animations: 'disabled' });
  await writeFile(geometryPath, `${JSON.stringify(payload, null, 2)}\n`);
  return { screenshot, geometryPath };
};

const reservePort = () =>
  new Promise((resolvePort, reject) => {
    const listener = net.createServer();
    listener.once('error', reject);
    listener.listen(0, '127.0.0.1', () => {
      const address = listener.address();
      listener.close((error) => (error ? reject(error) : resolvePort(address.port)));
    });
  });

const startApiEnvironment = async (t) => {
  const tempDirectory = await mkdtemp(join(tmpdir(), 'zhomind-browser-api-'));
  const port = await reservePort();
  const baseUrl = `http://127.0.0.1:${port}/api/v1`;
  const backendDirectory = resolve(process.cwd(), '../backend');
  const output = [];
  const apiProcess = spawn('uv', ['run', '--no-sync', 'python', 'tests/browser_acceptance_api.py', '--host', '127.0.0.1', '--port', String(port)], {
    cwd: backendDirectory,
    // Own process group so a failed journey can never leave the uv wrapper or
    // its python child behind to exhaust the runner's memory.
    detached: true,
    env: {
      ...process.env,
      PYTHONPATH: backendDirectory,
      DATABASE_URL: `sqlite+aiosqlite:///${join(tempDirectory, 'acceptance.db')}`,
      JWT_SECRET: 'browser-acceptance-secret',
      BOOTSTRAP_ADMIN_USERNAME: 'operator',
      BOOTSTRAP_ADMIN_PASSWORD: 'safe-password',
      SYSTEM_SETTINGS_DRAFT_ENABLED: 'true',
      SYSTEM_SETTINGS_APPLICATION_ENABLED: 'true',
      SYSTEM_SETTINGS_ENCRYPTION_KEY: 'MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY=',
      DENSE_EMBEDDING_DIM: '0',
      EMBEDDING_API_KEY: '',
      EMBEDDING_BASE_URL: '',
      EMBEDDING_MODEL: '',
      MILVUS_URI: '',
      MILVUS_TOKEN: '',
      RAG_PRIMARY_LLM_PROVIDER: 'browser-acceptance',
      RAG_LLM_FALLBACK_PROVIDERS: ''
    },
    stdio: ['ignore', 'pipe', 'pipe']
  });
  apiProcess.stdout.on('data', (chunk) => output.push(chunk.toString()));
  apiProcess.stderr.on('data', (chunk) => output.push(chunk.toString()));

  t.after(async () => {
    if (apiProcess.exitCode === null) {
      try {
        process.kill(-apiProcess.pid, 'SIGTERM');
      } catch {
        apiProcess.kill('SIGTERM');
      }
      await Promise.race([once(apiProcess, 'exit'), new Promise((resolveWait) => setTimeout(resolveWait, 5000))]);
      if (apiProcess.exitCode === null) {
        try {
          process.kill(-apiProcess.pid, 'SIGKILL');
        } catch {
          apiProcess.kill('SIGKILL');
        }
      }
    }
    await rm(tempDirectory, { recursive: true, force: true });
  });

  const deadline = Date.now() + 15000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${baseUrl}/health`);
      if (response.ok) return { baseUrl, port };
    } catch {
      // The runner has not bound its socket yet.
    }
    if (apiProcess.exitCode !== null) break;
    await new Promise((resolveWait) => setTimeout(resolveWait, 100));
  }

  throw new Error(`isolated application API environment did not become healthy:\n${output.join('')}`);
};

test('production browser acceptance starts an isolated application API environment', { timeout: 30000 }, async (t) => {
  const api = await startApiEnvironment(t);
  const response = await fetch(`${api.baseUrl}/health`);

  assert.equal(response.status, 200);
  assert.deepEqual((await response.json()).data.status, 'up');
});

const startWorkbench = async (t, { built, viewport = { width: 1440, height: 900 } }) => {
  const api = await startApiEnvironment(t);
  const previousProxyTarget = process.env.ZHOMIND_API_PROXY_TARGET;
  process.env.ZHOMIND_API_PROXY_TARGET = `http://127.0.0.1:${api.port}`;
  // Vite resolves `port: 0` from the project config (5173); reserve a distinct
  // random port so parallel test files never collide.
  const webPort = await reservePort();
  const server = built
    ? await preview({ preview: { host: '127.0.0.1', port: webPort, strictPort: true } })
    : await createServer({ server: { host: '127.0.0.1', port: webPort, strictPort: true } });
  if (!built) await server.listen();

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport });
  const page = await context.newPage();
  page.setDefaultTimeout(20000);

  t.after(async () => {
    await browser.close();
    await server.close();
    if (previousProxyTarget === undefined) delete process.env.ZHOMIND_API_PROXY_TARGET;
    else process.env.ZHOMIND_API_PROXY_TARGET = previousProxyTarget;
  });

  return { page, baseUrl: server.resolvedUrls.local[0], api };
};

const createTeamInvitation = async (api) => {
  const login = await fetch(`${api.baseUrl}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: 'operator', password: 'safe-password' })
  });
  assert.equal(login.status, 200);
  const token = (await login.json()).data.access_token;
  const invitation = await fetch(`${api.baseUrl}/members/invitations`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({})
  });
  assert.equal(invitation.status, 200);
  return (await invitation.json()).data.invitation_code;
};

const register = async (page, baseUrl, api, { username, role }) => {
  await page.goto(`${baseUrl}auth`);
  await page.getByRole('heading', { name: '身份验证' }).waitFor();
  assert.equal(await page.getByRole('navigation').count(), 0);
  if (role === 'admin') {
    await page.getByLabel('用户名').fill(username);
    await page.getByLabel('密码').fill('safe-password');
    await page.getByRole('button', { name: '登录' }).click();
    await page.waitForURL(/\/chat$/);
    await page.getByRole('heading', { name: '对话工作区' }).waitFor();
    return;
  }

  await page.getByRole('tab', { name: '注册' }).click();
  await page.getByLabel('用户名').fill(username);
  await page.getByLabel('密码').fill('safe-password');
  await page.getByLabel('团队邀请码').fill(await createTeamInvitation(api));
  await page.getByRole('button', { name: '完成注册' }).click();
  await page.waitForURL(/\/chat$/);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();
};

const readStableRegions = async (page) =>
  page.evaluate(() => {
    const visible = (element) => {
      const style = getComputedStyle(element);
      const bounds = element.getBoundingClientRect();
      return style.display !== 'none' && style.visibility !== 'hidden' && bounds.width > 0 && bounds.height > 0;
    };
    return [...document.querySelectorAll('main h1, main h2, main h3, main button, main input, main textarea, main select')]
      .filter(visible)
      .slice(0, 80)
      .map((element) => {
        const bounds = element.getBoundingClientRect();
        return {
          label: element.getAttribute('aria-label') || element.textContent.trim() || element.getAttribute('placeholder') || element.tagName,
          x: bounds.x,
          y: bounds.y,
          width: bounds.width,
          height: bounds.height
        };
      });
  });

const assertRenderedWorkspace = async (page, artifactLabel = 'workspace') => {
  await page.evaluate(() => document.fonts?.ready);
  await page.waitForTimeout(500);
  const initialRegions = await readStableRegions(page);
  const geometry = await page.evaluate(() => {
    const visible = (element) => {
      const style = getComputedStyle(element);
      const bounds = element.getBoundingClientRect();
      return style.display !== 'none' && style.visibility !== 'hidden' && bounds.width > 0 && bounds.height > 0;
    };
    const within = (inner, outer) =>
      inner.left >= outer.left - 1 && inner.right <= outer.right + 1 && inner.top >= outer.top - 1 && inner.bottom <= outer.bottom + 1;
    const primary = document.querySelector('main');
    const primaryBounds = primary?.getBoundingClientRect();
    const controls = [...document.querySelectorAll('button, input, textarea, select, [role="button"]')].filter(visible);
    const viewport = { width: window.innerWidth, height: window.innerHeight };
    const overflowingControls = controls.filter((element) => {
      const bounds = element.getBoundingClientRect();
      return bounds.left < -1 || bounds.right > viewport.width + 1;
    });
    const escapedControls = controls.filter((element) => {
      const container = element.closest('form, section, main, aside, nav, header, [role="dialog"]');
      return container && visible(container) && !within(element.getBoundingClientRect(), container.getBoundingClientRect());
    });
    const textElements = [...document.querySelectorAll('main h1, main h2, main h3, main p, main button, main label, main small, main th, main td, main summary')]
      .filter(visible)
      .slice(0, 100);
    const renderedTextElements = textElements.filter((element) => {
      const bounds = element.getBoundingClientRect();
      return bounds.width > 1 && bounds.height > 1;
    });
    const clippedText = renderedTextElements.filter((element) => {
      const style = getComputedStyle(element);
      return style.textOverflow !== 'ellipsis' && element.scrollWidth > element.clientWidth + 1;
    });
    const overlappingText = [];
    for (let outerIndex = 0; outerIndex < renderedTextElements.length; outerIndex += 1) {
      const outer = renderedTextElements[outerIndex];
      const outerBounds = outer.getBoundingClientRect();
      for (let innerIndex = outerIndex + 1; innerIndex < renderedTextElements.length; innerIndex += 1) {
        const inner = renderedTextElements[innerIndex];
        if (outer.contains(inner) || inner.contains(outer)) continue;
        const innerBounds = inner.getBoundingClientRect();
        const overlaps =
          Math.min(outerBounds.right, innerBounds.right) - Math.max(outerBounds.left, innerBounds.left) > 2 &&
          Math.min(outerBounds.bottom, innerBounds.bottom) - Math.max(outerBounds.top, innerBounds.top) > 2;
        if (overlaps) {
          overlappingText.push(`${outer.textContent.trim()} / ${inner.textContent.trim()}`);
        }
      }
    }
    return {
      hasPrimaryContent: Boolean(primary?.innerText.trim()),
      primaryWidth: primaryBounds?.width || 0,
      primaryHeight: primaryBounds?.height || 0,
      pageFitsViewport: document.documentElement.scrollWidth <= viewport.width,
      overflowingControls: overflowingControls.map((element) => element.getAttribute('aria-label') || element.textContent.trim()),
      escapedControls: escapedControls.map((element) => element.getAttribute('aria-label') || element.textContent.trim()),
      clippedText: clippedText.map((element) => element.textContent.trim()),
      overlappingText
    };
  });

  const artifact = await captureWorkspaceArtifacts(page, artifactLabel, { initialRegions, geometry });

  assert.equal(geometry.hasPrimaryContent, true);
  assert.ok(geometry.primaryWidth > 0 && geometry.primaryHeight > 0);
  assert.equal(geometry.pageFitsViewport, true);
  assert.deepEqual(geometry.overflowingControls, []);
  assert.deepEqual(geometry.escapedControls, []);
  assert.deepEqual(geometry.clippedText, []);
  assert.deepEqual(geometry.overlappingText, []);
  assert.ok(artifact.screenshot.byteLength > 1000);
  await page.waitForTimeout(300);
  const settledRegions = await readStableRegions(page);
  await writeFile(artifact.geometryPath, `${JSON.stringify({ initialRegions, geometry, settledRegions }, null, 2)}\n`);
  assert.equal(settledRegions.length, initialRegions.length);
  settledRegions.forEach((region, index) => {
    const initial = initialRegions[index];
    assert.equal(region.label, initial.label);
    assert.ok(Math.abs(region.x - initial.x) <= 1);
    assert.ok(Math.abs(region.y - initial.y) <= 1);
    assert.ok(Math.abs(region.width - initial.width) <= 1);
    assert.ok(Math.abs(region.height - initial.height) <= 1);
  });
};

for (const runtime of [
  { name: 'development server', built: false },
  { name: 'built production assets', built: true }
]) {
  test(`System Administrator completes the authorized workspace journey on ${runtime.name}`, { timeout: 120000 }, async (t) => {
    const { page, baseUrl, api } = await startWorkbench(t, runtime);

    await register(page, baseUrl, api, { username: 'operator', role: 'admin' });
    await page.getByPlaceholder('请输入需要检索的问题').fill('部署前需要做什么？');
    await page.getByRole('button', { name: '发送' }).click();
    const diagnostics = page.getByLabel('检索诊断');
    await diagnostics.waitFor();
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await diagnostics.getByText('检索诊断', { exact: true }).click();
    await diagnostics.getByText('候选数', { exact: true }).waitFor();
    assert.equal(await diagnostics.getByText('候选数', { exact: true }).isVisible(), true);
    await assertRenderedWorkspace(page, `admin-${runtime.built ? 'built' : 'development'}-chat`);

    await page.getByRole('link', { name: '文档库' }).click();
    await page.getByRole('heading', { name: '文档库' }).waitFor();
    await page.getByLabel('选择文档').setInputFiles({
      name: 'browser-upload.md',
      mimeType: 'text/markdown',
      buffer: Buffer.from('上传文档也需要完成变更审批。')
    });
    await page.getByRole('button', { name: '上传文档' }).click();
    await page.getByText(/文档 ID：/).waitFor();
    await page.getByText(/构建任务 ID：/).waitFor();
    await page.getByRole('button', { name: /查看构建任务/ }).click();
    await page.waitForURL(/\/jobs\?job=/);
    await page.getByRole('heading', { name: '构建任务' }).waitFor();

    await page.getByRole('link', { name: '文档库' }).click();
    await page.getByRole('heading', { name: '文档库' }).waitFor();
    await page.getByRole('button', { name: '查看文档 browser-inspection 的候选或已发布分块' }).click();
    await page.getByRole('heading', { name: '已发布版本分块' }).waitFor();
    await page.getByText('已发布分块可用于检查部署审批记录。').waitFor();
    await page.keyboard.press('Escape');
    await page.getByRole('heading', { name: '已发布版本分块' }).waitFor({ state: 'detached' });

    await page.getByRole('button', { name: '删除文档 browser-single-delete' }).click();
    await page.getByRole('dialog').getByRole('button', { name: '删除', exact: true }).click();
    await page.getByText('文档 browser-single-delete.md 已由服务端确认删除。').waitFor();
    assert.equal(await page.getByRole('cell', { name: 'browser-single-delete.md' }).count(), 0);

    await page.getByLabel('选中文档 browser-batch-partial-first').check();
    await page.getByLabel('选中文档 browser-batch-partial-second').check();
    const concurrentAdministratorPage = await page.context().newPage();
    concurrentAdministratorPage.setDefaultTimeout(20000);
    try {
      await concurrentAdministratorPage.goto(`${baseUrl}documents`);
      await concurrentAdministratorPage.getByRole('heading', { name: '文档库' }).waitFor();
      await concurrentAdministratorPage.getByRole('button', { name: '删除文档 browser-batch-partial-first' }).click();
      await concurrentAdministratorPage.getByRole('dialog').getByRole('button', { name: '删除', exact: true }).click();
      await concurrentAdministratorPage.getByText('文档 browser-batch-partial-first.md 已由服务端确认删除。').waitFor();
    } finally {
      await concurrentAdministratorPage.close();
    }

    await page.getByRole('button', { name: '批量删除' }).click();
    await page.getByRole('dialog').getByRole('button', { name: '删除', exact: true }).click();
    await page.getByText('已由服务端确认删除 1 个文档。').waitFor();
    assert.equal(await page.getByRole('cell', { name: 'browser-batch-partial-second.md' }).count(), 0);
    assert.equal(await page.getByRole('cell', { name: 'browser-batch-partial-first.md' }).isVisible(), true);
    assert.equal(
      await page.getByText('文档 browser-batch-partial-first.md：document not found').isVisible(),
      true
    );
    await assertRenderedWorkspace(page, `admin-${runtime.built ? 'built' : 'development'}-partial-delete`);
    await page.getByLabel('选中文档 browser-batch-partial-first').uncheck();

    await page.getByRole('button', { name: '重新构建文档 browser-inspection' }).click();
    await page.getByRole('heading', { name: '重新构建文档' }).waitFor();
    await page.getByLabel('重建分块策略').selectOption('paper');
    await page.getByRole('button', { name: '创建重建任务' }).click();
    await page.getByText(/^已创建重建任务 /).waitFor();

    await page.getByLabel('选中文档 browser-batch-first').check();
    await page.getByLabel('选中文档 browser-batch-second').check();
    await page.getByRole('button', { name: '批量重新构建' }).click();
    await page.getByRole('heading', { name: '重新构建 2 个文档' }).waitFor();
    await page.getByLabel('批量重建分块策略').selectOption('qa');
    await page.getByRole('button', { name: '创建 2 个重建任务' }).click();
    await page.getByText('已为 2 个文档创建重建任务。').waitFor();
    await assertRenderedWorkspace(page);
    await page.getByRole('button', { name: '前往构建任务' }).click();
    await page.waitForURL(/\/jobs$/);
    await page.getByRole('heading', { name: '构建任务' }).waitFor();
    await page.getByRole('button', { name: '取消任务 job-browser-cancelable' }).click();
    await page.getByText('任务 job-browser-cancelable 的取消结果已由服务端确认。').waitFor();
    assert.equal(await page.getByText('已取消 (canceled)').isVisible(), true);
    await assertRenderedWorkspace(page);

    await page.getByRole('link', { name: '系统设置' }).click();
    await page.getByRole('heading', { name: '系统设置' }).waitFor();
    await page.getByLabel('生成模型').fill('Qwen/Qwen3-14B');
    await page.getByRole('button', { name: '保存并应用' }).click();
    await page.getByText('设置已生效。').waitFor({ timeout: 8000 });
    assert.equal(await page.getByText('生效版本 2').isVisible(), true);
    await assertRenderedWorkspace(page);

    await page.setViewportSize({ width: 1024, height: 900 });
    for (const workspace of [
      { path: 'documents', title: '文档库', notice: '文档库当前仅支持桌面工作区。' },
      { path: 'jobs', title: '构建任务', notice: '构建任务当前仅支持桌面工作区。' },
      { path: 'config', title: '系统设置', notice: '系统设置当前仅支持桌面工作区。' }
    ]) {
      await page.goto(`${baseUrl}${workspace.path}`);
      await page.getByRole('heading', { name: workspace.title }).waitFor();
      assert.equal(await page.getByText(workspace.notice).count(), 0);
      await assertRenderedWorkspace(page, `admin-${runtime.built ? 'built' : 'development'}-compact-${workspace.path}`);
    }

    await page.setViewportSize({ width: 390, height: 844 });
    for (const workspace of [
      { path: 'documents', notice: '文档库当前仅支持桌面工作区。' },
      { path: 'jobs', notice: '构建任务当前仅支持桌面工作区。' },
      { path: 'config', notice: '系统设置当前仅支持桌面工作区。' }
    ]) {
      await page.goto(`${baseUrl}${workspace.path}`);
      await page.getByText(workspace.notice).waitFor();
      await assertRenderedWorkspace(page);
    }
  });

  test(`Knowledge User keeps Conversation Workspace usable and protected on ${runtime.name}`, { timeout: 30000 }, async (t) => {
    const { page, baseUrl, api } = await startWorkbench(t, runtime);

    await register(page, baseUrl, api, { username: 'knowledge-user', role: 'user' });
    assert.equal(await page.getByRole('link', { name: '文档库' }).count(), 0);
    assert.equal(await page.getByRole('link', { name: '构建任务' }).count(), 0);
    assert.equal(await page.getByRole('link', { name: '系统设置' }).count(), 0);
    const documentsStatus = await page.evaluate(async () => {
      const token = localStorage.getItem('access_token');
      const response = await fetch('/api/documents', { headers: { Authorization: `Bearer ${token}` } });
      return response.status;
    });
    assert.equal(documentsStatus, 403);

    const rail = page.getByRole('complementary', { name: '最近会话' });
    await rail.waitFor();
    await page.getByPlaceholder('请输入需要检索的问题').fill('部署前需要做什么？');
    await page.getByRole('button', { name: '发送' }).click();

    const summary = page.getByLabel('证据摘要');
    await summary.waitFor();
    assert.equal(await summary.getByText('证据充分').isVisible(), true);
    assert.equal(await page.getByLabel('检索诊断').count(), 0);
    const sourceButton = summary.getByRole('button', { name: '查看来源 browser-evidence.md' });
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await sourceButton.click();
    const excerptDrawer = page.getByRole('complementary', { name: '来源摘录' });
    await excerptDrawer.waitFor();
    const [answerBox, drawerBox, drawerAnimation] = await Promise.all([
      page.getByLabel('助手消息').boundingBox(),
      excerptDrawer.boundingBox(),
      excerptDrawer.evaluate((element) => getComputedStyle(element).animationName)
    ]);
    assert.ok(answerBox.x + answerBox.width <= drawerBox.x);
    assert.equal(drawerAnimation, 'none');
    await assertRenderedWorkspace(page);
    await page.keyboard.press('Escape');
    await excerptDrawer.waitFor({ state: 'detached' });
    assert.equal(await page.evaluate(() => document.activeElement?.getAttribute('aria-label')), '查看来源 browser-evidence.md');

    await page.setViewportSize({ width: 1024, height: 900 });
    assert.equal(await rail.isVisible(), false);
    assert.equal(await page.getByRole('button', { name: '会话', exact: true }).isVisible(), true);
    const [titleBox, composerBox] = await Promise.all([
      page.getByRole('heading', { name: '对话工作区' }).boundingBox(),
      page.getByPlaceholder('请输入需要检索的问题').boundingBox()
    ]);
    assert.ok(composerBox.x >= titleBox.x);
    assert.ok(composerBox.x + composerBox.width <= 1024);
    assert.ok(composerBox.y >= titleBox.y + titleBox.height);
    await assertRenderedWorkspace(page);

    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByRole('button', { name: '会话', exact: true }).click();
    const sessionDialog = page.getByRole('dialog');
    await sessionDialog.waitFor();
    assert.equal(await sessionDialog.getByRole('heading', { name: '最近会话' }).first().isVisible(), true);
    assert.equal(
      await page.evaluate(() => document.querySelector('[role="dialog"]')?.contains(document.activeElement)),
      true
    );
    await page.keyboard.press('Escape');
    await sessionDialog.waitFor({ state: 'detached' });
    assert.equal(await page.getByPlaceholder('请输入需要检索的问题').isVisible(), true);
    await assertRenderedWorkspace(page);

    await page.goto(`${baseUrl}documents`);
    await page.waitForURL(/\/chat\?notice=admin-required$/);
    assert.equal(await page.getByRole('heading', { name: '文档库' }).count(), 0);
  });
}
