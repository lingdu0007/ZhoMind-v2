import assert from 'node:assert/strict';
import test from 'node:test';
import { chromium } from 'playwright';
import { createServer, preview } from 'vite';

const jsonResponse = (data, status = 200) => ({
  status,
  contentType: 'application/json',
  body: JSON.stringify({ data })
});

const requestPath = (url) => new URL(url).pathname.replace(/^\/api(?:\/v1)?/, '');

const systemSettingsDraft = {
  draft: {
    model_provider: 'ark',
    llm_model: 'Qwen/Qwen3-32B',
    embedding_model: 'BAAI/bge-m3',
    retrieval_strategy: 'migration',
    retrieval_top_k: 8,
    score_threshold: 0.3,
    milvus_uri: 'http://milvus.internal:19530',
    index_name: 'zhomind_docs',
    runtime_timeout_ms: 8000,
    provider_api_key: { configured: true }
  },
  saved_version: 2,
  active_version: 2,
  last_modified: { actor: 'operator', at: '2026-07-31T10:15:00Z' },
  application_state: 'active',
  application: { version: 2, actor: 'operator', at: '2026-07-31T10:15:00Z', message: 'settings version is active' }
};

const sessions = [{ session_id: 'session-acceptance', updated_at: '2026-07-31T10:15:00Z', message_count: 2 }];

const installStreamingResponse = async (page) => {
  await page.addInitScript(() => {
    const nativeFetch = window.fetch.bind(window);
    window.fetch = async (input, init) => {
      if (!String(input).includes('/chat/stream')) return nativeFetch(input, init);

      const encoder = new TextEncoder();
      return new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(encoder.encode('event: content\ndata: {"content":"部署前需要完成变更审批。"}\n\n'));
            controller.enqueue(
              encoder.encode(
                'event: evidence_summary\ndata: {"evidence_summary":{"coverage":"sufficient","source_count":1,"sources":[{"source_id":"chunk-deploy-1","metadata":{"filename":"deploy-runbook.md"},"excerpt":"发布前必须由值班负责人完成变更审批。"}]}}\n\n'
              )
            );
            controller.enqueue(
              encoder.encode(
                'event: retrieval_diagnostics\ndata: {"retrieval_diagnostics":{"timeline":[{"step":"retrieve"}],"candidate_counts":{"retrieved":1,"reranked":1},"evidence_gate":{"outcome":"passed","reason":"sufficient_evidence"},"fallback":{"state":"not_used","hops":0,"final_provider":null},"provider_errors":[],"trace_preview":"raw trace remains administrator-only"}}\n\n'
              )
            );
            controller.enqueue(encoder.encode('event: done\ndata: [DONE]\n\n'));
            controller.close();
          }
        }),
        { status: 200, headers: { 'Content-Type': 'text/event-stream' } }
      );
    };
  });
};

const startWorkbench = async (
  t,
  { built, role = 'admin', viewport = { width: 1440, height: 900 }, conversationSessions = sessions }
) => {
  const port = 44000 + Math.floor(Math.random() * 1000);
  const server = built
    ? await preview({ preview: { host: '127.0.0.1', port, strictPort: true } })
    : await createServer({ server: { host: '127.0.0.1', port, strictPort: true } });
  if (!built) await server.listen();

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport });
  page.setDefaultTimeout(4000);

  await page.route((url) => url.pathname.startsWith('/api/'), async (route) => {
    const request = route.request();
    const path = requestPath(request.url());

    if (path === '/auth/login') {
      await route.fulfill(jsonResponse({ access_token: 'admin-token' }));
      return;
    }
    if (path === '/auth/me') {
      await route.fulfill(jsonResponse({ username: role === 'admin' ? 'operator' : 'knowledge-user', role }));
      return;
    }
    if (path === '/sessions') {
      await route.fulfill(jsonResponse({ sessions: conversationSessions }));
      return;
    }
    if (path === '/documents') {
      await route.fulfill(jsonResponse({ items: [] }));
      return;
    }
    if (path === '/documents/jobs') {
      await route.fulfill(jsonResponse({ items: [] }));
      return;
    }
    if (path === '/settings/draft') {
      await route.fulfill(jsonResponse(systemSettingsDraft));
      return;
    }

    await route.fulfill(jsonResponse({ message: `Unexpected request: ${request.method()} ${path}` }, 404));
  });

  t.after(async () => {
    await browser.close();
    await server.close();
  });

  return { page, baseUrl: server.resolvedUrls.local[0] };
};

const signIn = async (page, baseUrl, username) => {
  await page.goto(`${baseUrl}auth`);
  await page.getByRole('heading', { name: '身份验证' }).waitFor();
  assert.equal(await page.getByRole('navigation').count(), 0);
  await page.getByLabel('用户名').fill(username);
  await page.getByLabel('密码').fill('safe-password');
  await page.getByRole('button', { name: '登录' }).click();
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

const assertRenderedWorkspace = async (page) => {
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

  assert.equal(geometry.hasPrimaryContent, true);
  assert.ok(geometry.primaryWidth > 0 && geometry.primaryHeight > 0);
  assert.equal(geometry.pageFitsViewport, true);
  assert.deepEqual(geometry.overflowingControls, []);
  assert.deepEqual(geometry.escapedControls, []);
  assert.deepEqual(geometry.clippedText, []);
  assert.deepEqual(geometry.overlappingText, []);
  const screenshot = await page.screenshot({ animations: 'disabled' });
  assert.ok(screenshot.byteLength > 1000);
  await page.waitForTimeout(300);
  const settledRegions = await readStableRegions(page);
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
  test(`System Administrator completes the authorized workspace journey on ${runtime.name}`, { timeout: 30000 }, async (t) => {
    const { page, baseUrl } = await startWorkbench(t, runtime);
    await installStreamingResponse(page);

    await signIn(page, baseUrl, 'operator');
    await page.getByPlaceholder('请输入需要检索的问题').fill('部署前需要做什么？');
    await page.getByRole('button', { name: '发送' }).click();
    const diagnostics = page.getByLabel('检索诊断');
    await diagnostics.waitFor();
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await diagnostics.getByText('检索诊断', { exact: true }).click();
    await diagnostics.getByText('候选数', { exact: true }).waitFor();
    assert.equal(await diagnostics.getByText('候选数', { exact: true }).isVisible(), true);
    await assertRenderedWorkspace(page);

    for (const workspace of [
      { link: '文档库', heading: '文档库' },
      { link: '构建任务', heading: '构建任务' },
      { link: '系统设置', heading: '系统设置' }
    ]) {
      await page.getByRole('link', { name: workspace.link }).click();
      await page.getByRole('heading', { name: workspace.heading }).waitFor();
      await assertRenderedWorkspace(page);
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
    const { page, baseUrl } = await startWorkbench(t, { ...runtime, role: 'user' });
    await installStreamingResponse(page);

    await signIn(page, baseUrl, 'knowledge-user');
    assert.equal(await page.getByRole('link', { name: '文档库' }).count(), 0);
    assert.equal(await page.getByRole('link', { name: '构建任务' }).count(), 0);
    assert.equal(await page.getByRole('link', { name: '系统设置' }).count(), 0);

    const rail = page.getByRole('complementary', { name: '最近会话' });
    await rail.waitFor();
    assert.equal(await rail.getByText('session-acceptance').isVisible(), true);
    await page.getByPlaceholder('请输入需要检索的问题').fill('部署前需要做什么？');
    await page.getByRole('button', { name: '发送' }).click();

    const summary = page.getByLabel('证据摘要');
    await summary.waitFor();
    assert.equal(await summary.getByText('证据充分').isVisible(), true);
    assert.equal(await page.getByLabel('检索诊断').count(), 0);
    assert.equal(await page.getByText('raw trace remains administrator-only').count(), 0);
    const sourceButton = summary.getByRole('button', { name: '查看来源 deploy-runbook.md' });
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
    assert.equal(await page.evaluate(() => document.activeElement?.getAttribute('aria-label')), '查看来源 deploy-runbook.md');

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
