import assert from 'node:assert/strict';
import test from 'node:test';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const jsonResponse = (data, status = 200) => ({
  status,
  contentType: 'application/json',
  body: JSON.stringify({ data })
});

const startConversationWorkspace = async (t, apiHandler) => {
  const testPort = 42000 + Math.floor(Math.random() * 1000);
  const server = await createServer({ server: { host: '127.0.0.1', port: testPort, strictPort: true } });
  await server.listen();

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  page.setDefaultTimeout(5000);
  await page.route((url) => url.pathname.startsWith('/api/'), apiHandler);

  t.after(async () => {
    await browser.close();
    await server.close();
  });

  return { page, baseUrl: server.resolvedUrls.local[0] };
};

test('Conversation Workspace exposes recent sessions as a contextual rail with only stored metadata', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startConversationWorkspace(t, async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'lin', role: 'user' }));
      return;
    }
    if (path === '/api/sessions') {
      await route.fulfill(
        jsonResponse({
          sessions: [
            { session_id: 'session-20260730', updated_at: '2026-07-30T09:00:00Z', message_count: 4 }
          ]
        })
      );
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await page.addInitScript(() => localStorage.setItem('access_token', 'knowledge-user-token'));
  await page.goto(`${baseUrl}chat`);

  const rail = page.getByRole('complementary', { name: '最近会话' });
  await rail.waitFor();
  assert.equal(await rail.getByText('session-20260730').isVisible(), true);
  assert.equal(await rail.getByText('4 条消息').isVisible(), true);
  assert.equal(await rail.getByRole('button', { name: /session-20260730.*2026.*7.*30/ }).isVisible(), true);
  assert.equal(await rail.getByRole('button', { name: /^session-20260730/ }).getAttribute('aria-current'), null);
  assert.equal(await rail.getByText('标题').count(), 0);

  const [railBox, titleBox, composerBox] = await Promise.all([
    rail.boundingBox(),
    page.getByRole('heading', { name: '对话工作区' }).boundingBox(),
    page.getByPlaceholder('请输入需要检索的问题').boundingBox()
  ]);
  assert.ok(railBox.x + railBox.width <= titleBox.x);
  assert.ok(composerBox.x >= titleBox.x);
  assert.ok(composerBox.x + composerBox.width <= 1440);
});

test('Conversation Workspace rejects an empty question with an explicit composer state', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startConversationWorkspace(t, async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'lin', role: 'user' }));
      return;
    }
    if (path === '/api/sessions') {
      await route.fulfill(jsonResponse({ sessions: [] }));
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await page.addInitScript(() => localStorage.setItem('access_token', 'knowledge-user-token'));
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await page.getByRole('button', { name: '发送' }).click();
  assert.equal(await page.getByRole('alert').innerText(), '请输入问题后再发送。');
});

test('Conversation Workspace reports streaming and completed answers through the authenticated SSE workflow', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startConversationWorkspace(t, async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'lin', role: 'user' }));
      return;
    }
    if (path === '/api/sessions') {
      await route.fulfill(jsonResponse({ sessions: [] }));
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await page.addInitScript(() => {
    const nativeFetch = window.fetch.bind(window);
    window.__streamPayload = null;
    window.fetch = async (input, init) => {
      if (!String(input).includes('/api/chat/stream')) return nativeFetch(input, init);
      window.__streamPayload = JSON.parse(init.body);
      const encoder = new TextEncoder();
      return new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(encoder.encode('event: content\ndata: {"content":"这是流式答案。"}\n\n'));
            setTimeout(() => {
              controller.enqueue(encoder.encode('event: done\ndata: [DONE]\n\n'));
              controller.close();
            }, 200);
          }
        }),
        { status: 200, headers: { 'Content-Type': 'text/event-stream' } }
      );
    };
  });
  await page.addInitScript(() => localStorage.setItem('access_token', 'knowledge-user-token'));
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await page.getByPlaceholder('请输入需要检索的问题').fill('请说明部署流程');
  await page.getByRole('button', { name: '发送' }).click();

  await page.getByRole('status').filter({ hasText: '正在生成回答' }).waitFor();
  const streamingBox = await page.getByLabel('助手消息').boundingBox();
  assert.equal(await page.getByRole('button', { name: '发送' }).isDisabled(), true);
  assert.equal(await page.getByText('这是流式答案。').isVisible(), true);

  await page.getByRole('status').filter({ hasText: '已完成' }).waitFor();
  const completedBox = await page.getByLabel('助手消息').boundingBox();
  assert.equal(Math.abs(completedBox.height - streamingBox.height) <= 1, true);
  assert.match((await page.evaluate(() => window.__streamPayload)).session_id, /^session_\d+$/);
});

test('stopping a streamed answer preserves partial content and marks it incomplete', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startConversationWorkspace(t, async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'lin', role: 'user' }));
      return;
    }
    if (path === '/api/sessions') {
      await route.fulfill(jsonResponse({ sessions: [] }));
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await page.addInitScript(() => {
    const nativeFetch = window.fetch.bind(window);
    window.fetch = async (input, init) => {
      if (!String(input).includes('/api/chat/stream')) return nativeFetch(input, init);
      const encoder = new TextEncoder();
      return new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(encoder.encode('event: content\ndata: {"content":"已收到的内容。"}\n\n'));
            init.signal.addEventListener('abort', () => controller.error(new DOMException('aborted', 'AbortError')));
          }
        }),
        { status: 200, headers: { 'Content-Type': 'text/event-stream' } }
      );
    };
  });
  await page.addInitScript(() => localStorage.setItem('access_token', 'knowledge-user-token'));
  await page.goto(`${baseUrl}chat`);
  await page.getByPlaceholder('请输入需要检索的问题').fill('请开始回答');
  await page.getByRole('button', { name: '发送' }).click();
  await page.getByText('已收到的内容。', { exact: true }).waitFor();

  await page.getByRole('button', { name: '停止' }).click();
  await page.getByRole('status').filter({ hasText: '回答已停止，内容不完整' }).waitFor();
  assert.equal(await page.getByText('已收到的内容。', { exact: true }).isVisible(), true);
});

test('a failed answer explains that it can be retried and retries the same question', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startConversationWorkspace(t, async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'lin', role: 'user' }));
      return;
    }
    if (path === '/api/sessions') {
      await route.fulfill(jsonResponse({ sessions: [] }));
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await page.addInitScript(() => {
    const nativeFetch = window.fetch.bind(window);
    const encoder = new TextEncoder();
    window.__streamRequests = [];
    window.fetch = async (input, init) => {
      if (!String(input).includes('/api/chat/stream')) return nativeFetch(input, init);
      window.__streamRequests.push(JSON.parse(init.body));
      if (window.__streamRequests.length === 1) {
        return new Response(JSON.stringify({ code: 'UPSTREAM_UNAVAILABLE', message: '上游服务暂不可用' }), {
          status: 503,
          headers: { 'Content-Type': 'application/json' }
        });
      }
      return new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(encoder.encode('event: content\ndata: {"content":"重试完成。"}\n\n'));
            controller.enqueue(encoder.encode('event: done\ndata: [DONE]\n\n'));
            controller.close();
          }
        }),
        { status: 200, headers: { 'Content-Type': 'text/event-stream' } }
      );
    };
  });
  await page.addInitScript(() => localStorage.setItem('access_token', 'knowledge-user-token'));
  await page.goto(`${baseUrl}chat`);
  await page.getByPlaceholder('请输入需要检索的问题').fill('请说明应急流程');
  await page.getByRole('button', { name: '发送' }).click();

  await page.getByRole('status').filter({ hasText: '回答失败，可重试' }).waitFor();
  assert.equal(await page.getByText('请求失败：上游服务暂不可用').isVisible(), true);
  await page.getByRole('button', { name: '重试' }).click();
  await page.getByText('重试完成。').waitFor();
  assert.equal(await page.evaluate(() => window.__streamRequests.length), 2);
});

test('an evidence gate rejection is shown as insufficient evidence instead of a response failure', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startConversationWorkspace(t, async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'lin', role: 'user' }));
      return;
    }
    if (path === '/api/sessions') {
      await route.fulfill(jsonResponse({ sessions: [] }));
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await page.addInitScript(() => {
    const nativeFetch = window.fetch.bind(window);
    window.fetch = async (input, init) => {
      if (!String(input).includes('/api/chat/stream')) return nativeFetch(input, init);
      return new Response(
        new ReadableStream({
          start(controller) {
            const encoder = new TextEncoder();
            controller.enqueue(
              encoder.encode(
                'event: evidence_summary\ndata: {"evidence_summary":{"coverage":"insufficient","source_count":0,"sources":[]}}\n\n'
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
  await page.addInitScript(() => localStorage.setItem('access_token', 'knowledge-user-token'));
  await page.goto(`${baseUrl}chat`);
  await page.getByPlaceholder('请输入需要检索的问题').fill('没有证据的问题');
  await page.getByRole('button', { name: '发送' }).click();

  await page.getByRole('status').filter({ hasText: '证据不足' }).waitFor();
  assert.equal(await page.getByText('未检索到足够相关的知识片段，请补充更具体的问题或关键词。').isVisible(), true);
  assert.equal(await page.getByText('回答失败，可重试').count(), 0);
});

test('Conversation Sessions load in chronological order, reset without a title, and remain visible until deletion confirms', { timeout: 30000 }, async (t) => {
  let deleted = false;
  let resolveDelete;
  const deleteConfirmed = new Promise((resolve) => {
    resolveDelete = resolve;
  });
  const { page, baseUrl } = await startConversationWorkspace(t, async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'lin', role: 'user' }));
      return;
    }
    if (path === '/api/sessions' && route.request().method() === 'GET') {
      await route.fulfill(
        jsonResponse({
          sessions: deleted
            ? []
            : [{ session_id: 'session-history', updated_at: '2026-07-30T09:00:00Z', message_count: 2 }]
        })
      );
      return;
    }
    if (path === '/api/sessions/session-history' && route.request().method() === 'GET') {
      await route.fulfill(
        jsonResponse({
          session_id: 'session-history',
          messages: [
            { type: 'user', content: '历史问题' },
            { type: 'assistant', content: '历史回答' }
          ]
        })
      );
      return;
    }
    if (path === '/api/sessions/session-history' && route.request().method() === 'DELETE') {
      await deleteConfirmed;
      deleted = true;
      await route.fulfill(jsonResponse({ session_id: 'session-history', deleted: true }));
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await page.addInitScript(() => localStorage.setItem('access_token', 'knowledge-user-token'));
  await page.goto(`${baseUrl}chat`);
  const rail = page.getByRole('complementary', { name: '最近会话' });
  await rail.getByRole('button', { name: /^session-history/ }).click();
  const userMessage = page.getByLabel('用户消息');
  const assistantMessage = page.getByLabel('助手消息');
  await assistantMessage.waitFor();
  assert.match(await userMessage.innerText(), /你\s+历史问题/);
  assert.match(await assistantMessage.innerText(), /助手\s+历史回答/);

  await page.getByRole('button', { name: '新建会话' }).first().click();
  await page.getByRole('heading', { name: '从团队知识开始提问' }).waitFor();
  assert.equal(await page.getByText('标题').count(), 0);

  await rail.getByRole('button', { name: /^session-history/ }).click();
  await rail.getByRole('button', { name: '删除会话 session-history' }).click();
  const dialog = page.getByRole('dialog');
  await dialog.getByRole('button', { name: '删除' }).click();
  assert.equal(await rail.getByText('session-history').isVisible(), true);

  resolveDelete();
  await rail.getByText('session-history').waitFor({ state: 'detached' });
});

test('a backend rejection of session deletion keeps the selected Conversation Session visible', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startConversationWorkspace(t, async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'lin', role: 'user' }));
      return;
    }
    if (path === '/api/sessions' && route.request().method() === 'GET') {
      await route.fulfill(jsonResponse({ sessions: [{ session_id: 'session-retained', updated_at: '2026-07-30T09:00:00Z', message_count: 2 }] }));
      return;
    }
    if (path === '/api/sessions/session-retained' && route.request().method() === 'GET') {
      await route.fulfill(jsonResponse({ messages: [{ type: 'user', content: '仍在查看的问题' }, { type: 'assistant', content: '仍在查看的回答' }] }));
      return;
    }
    if (path === '/api/sessions/session-retained' && route.request().method() === 'DELETE') {
      await route.fulfill(jsonResponse({ session_id: 'session-retained', deleted: false }));
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await page.addInitScript(() => localStorage.setItem('access_token', 'knowledge-user-token'));
  await page.goto(`${baseUrl}chat`);
  const rail = page.getByRole('complementary', { name: '最近会话' });
  await rail.getByRole('button', { name: /^session-retained/ }).click();
  await page.getByText('仍在查看的回答').waitFor();

  await rail.getByRole('button', { name: '删除会话 session-retained' }).click();
  await page.getByRole('dialog').getByRole('button', { name: '删除' }).click();
  await page.getByRole('alert').waitFor();
  assert.equal(await rail.getByText('session-retained').isVisible(), true);
  assert.equal(await page.getByText('仍在查看的回答').isVisible(), true);
});

test('the contextual session rail collapses before the reading column and Conversation Workspace remains usable on mobile', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startConversationWorkspace(t, async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'lin', role: 'user' }));
      return;
    }
    if (path === '/api/sessions') {
      await route.fulfill(jsonResponse({ sessions: [] }));
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await page.addInitScript(() => localStorage.setItem('access_token', 'knowledge-user-token'));
  await page.setViewportSize({ width: 1024, height: 900 });
  await page.goto(`${baseUrl}chat`);
  const rail = page.getByRole('complementary', { name: '最近会话' });
  assert.equal(await rail.isVisible(), false);
  assert.equal(await page.getByRole('button', { name: '会话', exact: true }).isVisible(), true);
  const [compactTitleBox, compactComposerBox] = await Promise.all([
    page.getByRole('heading', { name: '对话工作区' }).boundingBox(),
    page.getByPlaceholder('请输入需要检索的问题').boundingBox()
  ]);
  assert.equal(compactComposerBox.x >= compactTitleBox.x, true);
  assert.equal(compactComposerBox.x + compactComposerBox.width <= 1024, true);

  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole('button', { name: '会话', exact: true }).click();
  assert.equal(await page.getByRole('dialog').isVisible(), true);
  assert.equal(await page.getByPlaceholder('请输入需要检索的问题').isVisible(), true);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
});

test('the empty Conversation Workspace names the supported internal knowledge scope', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startConversationWorkspace(t, async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'lin', role: 'user' }));
      return;
    }
    if (path === '/api/sessions') {
      await route.fulfill(jsonResponse({ sessions: [] }));
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await page.addInitScript(() => localStorage.setItem('access_token', 'knowledge-user-token'));
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '从团队知识开始提问' }).waitFor();
  assert.equal(await page.getByText('可查询部署规范、事故手册、产品决策和运行流程。').isVisible(), true);
});

test('a completed response remains readable without overflow at a 390-pixel viewport', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startConversationWorkspace(t, async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'lin', role: 'user' }));
      return;
    }
    if (path === '/api/sessions') {
      await route.fulfill(jsonResponse({ sessions: [] }));
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await page.addInitScript(() => {
    const nativeFetch = window.fetch.bind(window);
    window.fetch = async (input, init) => {
      if (!String(input).includes('/api/chat/stream')) return nativeFetch(input, init);
      return new Response(
        new ReadableStream({
          start(controller) {
            const encoder = new TextEncoder();
            controller.enqueue(encoder.encode('event: content\ndata: {"content":"移动端可以阅读这段完整回答。"}\n\n'));
            controller.enqueue(encoder.encode('event: done\ndata: [DONE]\n\n'));
            controller.close();
          }
        }),
        { status: 200, headers: { 'Content-Type': 'text/event-stream' } }
      );
    };
  });
  await page.addInitScript(() => localStorage.setItem('access_token', 'knowledge-user-token'));
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(`${baseUrl}chat`);
  await page.getByPlaceholder('请输入需要检索的问题').fill('移动端问题');
  await page.getByRole('button', { name: '发送' }).click();
  await page.getByText('移动端可以阅读这段完整回答。').waitFor();
  await page.getByRole('status').filter({ hasText: '已完成' }).waitFor();
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
});

test('a Knowledge User can inspect Evidence Summary source excerpts without exposing Retrieval Diagnostics', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startConversationWorkspace(t, async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'lin', role: 'user' }));
      return;
    }
    if (path === '/api/sessions') {
      await route.fulfill(jsonResponse({ sessions: [] }));
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await page.addInitScript(() => {
    const nativeFetch = window.fetch.bind(window);
    window.fetch = async (input, init) => {
      if (!String(input).includes('/api/chat/stream')) return nativeFetch(input, init);
      const encoder = new TextEncoder();
      return new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(encoder.encode('event: content\ndata: {"content":"部署前需要完成审批。"}\n\n'));
            controller.enqueue(
              encoder.encode(
                'event: evidence_summary\ndata: {"evidence_summary":{"coverage":"sufficient","source_count":1,"sources":[{"source_id":"chunk-deploy-1","metadata":{"filename":"deploy-runbook.md"},"excerpt":"发布前必须由值班负责人完成变更审批。"}]}}\n\n'
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
  await page.addInitScript(() => localStorage.setItem('access_token', 'knowledge-user-token'));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(`${baseUrl}chat`);
  await page.getByPlaceholder('请输入需要检索的问题').fill('部署前需要做什么？');
  await page.getByRole('button', { name: '发送' }).click();

  const summary = page.getByLabel('证据摘要');
  await summary.waitFor();
  assert.equal(await summary.getByText('证据充分').isVisible(), true);
  assert.equal(await summary.getByText('1 个来源').isVisible(), true);
  assert.equal(await page.getByText('RAG Trace').count(), 0);

  const sourceButton = summary.getByRole('button', { name: '查看来源 deploy-runbook.md' });
  await sourceButton.click();
  const excerptDrawer = page.getByRole('complementary', { name: '来源摘录' });
  await excerptDrawer.waitFor();
  assert.equal(await excerptDrawer.getByText('发布前必须由值班负责人完成变更审批。').isVisible(), true);

  const [answerBox, drawerBox] = await Promise.all([
    page.getByLabel('助手消息').boundingBox(),
    excerptDrawer.boundingBox()
  ]);
  assert.equal(answerBox.x + answerBox.width <= drawerBox.x, true);

  await page.keyboard.press('Escape');
  await excerptDrawer.waitFor({ state: 'detached' });
  assert.equal(await page.evaluate(() => document.activeElement?.getAttribute('aria-label')), '查看来源 deploy-runbook.md');

  await page.setViewportSize({ width: 1024, height: 900 });
  await sourceButton.click();
  const compactDrawer = page.getByRole('complementary', { name: '来源摘录' });
  await compactDrawer.waitFor();
  const [compactAnswerBox, compactDrawerBox] = await Promise.all([
    page.getByLabel('助手消息').boundingBox(),
    compactDrawer.boundingBox()
  ]);
  assert.equal(compactDrawerBox.y >= compactAnswerBox.y + compactAnswerBox.height, true);
});

test('historical Evidence Summaries retain source identity and show unavailable or insufficient coverage honestly', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startConversationWorkspace(t, async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/auth/me') {
      await route.fulfill(jsonResponse({ username: 'lin', role: 'user' }));
      return;
    }
    if (path === '/api/sessions' && route.request().method() === 'GET') {
      await route.fulfill(
        jsonResponse({ sessions: [{ session_id: 'session-evidence-history', updated_at: '2026-07-30T10:00:00Z', message_count: 4 }] })
      );
      return;
    }
    if (path === '/api/sessions/session-evidence-history') {
      await route.fulfill(
        jsonResponse({
          messages: [
            { type: 'user', content: '历史的部署问题' },
            {
              type: 'assistant',
              content: '历史回答有可核对来源。',
              evidence_summary: {
                coverage: 'sufficient',
                source_count: 1,
                sources: [{ source_id: 'chunk-history-7', metadata: {}, excerpt: '历史来源摘录。' }]
              }
            },
            {
              type: 'assistant',
              content: '历史回答没有可用来源。',
              evidence_summary: { coverage: 'unavailable', source_count: 0, sources: [] }
            },
            {
              type: 'assistant',
              content: '历史回答证据不足。',
              evidence_summary: { coverage: 'insufficient', source_count: 0, sources: [] }
            }
          ]
        })
      );
      return;
    }
    await route.fulfill(jsonResponse({ message: `Unexpected request: ${path}` }, 404));
  });

  await page.addInitScript(() => localStorage.setItem('access_token', 'knowledge-user-token'));
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('complementary', { name: '最近会话' }).getByRole('button', { name: /^session-evidence-history/ }).click();

  const sourceAnswer = page.getByLabel('助手消息').filter({ hasText: '历史回答有可核对来源。' });
  const sourceButton = sourceAnswer.getByRole('button', { name: '查看来源 chunk-history-7' });
  await sourceButton.click();
  await page.getByRole('complementary', { name: '来源摘录' }).getByText('历史来源摘录。').waitFor();
  await page.getByRole('button', { name: '关闭来源摘录' }).click();

  const unavailableAnswer = page.getByLabel('助手消息').filter({ hasText: '历史回答没有可用来源。' });
  const insufficientAnswer = page.getByLabel('助手消息').filter({ hasText: '历史回答证据不足。' });
  assert.equal(await unavailableAnswer.getByText('证据不可用').isVisible(), true);
  assert.equal(await unavailableAnswer.getByText('没有可供核对的来源摘录。').isVisible(), true);
  assert.equal(await insufficientAnswer.getByText('证据不足', { exact: true }).isVisible(), true);
});
