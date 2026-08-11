// Conversation Workspace browser acceptance over a disposable real API.
// Roles come from the server; chat answers use the real deterministic LLM and
// real SSE streaming over normal HTTP network traffic.
import assert from 'node:assert/strict';
import test from 'node:test';
import { loginAdmin, registerKnowledgeUserViaApi, startWorkbench } from './acceptance-env.mjs';

const sendQuestion = async (page, question) => {
  await page.getByPlaceholder('请输入需要检索的问题').fill(question);
  await page.getByRole('button', { name: '发送' }).click();
};

/** Start a workbench, register a real Knowledge User, and store its real token. */
const startAsKnowledgeUser = async (t, { env = {}, viewport } = {}) => {
  const workbench = await startWorkbench(t, { env, viewport });
  const token = await registerKnowledgeUserViaApi(workbench.api, 'lin');
  await workbench.page.addInitScript(
    ({ storedToken }) => localStorage.setItem('access_token', storedToken),
    { storedToken: token }
  );
  return { ...workbench, token };
};

test('Conversation Workspace exposes recent sessions as a contextual rail with only stored metadata', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.goto(`${baseUrl}chat`);

  const rail = page.getByRole('complementary', { name: '最近会话' });
  await rail.waitFor();
  // The session rail loads asynchronously after navigation.
  await rail.getByText('session-evidence-history').waitFor();
  assert.equal(await rail.getByText('session-evidence-history').isVisible(), true);
  assert.equal(await rail.getByText('4 条消息').isVisible(), true);
  assert.equal(await rail.getByRole('button', { name: /^session-evidence-history/ }).getAttribute('aria-current'), null);
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
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await page.getByRole('button', { name: '发送' }).click();
  assert.equal(await page.getByRole('alert').innerText(), '请输入问题后再发送。');
});

test('Conversation Workspace reports streaming and completed answers through the authenticated SSE workflow', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {
    env: { BROWSER_ACCEPTANCE_LLM_DELAY_MS: '1500' }
  });
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await sendQuestion(page, '部署前需要做什么？');

  await page.getByText('部署前需要完成变更审批。').waitFor();
  await page.getByRole('status').filter({ hasText: '已完成' }).waitFor();
  const completedBox = await page.getByLabel('助手消息').boundingBox();
  assert.equal(completedBox.height > 0, true);
});

test('stopping a streamed answer marks it incomplete without fabricated content', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {
    env: { BROWSER_ACCEPTANCE_LLM_DELAY_MS: '4000' }
  });
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await sendQuestion(page, '部署前需要做什么？');
  await page.getByRole('button', { name: '停止' }).waitFor();
  await page.getByRole('button', { name: '停止' }).click();

  await page.getByRole('status').filter({ hasText: '回答已停止，内容不完整' }).waitFor();
  assert.equal(await page.getByText('回答已停止，未生成可保留的内容。', { exact: true }).isVisible(), true);
});

test('a generation-unavailable answer stays fail-closed and a retried question succeeds', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {
    env: { BROWSER_ACCEPTANCE_FAIL_FIRST: '1' }
  });
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await sendQuestion(page, '部署前需要做什么？');
  await page.getByText('【生成不可用】生成服务暂不可用，请稍后重试。').waitFor();

  await sendQuestion(page, '部署前需要做什么？');
  await page.getByText('部署前需要完成变更审批。').waitFor();
  assert.equal(await page.getByText('【生成不可用】生成服务暂不可用，请稍后重试。').count(), 1);
});

test('an evidence gate rejection is shown as insufficient evidence instead of a response failure', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await sendQuestion(page, 'zzzz 完全不存在的内容 987654321');
  await page.getByRole('status').filter({ hasText: '证据不足' }).waitFor();
  assert.equal(await page.getByText('未检索到足够相关的知识片段，请补充更具体的问题或关键词。').isVisible(), true);
  assert.equal(await page.getByText('回答失败，可重试').count(), 0);
});

test('Conversation Sessions load in chronological order, reset without a title, and remain visible until deletion confirms', { timeout: 30000 }, async (t) => {
  const { page, baseUrl, api, token } = await startAsKnowledgeUser(t, {});
  const chat = await fetch(`${api.baseUrl}/chat`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: '为会话列表创建一条消息' })
  });
  assert.equal(chat.status, 200);

  await page.goto(`${baseUrl}chat`);
  const rail = page.getByRole('complementary', { name: '最近会话' });
  await rail.getByRole('button', { name: /^session-evidence-history/ }).click();
  const userMessage = page.getByLabel('用户消息');
  const assistantMessage = page.getByLabel('助手消息');
  await assistantMessage.first().waitFor();
  assert.match(await userMessage.first().innerText(), /你\s+历史的部署问题/);
  assert.match(await assistantMessage.first().innerText(), /助手\s+历史回答有可核对来源。/);

  await page.getByRole('button', { name: '新建会话' }).first().click();
  await page.getByRole('heading', { name: '从团队知识开始提问' }).waitFor();
  assert.equal(await page.getByText('标题').count(), 0);

  await rail.getByRole('button', { name: /^session-evidence-history/ }).click();
  await rail.getByRole('button', { name: '删除会话 session-evidence-history' }).click();
  const dialog = page.getByRole('dialog');
  await dialog.getByRole('button', { name: '删除' }).click();
  await rail.getByText('session-evidence-history').waitFor({ state: 'detached' });
  assert.equal(await rail.getByText('session-evidence-history').count(), 0);
});

test('the contextual session rail collapses before the reading column and Conversation Workspace remains usable on mobile', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
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
  const { page, baseUrl } = await startAsKnowledgeUser(t, { env: { BROWSER_ACCEPTANCE_SEED: 'minimal' } });
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '从团队知识开始提问' }).waitFor();
  assert.equal(await page.getByText('可查询部署规范、事故手册、产品决策和运行流程。').isVisible(), true);
});

test('a completed response remains readable without overflow at a 390-pixel viewport', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, { viewport: { width: 390, height: 844 } });
  await page.goto(`${baseUrl}chat`);
  await sendQuestion(page, '部署前需要做什么？');
  await page.getByText('部署前需要完成变更审批。').waitFor();
  await page.getByRole('status').filter({ hasText: '已完成' }).waitFor();
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
});

test('a Knowledge User can inspect Evidence Summary source excerpts without exposing Retrieval Diagnostics', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, { viewport: { width: 1440, height: 900 } });
  await page.goto(`${baseUrl}chat`);
  await sendQuestion(page, '部署前需要做什么？');

  const summary = page.getByLabel('证据摘要');
  await summary.waitFor();
  assert.equal(await summary.getByText('证据充分').isVisible(), true);
  assert.equal(await summary.getByText(/(\d+) 个来源/).isVisible(), true);
  assert.equal(await page.getByText('RAG Trace').count(), 0);
  assert.equal(await page.getByLabel('检索诊断').count(), 0);

  const sourceButton = summary.getByRole('button', { name: '查看来源 browser-evidence.md' });
  await sourceButton.click();
  const excerptDrawer = page.getByRole('complementary', { name: '来源摘录' });
  await excerptDrawer.waitFor();
  assert.equal(await excerptDrawer.getByText('部署前需要完成变更审批。').isVisible(), true);

  const [answerBox, drawerBox] = await Promise.all([
    page.getByLabel('助手消息').boundingBox(),
    excerptDrawer.boundingBox()
  ]);
  assert.equal(answerBox.x + answerBox.width <= drawerBox.x, true);

  await page.keyboard.press('Escape');
  await excerptDrawer.waitFor({ state: 'detached' });
  assert.equal(await page.evaluate(() => document.activeElement?.getAttribute('aria-label')), '查看来源 browser-evidence.md');
});

test('historical Evidence Summaries retain source identity and show unavailable or insufficient coverage honestly', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('complementary', { name: '最近会话' }).getByRole('button', { name: /^session-evidence-history/ }).click();

  const sourceAnswer = page.getByLabel('助手消息').filter({ hasText: '历史回答有可核对来源。' });
  const sourceButton = sourceAnswer.getByRole('button', { name: '查看来源 deploy-runbook.md' });
  await sourceButton.click();
  await page.getByRole('complementary', { name: '来源摘录' }).getByText('历史来源摘录。').waitFor();
  await page.getByRole('button', { name: '关闭来源摘录' }).click();

  const unavailableAnswer = page.getByLabel('助手消息').filter({ hasText: '历史回答没有可用来源。' });
  const insufficientAnswer = page.getByLabel('助手消息').filter({ hasText: '历史回答证据不足。' });
  assert.equal(await unavailableAnswer.getByText('证据不可用').isVisible(), true);
  assert.equal(await unavailableAnswer.getByText('没有可供核对的来源摘录。').isVisible(), true);
  assert.equal(await insufficientAnswer.getByText('证据不足', { exact: true }).isVisible(), true);
  assert.equal(await page.getByLabel('检索诊断').count(), 0);
});

test('a System Administrator can expand bounded Retrieval Diagnostics for a live answer', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, {});
  await loginAdmin(page, baseUrl);

  await sendQuestion(page, '部署前需要做什么？');

  const diagnostics = page.getByLabel('检索诊断');
  await diagnostics.waitFor();
  const disclosure = diagnostics.locator('details');
  assert.equal(await disclosure.getAttribute('open'), null);

  const summary = diagnostics.locator('summary');
  await summary.focus();
  await page.keyboard.press('Enter');
  assert.equal(await disclosure.getAttribute('open'), '');
  const diagnosticsContent = diagnostics.locator('.retrieval-diagnostics__content');
  await page.emulateMedia({ reducedMotion: 'reduce' });
  assert.equal(await diagnosticsContent.evaluate((element) => getComputedStyle(element).animationName), 'none');
  assert.equal(await diagnostics.getByText('检索时间线').isVisible(), true);
  assert.equal(await diagnostics.getByText('retrieve', { exact: true }).isVisible(), true);
  assert.equal(await diagnostics.getByText(/召回候选 \d+/).isVisible(), true);
  assert.equal(await diagnostics.getByText('门禁通过').isVisible(), true);
  assert.equal(await diagnostics.getByText('脱敏 trace 预览').isVisible(), true);
});

test('a System Administrator sees unavailable Retrieval Diagnostics fields for historical answers', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, {});
  await loginAdmin(page, baseUrl);
  await page.getByRole('complementary', { name: '最近会话' }).getByRole('button', { name: /^session-admin-diagnostics/ }).click();

  const diagnostics = page.getByLabel('检索诊断');
  await diagnostics.waitFor();
  const disclosure = diagnostics.locator('details');
  assert.equal(await disclosure.getAttribute('open'), null);
  await diagnostics.locator('summary').click();
  await diagnostics.getByText('未返回检索时间线。').waitFor();
  assert.equal(await diagnostics.getByText('未返回检索时间线。').isVisible(), true);
  assert.equal(await diagnostics.getByText('召回候选 不可用').isVisible(), true);
  assert.equal(await diagnostics.getByText('重排候选 不可用').isVisible(), true);
  assert.equal(await diagnostics.getByText('门禁不可用').isVisible(), true);
  assert.equal(await diagnostics.getByText('回退状态不可用').isVisible(), true);
  assert.equal(await diagnostics.getByText('未返回 trace 预览。').isVisible(), true);
});
