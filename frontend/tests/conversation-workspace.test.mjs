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

const completedInsufficientEvents = ({
  answerId,
  executionId,
  qcsId,
  question,
  conditions = [],
  answerText = '已完成的回答。',
  includeInsufficientEvidenceReply = true
}) => {
  const evidenceSummary = { coverage: 'insufficient', source_count: 0, sources: [] };
  const insufficientEvidenceReply = {
    outcome: 'insufficient_evidence_reply',
    reason: 'decision_not_covered',
    query_condition_set_identity: qcsId
  };
  const execution = {
    id: executionId,
    assistant_message_id: answerId,
    state: 'completed',
    question,
    answer_text: answerText,
    query_condition_set: {
      identity: qcsId,
      normalized_question: question,
      conditions
    },
    condition_provenance: { mode: conditions.length ? 'explicit' : 'question_normalized' },
    outcome: 'insufficient_evidence_reply',
    evidence_set_identity: null,
    item_identities: [],
    snapshot_ids: [],
    knowledge_version_identities: [],
    insufficient_evidence_reply: insufficientEvidenceReply,
    evidence_summary: evidenceSummary
  };
  const events = [
    `event: answer_identity\ndata: ${JSON.stringify({ answer_id: answerId })}\n\n`,
    `event: answer_execution\ndata: ${JSON.stringify({ answer_execution: execution })}\n\n`,
    'event: outcome\ndata: {"outcome":"insufficient_evidence_reply"}\n\n'
  ];
  if (includeInsufficientEvidenceReply) {
    events.push(
      `event: insufficient_evidence_reply\ndata: ${JSON.stringify({
        insufficient_evidence_reply: insufficientEvidenceReply
      })}\n\n`
    );
  }
  events.push(`event: evidence_summary\ndata: ${JSON.stringify({ evidence_summary: evidenceSummary })}\n\n`);
  return events;
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

test('Conversation Workspace keeps Query Conditions visible, editable, inherited, and frozen per turn', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  const composerConditions = page.locator('.composer').getByLabel('查询条件');
  await composerConditions.getByRole('button', { name: '添加查询条件' }).click();
  await composerConditions.getByLabel('条件 1 字段').fill('environment');
  await composerConditions.getByLabel('条件 1 运算符').fill('equals');
  await composerConditions.getByLabel('条件 1 值').fill('production');

  await sendQuestion(page, '部署前需要做什么？');
  const firstQuestion = page.getByLabel('用户消息').last();
  const firstFrozenConditions = firstQuestion.getByLabel('查询条件');
  await firstFrozenConditions.getByText('environment').waitFor();
  assert.equal(await firstFrozenConditions.getByText('production').isVisible(), true);

  await composerConditions.getByText('沿用上一轮条件', { exact: true }).click();
  assert.equal(await composerConditions.getByRole('checkbox', { name: '沿用上一轮条件' }).isChecked(), true);
  await sendQuestion(page, '部署后需要复核什么？');
  const secondQuestion = page.getByLabel('用户消息').last();
  const secondFrozenConditions = secondQuestion.getByLabel('查询条件');
  await secondFrozenConditions.getByText('environment').waitFor();
  assert.equal(await secondFrozenConditions.getByText('production').isVisible(), true);
  assert.equal(await secondFrozenConditions.getByText('继承').isVisible(), true);

  await composerConditions.getByText('沿用上一轮条件', { exact: true }).click();
  assert.equal(await composerConditions.getByRole('checkbox', { name: '沿用上一轮条件' }).isChecked(), false);
  await composerConditions.getByRole('button', { name: '重置查询条件' }).click();
  assert.equal(await composerConditions.getByLabel('条件 1 字段').count(), 0);

  await composerConditions.getByRole('button', { name: '添加查询条件' }).click();
  await composerConditions.getByLabel('条件 1 字段').fill('environment');
  await composerConditions.getByLabel('条件 1 运算符').fill('equals');
  await composerConditions.getByLabel('条件 1 值').fill('staging');
  await page.getByRole('button', { name: '新建会话' }).first().click();
  assert.equal(await composerConditions.getByLabel('条件 1 字段').count(), 0);
  assert.equal(await composerConditions.getByRole('checkbox', { name: '沿用上一轮条件' }).isChecked(), false);

  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForFunction(() => document.documentElement.scrollWidth <= window.innerWidth);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
});

test('a contradictory SSE terminal projection fails instead of becoming a completed answer', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.route('**/api/chat/stream', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: [
        ...completedInsufficientEvents({
          answerId: 'assistant-contradictory',
          executionId: 'answer-execution-contradictory',
          qcsId: 'qcs-contradictory',
          question: '终态事件矛盾时应失败',
          answerText: 'partial answer'
        }),
        'event: outcome\ndata: {"outcome":"evidence_gated_answer"}\n\n',
        'event: content\ndata: {"content":"partial answer"}\n\n',
        'event: evidence_summary\ndata: {"evidence_summary":{"coverage":"sufficient","source_count":1,"sources":[{"snapshot_id":"forged"}]}}\n\n',
        'event: done\ndata: [DONE]\n\n'
      ].join('')
    });
  });
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await sendQuestion(page, '终态事件矛盾时应失败');

  const answer = page.getByLabel('助手消息').last();
  await answer.getByRole('status').filter({ hasText: '回答失败，可重试' }).waitFor();
  assert.equal(await answer.getByRole('status').filter({ hasText: '已完成' }).count(), 0);
  assert.equal(await answer.getByText('partial answer').count(), 0);
});

test('a repeated SSE terminal field fails instead of accepting a duplicated completion', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.route('**/api/chat/stream', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: [
        ...completedInsufficientEvents({
          answerId: 'assistant-duplicated',
          executionId: 'answer-execution-duplicated',
          qcsId: 'qcs-duplicated',
          question: '重复终态必须失败',
          answerText: 'partial answer'
        }),
        'event: outcome\ndata: {"outcome":"insufficient_evidence_reply"}\n\n',
        'event: content\ndata: {"content":"partial answer"}\n\n',
        'event: done\ndata: [DONE]\n\n'
      ].join('')
    });
  });
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await sendQuestion(page, '重复终态必须失败');

  const answer = page.getByLabel('助手消息').last();
  await answer.getByRole('status').filter({ hasText: '回答失败，可重试' }).waitFor();
  assert.equal(await answer.getByRole('status').filter({ hasText: '证据不足' }).count(), 0);
});

test('duplicate JSON terminal fields fail instead of accepting the last overwritten value', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.route('**/api/chat/stream', async (route) => {
    const terminalEvents = completedInsufficientEvents({
      answerId: 'assistant-duplicate-json-field',
      executionId: 'answer-execution-duplicate-json-field',
      qcsId: 'qcs-duplicate-json-field',
      question: '重复 JSON 终态字段必须失败'
    });
    terminalEvents[2] =
      'event: outcome\ndata: {"outcome":"evidence_gated_answer","outcome":"insufficient_evidence_reply"}\n\n';
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: [...terminalEvents, 'event: content\ndata: {"content":"已完成的回答。"}\n\n', 'event: done\ndata: [DONE]\n\n'].join('')
    });
  });
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await sendQuestion(page, '重复 JSON 终态字段必须失败');

  const answer = page.getByLabel('助手消息').last();
  await answer.getByRole('status').filter({ hasText: '回答失败，可重试' }).waitFor();
  assert.equal(await answer.getByRole('status').filter({ hasText: '证据不足' }).count(), 0);
  assert.equal(await answer.getByLabel('证据摘要').count(), 0);
});

test('retrying a failed condition-bound turn resubmits its frozen Query Condition Set', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  const requestPayloads = [];
  await page.route('**/api/chat/stream', async (route) => {
    const requestPayload = JSON.parse(route.request().postData() || '{}');
    requestPayloads.push(requestPayload);
    const [frozenCondition] = requestPayload.query_conditions || [];
    assert.ok(frozenCondition);
    if (requestPayloads.length === 1) {
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: [
          'event: answer_identity\ndata: {"answer_id":"assistant-retry-failed"}\n\n',
          `event: answer_execution\ndata: ${JSON.stringify({
            answer_execution: {
              id: 'answer-execution-retry',
              assistant_message_id: 'assistant-retry-failed',
              state: 'failed',
              question: '条件失败后重试',
              query_condition_set: {
                identity: 'qcs-retry',
                normalized_question: '条件失败后重试',
                conditions: [frozenCondition]
              },
              condition_provenance: { mode: 'explicit' },
              failure_code: 'ANSWER_EXECUTION_FAILED'
            }
          })}\n\n`,
          'event: error\ndata: {"code":"CHAT_STREAM_FAILED","message":"执行失败"}\n\n',
          'event: done\ndata: [DONE]\n\n'
        ].join('')
      });
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: [
        ...completedInsufficientEvents({
          answerId: 'assistant-retry',
          executionId: 'answer-execution-retry-complete',
          qcsId: 'qcs-retry-complete',
          question: '条件失败后重试',
          conditions: [frozenCondition],
          answerText: '已冻结条件后重试。'
        }),
        'event: content\ndata: {"content":"已冻结条件后重试。"}\n\n',
        'event: done\ndata: [DONE]\n\n'
      ].join('')
    });
  });
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  const composerConditions = page.locator('.composer').getByLabel('查询条件');
  await composerConditions.getByRole('button', { name: '添加查询条件' }).click();
  await composerConditions.getByLabel('条件 1 字段').fill('environment');
  await composerConditions.getByLabel('条件 1 运算符').fill('equals');
  await composerConditions.getByLabel('条件 1 值').fill('production');
  await sendQuestion(page, '条件失败后重试');

  const failedAnswer = page.getByLabel('助手消息').last();
  await failedAnswer.getByRole('status').filter({ hasText: '回答失败，可重试' }).waitFor();
  await failedAnswer.getByRole('button', { name: '重试' }).click();
  const retriedAnswer = page.getByLabel('助手消息').last();
  await retriedAnswer.getByRole('status').filter({ hasText: '证据不足' }).waitFor();

  assert.equal(requestPayloads.length, 2);
  assert.deepEqual(requestPayloads[1].query_conditions, [requestPayloads[0].query_conditions[0]]);
  assert.equal(requestPayloads[1].inherit_conditions, undefined);
});

test('a repeated SSE done terminal fails instead of accepting a second completion boundary', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.route('**/api/chat/stream', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: [
        ...completedInsufficientEvents({
          answerId: 'assistant-done-duplicated',
          executionId: 'answer-execution-done-duplicated',
          qcsId: 'qcs-done-duplicated',
          question: '重复 done 必须失败'
        }),
        'event: content\ndata: {"content":"已完成的回答。"}\n\n',
        'event: done\ndata: [DONE]\n\n',
        'event: done\ndata: [DONE]\n\n'
      ].join('')
    });
  });
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await sendQuestion(page, '重复 done 必须失败');

  const answer = page.getByLabel('助手消息').last();
  await answer.getByRole('status').filter({ hasText: '回答失败，可重试' }).waitFor();
  assert.equal(await answer.getByRole('status').filter({ hasText: '证据不足' }).count(), 0);
});

for (const [terminalName, terminalFrames] of [
  [
    'an empty done frame before a valid terminal',
    ['event: done\n\n', 'event: done\ndata: [DONE]\n\n']
  ],
  [
    'a quoted done frame before a valid terminal',
    ['event: done\ndata: "[DONE]"\n\n', 'event: done\ndata: [DONE]\n\n']
  ],
  [
    'a terminal marker on a stage event before a valid terminal',
    ['event: stage\ndata: [DONE]\n\n', 'event: done\ndata: [DONE]\n\n']
  ],
  [
    'a malformed done frame after a valid terminal',
    ['event: done\ndata: [DONE]\n\n', 'event: done\ndata: {"state":"failed"}\n\n']
  ]
]) {
  test(`${terminalName} fails closed instead of preserving a completed projection`, { timeout: 30000 }, async (t) => {
    const { page, baseUrl } = await startAsKnowledgeUser(t, {});
    await page.route('**/api/chat/stream', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: [
          ...completedInsufficientEvents({
            answerId: `assistant-${terminalName}`,
            executionId: `answer-execution-${terminalName}`,
            qcsId: `qcs-${terminalName}`,
            question: '畸形 done 不能保留完成投影'
          }),
          'event: content\ndata: {"content":"已完成的回答。"}\n\n',
          ...terminalFrames
        ].join('')
      });
    });
    await page.goto(`${baseUrl}chat`);
    await page.getByRole('heading', { name: '对话工作区' }).waitFor();

    await sendQuestion(page, '畸形 done 不能保留完成投影');

    const answer = page.getByLabel('助手消息').last();
    await answer.getByRole('status').filter({ hasText: '回答失败，可重试' }).waitFor();
    assert.equal(await answer.getByRole('status').filter({ hasText: '证据不足' }).count(), 0);
    assert.equal(await answer.getByLabel('证据摘要').count(), 0);
  });
}

test('a split SSE terminal frame after done fails instead of accepting the first chunk as final', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  const firstChunk = [
    ...completedInsufficientEvents({
      answerId: 'assistant-split-done',
      executionId: 'answer-execution-split-done',
      qcsId: 'qcs-split-done',
      question: '分块终态必须失败'
    }),
    'event: content\ndata: {"content":"已完成的回答。"}\n\n',
    'event: done\ndata: [DONE]\n\n'
  ].join('');
  const duplicateDone = 'event: done\ndata: [DONE]\n\n';
  await page.addInitScript(
    ({ firstChunk: initial, duplicateDone: repeated }) => {
      const originalFetch = window.fetch.bind(window);
      window.fetch = async (input, init) => {
        const url = typeof input === 'string' ? input : input.url;
        if (!url.includes('/api/chat/stream')) {
          return originalFetch(input, init);
        }
        const encoder = new TextEncoder();
        const body = new ReadableStream({
          start(controller) {
            controller.enqueue(encoder.encode(initial));
            window.setTimeout(() => {
              controller.enqueue(encoder.encode(repeated));
              controller.close();
            }, 20);
          }
        });
        return new Response(body, {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' }
        });
      };
    },
    { firstChunk, duplicateDone }
  );
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await sendQuestion(page, '分块终态必须失败');

  const answer = page.getByLabel('助手消息').last();
  await answer.getByRole('status').filter({ hasText: '回答失败，可重试' }).waitFor();
  assert.equal(await answer.getByRole('status').filter({ hasText: '证据不足' }).count(), 0);
});

test('a semantic SSE frame after done fails instead of changing a completed projection', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.route('**/api/chat/stream', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: [
        ...completedInsufficientEvents({
          answerId: 'assistant-after-done',
          executionId: 'answer-execution-after-done',
          qcsId: 'qcs-after-done',
          question: '终态后内容必须失败'
        }),
        'event: content\ndata: {"content":"已完成的回答。"}\n\n',
        'event: done\ndata: [DONE]\n\n',
        'event: content\ndata: {"content":"不允许在 done 后追加。"}\n\n'
      ].join('')
    });
  });
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await sendQuestion(page, '终态后内容必须失败');

  const answer = page.getByLabel('助手消息').last();
  await answer.getByRole('status').filter({ hasText: '回答失败，可重试' }).waitFor();
  assert.equal(await answer.getByRole('status').filter({ hasText: '证据不足' }).count(), 0);
  assert.equal(await answer.getByLabel('证据摘要').count(), 0);
  assert.equal(await answer.getByText('拒答原因：知识片段不足，建议补充关键词或限定范围。').count(), 0);
});

for (const [terminalName, terminalFrame] of [
  ['an empty semantic SSE frame after done', 'event: outcome\n\n'],
  [
    'an unterminated semantic SSE frame after done',
    'event: outcome\ndata: {"outcome":"evidence_gated_answer"}'
  ]
]) {
  test(`${terminalName} fails instead of preserving a completed projection`, { timeout: 30000 }, async (t) => {
    const { page, baseUrl } = await startAsKnowledgeUser(t, {});
    await page.route('**/api/chat/stream', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: [
          ...completedInsufficientEvents({
            answerId: `assistant-${terminalName}`,
            executionId: `answer-execution-${terminalName}`,
            qcsId: `qcs-${terminalName}`,
            question: '空或截断的终态事件必须失败'
          }),
          'event: content\ndata: {"content":"已完成的回答。"}\n\n',
          'event: done\ndata: [DONE]\n\n',
          terminalFrame
        ].join('')
      });
    });
    await page.goto(`${baseUrl}chat`);
    await page.getByRole('heading', { name: '对话工作区' }).waitFor();

    await sendQuestion(page, '空或截断的终态事件必须失败');

    const answer = page.getByLabel('助手消息').last();
    await answer.getByRole('status').filter({ hasText: '回答失败，可重试' }).waitFor();
    assert.equal(await answer.getByRole('status').filter({ hasText: '证据不足' }).count(), 0);
    assert.equal(await answer.getByLabel('证据摘要').count(), 0);
  });
}

test('a completed SSE execution for a different submitted question fails closed', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.route('**/api/chat/stream', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: [
        ...completedInsufficientEvents({
          answerId: 'assistant-question-drift',
          executionId: 'answer-execution-question-drift',
          qcsId: 'qcs-question-drift',
          question: '伪造的另一个问题'
        }),
        'event: content\ndata: {"content":"已完成的回答。"}\n\n',
        'event: done\ndata: [DONE]\n\n'
      ].join('')
    });
  });
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await sendQuestion(page, '提交的问题必须冻结');

  const answer = page.getByLabel('助手消息').last();
  await answer.getByRole('status').filter({ hasText: '回答失败，可重试' }).waitFor();
  assert.equal(await answer.getByRole('status').filter({ hasText: '证据不足' }).count(), 0);
});

test('a completed SSE execution with a different submitted Query Condition Set fails closed', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  const requestPayloads = [];
  await page.route('**/api/chat/stream', async (route) => {
    requestPayloads.push(JSON.parse(route.request().postData() || '{}'));
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: [
        ...completedInsufficientEvents({
          answerId: 'assistant-condition-drift',
          executionId: 'answer-execution-condition-drift',
          qcsId: 'qcs-condition-drift',
          question: '提交的条件必须冻结',
          conditions: [
            {
              condition_id: 'environment-staging',
              field: 'environment',
              operator: 'equals',
              value: 'staging'
            }
          ]
        }),
        'event: content\ndata: {"content":"已完成的回答。"}\n\n',
        'event: done\ndata: [DONE]\n\n'
      ].join('')
    });
  });
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  const composerConditions = page.locator('.composer').getByLabel('查询条件');
  await composerConditions.getByRole('button', { name: '添加查询条件' }).click();
  await composerConditions.getByLabel('条件 1 字段').fill('environment');
  await composerConditions.getByLabel('条件 1 运算符').fill('equals');
  await composerConditions.getByLabel('条件 1 值').fill('production');
  await sendQuestion(page, '提交的条件必须冻结');

  const answer = page.getByLabel('助手消息').last();
  await answer.getByRole('status').filter({ hasText: '回答失败，可重试' }).waitFor();
  assert.equal(await answer.getByRole('status').filter({ hasText: '证据不足' }).count(), 0);
  await answer.getByRole('button', { name: '重试' }).click();
  await page.getByRole('alert').filter({ hasText: 'recovered admitted execution' }).waitFor();
  assert.equal(requestPayloads.length, 1);
});

test('a missing structured insufficiency terminal event fails instead of inferring insufficiency', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.route('**/api/chat/stream', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: [
        ...completedInsufficientEvents({
          answerId: 'assistant-missing-insufficiency',
          executionId: 'answer-execution-missing-insufficiency',
          qcsId: 'qcs-missing-insufficiency',
          question: '缺少结构化拒答必须失败',
          includeInsufficientEvidenceReply: false
        }),
        'event: content\ndata: {"content":"已完成的回答。"}\n\n',
        'event: done\ndata: [DONE]\n\n'
      ].join('')
    });
  });
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await sendQuestion(page, '缺少结构化拒答必须失败');

  const answer = page.getByLabel('助手消息').last();
  await answer.getByRole('status').filter({ hasText: '回答失败，可重试' }).waitFor();
  assert.equal(await answer.getByRole('status').filter({ hasText: '证据不足' }).count(), 0);
});

test('Conversation Workspace renders a frozen insufficient answer through the authenticated SSE workflow', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await sendQuestion(page, '部署前需要做什么？');

  const answer = page.getByLabel('助手消息').last();
  await answer.getByRole('status').filter({ hasText: '证据不足' }).waitFor();
  assert.equal(
    await answer.getByText('未检索到足够相关的知识片段，请补充更具体的问题或关键词。').isVisible(),
    true
  );
  const summary = answer.getByLabel('证据摘要');
  assert.equal(await summary.getByText('证据不足').isVisible(), true);
  assert.equal(await summary.getByText('0 个来源').isVisible(), true);
  const completedBox = await answer.boundingBox();
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

test('a stream that closes before done remains a transport failure instead of a completed answer', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.route('**/api/chat/stream', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: 'event: content\ndata: {"content":"partial answer"}\n\n'
    });
  });
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await sendQuestion(page, '流在完成前中断时应该怎样显示？');

  const answer = page.getByLabel('助手消息').last();
  await answer.getByRole('status').filter({ hasText: '回答失败，可重试' }).waitFor();
  assert.equal(await answer.getByText('partial answer').isVisible(), true);
  assert.equal(await answer.getByRole('status').filter({ hasText: '已完成' }).count(), 0);
});

test('a post-admission stream interruption requires recovered frozen Query Conditions before retry', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  const requestPayloads = [];
  await page.route('**/api/chat/stream', async (route) => {
    requestPayloads.push(JSON.parse(route.request().postData() || '{}'));
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: 'event: stage\ndata: {"stage":"retrieval","message":"正在检索"}\n\n'
    });
  });
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await sendQuestion(page, 'admission 后中断必须恢复冻结条件');

  const answer = page.getByLabel('助手消息').last();
  await answer.getByRole('status').filter({ hasText: '回答失败，可重试' }).waitFor();
  await answer.getByRole('button', { name: '重试' }).click();
  await page.getByRole('alert').filter({ hasText: 'recovered admitted execution' }).waitFor();
  assert.equal(requestPayloads.length, 1);
});

test('an assistant identity before interruption requires recovered frozen Query Conditions before retry', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  const requestPayloads = [];
  await page.route('**/api/chat/stream', async (route) => {
    requestPayloads.push(JSON.parse(route.request().postData() || '{}'));
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: 'event: answer_identity\ndata: {"answer_id":"assistant-identity-only"}\n\n'
    });
  });
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await sendQuestion(page, 'identity 后中断必须恢复冻结条件');

  const answer = page.getByLabel('助手消息').last();
  await answer.getByRole('status').filter({ hasText: '回答失败，可重试' }).waitFor();
  await answer.getByRole('button', { name: '重试' }).click();
  await page.getByRole('alert').filter({ hasText: 'recovered admitted execution' }).waitFor();
  assert.equal(requestPayloads.length, 1);
});

test('a stream from a previously selected private conversation cannot rewrite reloaded history', { timeout: 30000 }, async (t) => {
  const delayedAnswer = '来自先前会话的回答不能写入当前历史。';
  const delayedTerminal = [
    ...completedInsufficientEvents({
      answerId: 'assistant-prior-session-stream',
      executionId: 'answer-execution-prior-session-stream',
      qcsId: 'qcs-prior-session-stream',
      question: '先前会话的回答',
      answerText: delayedAnswer
    }),
    `event: content\ndata: ${JSON.stringify({ content: delayedAnswer })}\n\n`,
    'event: done\ndata: [DONE]\n\n'
  ].join('');
  const initialStage = 'event: stage\ndata: {"stage":"retrieval","message":"正在检索"}\n\n';
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.addInitScript(
    ({ stage, terminal }) => {
      const originalFetch = window.fetch.bind(window);
      window.__releaseTicket20PriorSessionStream = () => {};
      window.fetch = async (input, init) => {
        const url = typeof input === 'string' ? input : input.url;
        if (!url.includes('/api/chat/stream')) {
          return originalFetch(input, init);
        }
        const encoder = new TextEncoder();
        const body = new ReadableStream({
          start(controller) {
            controller.enqueue(encoder.encode(stage));
            window.__releaseTicket20PriorSessionStream = () => {
              controller.enqueue(encoder.encode(terminal));
              controller.close();
            };
          }
        });
        return new Response(body, {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' }
        });
      };
    },
    { stage: initialStage, terminal: delayedTerminal }
  );
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await page.getByRole('button', { name: '新建会话' }).first().click();
  await sendQuestion(page, '先前会话的回答');
  await page.getByRole('status').filter({ hasText: '正在检索' }).waitFor();

  await page.getByRole('button', { name: /^session-evidence-history/ }).click();
  const firstHistoricalAnswer = page.getByLabel('助手消息').first();
  await firstHistoricalAnswer.getByText('历史回答有可核对来源。').waitFor();
  const frozenHistoryProjection = await firstHistoricalAnswer.innerText();
  await page.evaluate(() => window.__releaseTicket20PriorSessionStream());
  await page.waitForTimeout(500);

  assert.equal(await page.getByText(delayedAnswer, { exact: false }).count(), 0);
  assert.equal(await firstHistoricalAnswer.innerText(), frozenHistoryProjection);
});

for (const [terminalName, terminal] of [
  ['a truncated done frame', 'event: done\ndata: "[DO'],
  ['a done frame without the exact marker', 'event: done\ndata: {"state":"failed"}\n\n']
]) {
  test(`${terminalName} fails closed and clears completed retry authority`, { timeout: 30000 }, async (t) => {
    const { page, baseUrl } = await startAsKnowledgeUser(t, {});
    const requestPayloads = [];
    await page.route('**/api/chat/stream', async (route) => {
      requestPayloads.push(JSON.parse(route.request().postData() || '{}'));
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: [
          ...completedInsufficientEvents({
            answerId: `assistant-${terminalName}`,
            executionId: `answer-execution-${terminalName}`,
            qcsId: `qcs-${terminalName}`,
            question: '终态 marker 必须严格匹配'
          }),
          'event: content\ndata: {"content":"已完成的回答。"}\n\n',
          terminal
        ].join('')
      });
    });
    await page.goto(`${baseUrl}chat`);
    await page.getByRole('heading', { name: '对话工作区' }).waitFor();

    await sendQuestion(page, '终态 marker 必须严格匹配');

    const answer = page.getByLabel('助手消息').last();
    await answer.getByRole('status').filter({ hasText: '回答失败，可重试' }).waitFor();
    assert.equal(await answer.getByRole('status').filter({ hasText: '证据不足' }).count(), 0);
    assert.equal(await answer.getByLabel('证据摘要').count(), 0);
    await answer.getByRole('button', { name: '重试' }).click();
    await page.getByRole('alert').filter({ hasText: 'recovered admitted execution' }).waitFor();
    assert.equal(requestPayloads.length, 1);
  });
}

test('ineligible retrieval candidates remain insufficient even when the configured provider would fail', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {
    env: { BROWSER_ACCEPTANCE_FAIL_FIRST: '1' }
  });
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await sendQuestion(page, '部署前需要做什么？');
  const firstAnswer = page.getByLabel('助手消息').last();
  await firstAnswer.getByRole('status').filter({ hasText: '证据不足' }).waitFor();

  await sendQuestion(page, '部署前需要做什么？');
  const secondAnswer = page.getByLabel('助手消息').last();
  await secondAnswer.getByRole('status').filter({ hasText: '证据不足' }).waitFor();
  assert.equal(await firstAnswer.getByLabel('证据摘要').getByText('0 个来源').isVisible(), true);
  assert.equal(await secondAnswer.getByLabel('证据摘要').getByText('0 个来源').isVisible(), true);
  assert.equal(await page.getByText('【生成不可用】生成服务暂不可用，请稍后重试。').count(), 0);
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

test('a completed frozen insufficiency remains readable without overflow at a 390-pixel viewport', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, { viewport: { width: 390, height: 844 } });
  await page.goto(`${baseUrl}chat`);
  await sendQuestion(page, '部署前需要做什么？');
  const answer = page.getByLabel('助手消息').last();
  await answer.getByRole('status').filter({ hasText: '证据不足' }).waitFor();
  assert.equal(
    await answer.getByText('未检索到足够相关的知识片段，请补充更具体的问题或关键词。').isVisible(),
    true
  );
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
});

test('a Knowledge User sees a frozen insufficient Evidence Summary without fabricated excerpts or diagnostics', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, { viewport: { width: 1440, height: 900 } });
  await page.goto(`${baseUrl}chat`);
  await sendQuestion(page, '部署前需要做什么？');

  const answer = page.getByLabel('助手消息').last();
  const summary = answer.getByLabel('证据摘要');
  await summary.waitFor();
  assert.equal(await summary.getByText('证据不足').isVisible(), true);
  assert.equal(await summary.getByText('0 个来源').isVisible(), true);
  assert.equal(await page.getByText('RAG Trace').count(), 0);
  assert.equal(await page.getByLabel('检索诊断').count(), 0);
  assert.equal(await summary.getByRole('button', { name: /查看来源/ }).count(), 0);
  assert.equal(await answer.getByLabel('知识反馈').count(), 0);
});

test('an ineligible retrieval hit does not turn into a Public Source Citation', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, { viewport: { width: 1440, height: 900 } });
  await page.goto(`${baseUrl}chat`);
  await sendQuestion(page, '什么时候使用 deterministic workflow？');

  const answer = page.getByLabel('助手消息').last();
  await answer.getByRole('status').filter({ hasText: '证据不足' }).waitFor();
  const summary = answer.getByLabel('证据摘要');
  assert.equal(await summary.getByText('证据不足').isVisible(), true);
  assert.equal(await summary.getByText('0 个来源').isVisible(), true);
  assert.equal(await summary.getByRole('button', { name: /查看来源/ }).count(), 0);
  assert.equal(await page.getByRole('complementary', { name: '来源摘录' }).count(), 0);
  assert.equal(await answer.getByText('Anthropic', { exact: true }).count(), 0);
  assert.equal(await answer.getByText(/internal-chunk|internal-document/).count(), 0);

  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
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
  assert.equal(await diagnostics.getByText('门禁拒绝').isVisible(), true);
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
