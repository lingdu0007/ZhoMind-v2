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

const expectClosedResultUnavailable = async (answer) => {
  const unavailable = answer.locator('.answer-outcome--contract-failure');
  await unavailable.waitFor();
  assert.equal(
    await unavailable.getByText('Closed result unavailable', { exact: true }).isVisible(),
    true
  );
  assert.equal(await answer.getByText('Supported by published knowledge', { exact: true }).count(), 0);
  assert.equal(await answer.getByText('Insufficient Evidence Reply', { exact: true }).count(), 0);
  assert.equal(await answer.getByLabel('证据摘要').count(), 0);
  assert.equal(await answer.getByRole('button', { name: /打开引用/ }).count(), 0);
  assert.equal(await answer.getByRole('button', { name: '重试' }).count(), 0);
};

const expectInsufficientEvidenceReply = async (answer) => {
  const insufficient = answer.getByLabel('证据不足回复');
  await insufficient.waitFor();
  assert.equal(
    await insufficient.getByText('Insufficient Evidence Reply', { exact: true }).isVisible(),
    true
  );
  assert.equal(await answer.getByLabel('证据摘要').count(), 0);
  assert.equal(await answer.getByRole('button', { name: /打开引用/ }).count(), 0);
  return insufficient;
};

const completedInsufficientEvents = ({
  answerId,
  executionId,
  qcsId,
  question,
  conditions = [],
  answerText = '已完成的回答。',
  reason = 'decision_not_covered',
  includeInsufficientEvidenceReply = true
}) => {
  const evidenceSummary = { coverage: 'insufficient', source_count: 0, sources: [] };
  const insufficientEvidenceReply = {
    outcome: 'insufficient_evidence_reply',
    reason,
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

const completedSupportedEvents = ({
  answerId,
  executionId,
  qcsId,
  question,
  sourceUrl = 'https://www.anthropic.com/engineering/building-effective-agents',
  sourceAccessScope = '',
  includeConditionAndVersionScope = false,
  answerText: suppliedAnswerText
}) => {
  const snapshotId = 'b'.repeat(64);
  const evidenceSetIdentity = 'c'.repeat(64);
  const source = {
    citation_id: 'S1',
    citation_identity: 'a'.repeat(64),
    entry_id: 'synthetic-workflow-001',
    entry_title: 'Prefer deterministic workflows',
    section_id: 'recommendation_or_reviewed_branches',
    source_title: 'Building effective agents',
    source_authority: 'Anthropic',
    source_url: sourceUrl,
    source_version: '2024-12-19',
    review_date: '2026-08-12',
    review_status: 'approved',
    publication_version: 'v1',
    assurance_level: 'claim_linked',
    applicability_conditions: [
      { field: 'execution_path', operator: 'equals', value: 'known' }
    ],
    non_applicability_conditions: [
      { field: 'autonomy', operator: 'equals', value: 'unbounded' }
    ],
    snapshot_id: snapshotId,
    excerpt: 'Known execution paths should use deterministic workflows.'
  };
  if (sourceAccessScope) source.source_access_scope = sourceAccessScope;
  const evidenceSummary = {
    coverage: 'sufficient',
    source_count: 1,
    sources: [source]
  };
  const answerLines = [
    '【Supported by published knowledge】',
    '',
    '## Recommendation',
    'Use deterministic workflows for known execution paths. [S1]',
    '',
    '## Applicability Limits',
    'Use this only when termination conditions are explicit. [S1]',
    '',
    '## Alternatives',
    'Use bounded agent autonomy only for dynamic paths. [S1]',
    '',
    '## Minimal Implementation or Acceptance Check',
    'Verify the workflow terminates on an explicit condition. [S1]'
  ];
  if (includeConditionAndVersionScope) {
    answerLines.push(
      '',
      '## Missing Conditions and Version Scope',
      'Confirm the deployment version before applying this workflow. [S1]'
    );
  }
  const answerText = suppliedAnswerText || answerLines.join('\n');
  const execution = {
    id: executionId,
    assistant_message_id: answerId,
    state: 'completed',
    question,
    answer_text: answerText,
    query_condition_set: {
      identity: qcsId,
      normalized_question: question,
      conditions: []
    },
    condition_provenance: { mode: 'question_normalized' },
    outcome: 'evidence_gated_answer',
    evidence_set_identity: evidenceSetIdentity,
    item_identities: ['d'.repeat(64)],
    snapshot_ids: [snapshotId],
    knowledge_version_identities: ['e'.repeat(64)],
    evidence_summary: evidenceSummary
  };
  return [
    `event: answer_identity\ndata: ${JSON.stringify({ answer_id: answerId })}\n\n`,
    `event: answer_execution\ndata: ${JSON.stringify({ answer_execution: execution })}\n\n`,
    'event: outcome\ndata: {"outcome":"evidence_gated_answer"}\n\n',
    `event: content\ndata: ${JSON.stringify({ content: answerText })}\n\n`,
    `event: evidence_summary\ndata: ${JSON.stringify({ evidence_summary: evidenceSummary })}\n\n`,
    'event: done\ndata: [DONE]\n\n'
  ];
};

const completedGenerationUnavailableEvents = ({ answerId, executionId, qcsId, question }) => {
  const answerText = '【Generation Unavailable】\nThis frozen provider response must not be rendered.';
  const evidenceSummary = { coverage: 'unavailable', source_count: 0, sources: [] };
  const execution = {
    id: executionId,
    assistant_message_id: answerId,
    state: 'completed',
    question,
    answer_text: answerText,
    query_condition_set: {
      identity: qcsId,
      normalized_question: question,
      conditions: []
    },
    condition_provenance: { mode: 'question_normalized' },
    outcome: 'generation_unavailable',
    evidence_set_identity: 'f'.repeat(64),
    item_identities: ['1'.repeat(64)],
    snapshot_ids: ['2'.repeat(64)],
    knowledge_version_identities: ['3'.repeat(64)],
    evidence_summary: evidenceSummary
  };
  return [
    `event: answer_identity\ndata: ${JSON.stringify({ answer_id: answerId })}\n\n`,
    `event: answer_execution\ndata: ${JSON.stringify({ answer_execution: execution })}\n\n`,
    'event: outcome\ndata: {"outcome":"generation_unavailable"}\n\n',
    `event: content\ndata: ${JSON.stringify({ content: answerText })}\n\n`,
    `event: evidence_summary\ndata: ${JSON.stringify({ evidence_summary: evidenceSummary })}\n\n`,
    'event: done\ndata: [DONE]\n\n'
  ];
};

const terminalExecutionEvents = ({
  answerId,
  executionId,
  qcsId,
  question,
  state,
  failureCode
}) => {
  const execution = {
    id: executionId,
    assistant_message_id: answerId,
    state,
    question,
    query_condition_set: {
      identity: qcsId,
      normalized_question: question,
      conditions: []
    },
    condition_provenance: { mode: 'question_normalized' },
    failure_code: failureCode
  };
  return [
    `event: answer_identity\ndata: ${JSON.stringify({ answer_id: answerId })}\n\n`,
    `event: answer_execution\ndata: ${JSON.stringify({ answer_execution: execution })}\n\n`,
    `event: error\ndata: ${JSON.stringify({ code: failureCode, message: `${state} terminal` })}\n\n`,
    'event: done\ndata: [DONE]\n\n'
  ];
};

const assistantPersistenceFailureEvents = ({ answerId, executionId, qcsId, question }) => {
  const execution = {
    id: executionId,
    state: 'failed',
    question,
    query_condition_set: {
      identity: qcsId,
      normalized_question: question,
      conditions: []
    },
    condition_provenance: { mode: 'question_normalized' },
    failure_code: 'ANSWER_EXECUTION_PERSISTENCE_FAILED'
  };
  return [
    `event: answer_identity\ndata: ${JSON.stringify({ answer_id: answerId })}\n\n`,
    `event: answer_execution\ndata: ${JSON.stringify({ answer_execution: execution })}\n\n`,
    'event: error\ndata: {"code":"ANSWER_EXECUTION_PERSISTENCE_FAILED","message":"assistant persistence failed"}\n\n',
    'event: done\ndata: [DONE]\n\n'
  ];
};

const completedSupportedHistory = ({ withdrawn = false } = {}) => {
  const answerId = 'assistant-history-supported';
  const question = '历史的 deterministic workflow 问题';
  const snapshotId = '9'.repeat(64);
  const source = {
    citation_id: 'S1',
    citation_identity: '8'.repeat(64),
    entry_id: 'synthetic-workflow-001',
    entry_title: 'Prefer deterministic workflows',
    section_id: 'recommendation_or_reviewed_branches',
    source_title: 'Building effective agents',
    source_authority: 'Anthropic',
    source_url: 'https://www.anthropic.com/engineering/building-effective-agents',
    source_version: '2024-12-19',
    review_date: '2026-08-12',
    publication_version: 'v1',
    snapshot_id: snapshotId
  };
  if (withdrawn) source.withdrawal_notice = 'This source has been withdrawn.';
  else source.excerpt = 'Known execution paths should use deterministic workflows.';
  const evidenceSummary = { coverage: 'sufficient', source_count: 1, sources: [source] };
  const answerText = [
    '【Supported by published knowledge】',
    '',
    '## Recommendation',
    'Use deterministic workflows for known paths. [S1]',
    '',
    '## Applicability Limits',
    'Require explicit termination conditions. [S1]',
    '',
    '## Alternatives',
    'Use bounded autonomy for dynamic paths. [S1]',
    '',
    '## Minimal Implementation or Acceptance Check',
    'Verify an explicit termination condition. [S1]'
  ].join('\n');
  const execution = {
    id: 'answer-execution-history-supported',
    assistant_message_id: answerId,
    state: 'completed',
    question,
    answer_text: answerText,
    query_condition_set: {
      identity: '7'.repeat(64),
      normalized_question: question,
      conditions: []
    },
    condition_provenance: { mode: 'question_normalized' },
    outcome: 'evidence_gated_answer',
    evidence_set_identity: '6'.repeat(64),
    item_identities: ['5'.repeat(64)],
    snapshot_ids: [snapshotId],
    knowledge_version_identities: ['4'.repeat(64)],
    evidence_summary: evidenceSummary
  };
  return {
    user: {
      id: 'user-history-supported',
      type: 'user',
      content: question,
      answer_execution: execution
    },
    assistant: {
      id: answerId,
      type: 'assistant',
      content: answerText,
      answer_execution: execution,
      outcome: 'evidence_gated_answer',
      evidence_summary: evidenceSummary
    }
  };
};

const completedInsufficientHistory = ({
  question,
  answerId = 'assistant-history-insufficient',
  answerText = '已完成的历史不足回答。',
  qcsId = 'qcs-history-insufficient',
  executionId = 'answer-execution-history-insufficient',
  userId = 'user-history-insufficient'
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
      conditions: []
    },
    condition_provenance: { mode: 'question_normalized' },
    outcome: 'insufficient_evidence_reply',
    evidence_set_identity: null,
    item_identities: [],
    snapshot_ids: [],
    knowledge_version_identities: [],
    insufficient_evidence_reply: insufficientEvidenceReply,
    evidence_summary: evidenceSummary
  };
  return {
    user: {
      id: userId,
      type: 'user',
      content: question,
      answer_execution: execution
    },
    assistant: {
      id: answerId,
      type: 'assistant',
      content: answerText,
      answer_execution: execution,
      outcome: 'insufficient_evidence_reply',
      insufficient_evidence_reply: insufficientEvidenceReply,
      evidence_summary: evidenceSummary
    }
  };
};

const persistenceFailureHistory = () => {
  const question = '历史持久化失败的部署问题';
  return {
    user: {
      id: 'user-history-persistence-failure',
      type: 'user',
      content: question,
      answer_execution: {
        id: 'answer-execution-history-persistence-failure',
        state: 'failed',
        question,
        query_condition_set: {
          identity: 'qcs-history-persistence-failure',
          normalized_question: question,
          conditions: []
        },
        condition_provenance: { mode: 'question_normalized' },
        failure_code: 'ANSWER_EXECUTION_PERSISTENCE_FAILED'
      }
    }
  };
};

test('a contradictory stored user question invalidates its paired completed assistant history', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  const history = completedInsufficientHistory({
    question: '历史中的原始知识问题'
  });
  history.user.content = '被篡改的历史用户问题';

  await page.route(/\/api\/sessions(?:\?.*)?$/, async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        data: {
          sessions: [
            {
              session_id: 'session-contradictory-history',
              updated_at: '2026-09-07T08:00:00Z',
              message_count: 2
            }
          ]
        }
      })
    });
  });
  await page.route(/\/api\/sessions\/session-contradictory-history(?:\?.*)?$/, async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        data: {
          session_id: 'session-contradictory-history',
          messages: [history.user, history.assistant]
        }
      })
    });
  });
  await page.goto(`${baseUrl}chat`);

  const rail = page.getByRole('complementary', { name: '最近会话' });
  await rail.getByRole('button', { name: /^session-contradictory-history/ }).click();
  const answer = page.getByLabel('助手消息').last();
  await expectClosedResultUnavailable(answer);
  assert.equal(await answer.getByText('已完成的历史不足回答。', { exact: true }).count(), 0);
  assert.equal(
    await page.getByRole('checkbox', { name: '沿用上一轮条件' }).isDisabled(),
    true
  );
});

test('a missing user-side assistant binding invalidates the paired completed assistant history', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  const history = completedInsufficientHistory({
    question: '绑定缺失的历史知识问题'
  });
  history.user.answer_execution = {
    ...history.user.answer_execution,
    assistant_message_id: null
  };

  await page.route(/\/api\/sessions(?:\?.*)?$/, async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        data: {
          sessions: [
            {
              session_id: 'session-missing-user-binding',
              updated_at: '2026-09-07T08:00:00Z',
              message_count: 2
            }
          ]
        }
      })
    });
  });
  await page.route(/\/api\/sessions\/session-missing-user-binding(?:\?.*)?$/, async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        data: {
          session_id: 'session-missing-user-binding',
          messages: [history.user, history.assistant]
        }
      })
    });
  });
  await page.goto(`${baseUrl}chat`);

  const rail = page.getByRole('complementary', { name: '最近会话' });
  await rail.getByRole('button', { name: /^session-missing-user-binding/ }).click();
  const answer = page.getByLabel('助手消息').last();
  await expectClosedResultUnavailable(answer);
  assert.equal(await answer.getByLabel('知识缺口报告').count(), 0);
  assert.equal(
    await page.getByRole('checkbox', { name: '沿用上一轮条件' }).isDisabled(),
    true
  );
});

test('switching private histories does not retain a prior answer knowledge-gap draft', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  const firstHistory = completedInsufficientHistory({
    question: '第一个历史缺口问题',
    answerId: 'assistant-feedback-first',
    answerText: '第一个历史不足回答。',
    qcsId: 'qcs-feedback-first',
    executionId: 'answer-execution-feedback-first',
    userId: 'user-feedback-first'
  });
  const secondHistory = completedInsufficientHistory({
    question: '第二个历史缺口问题',
    answerId: 'assistant-feedback-second',
    answerText: '第二个历史不足回答。',
    qcsId: 'qcs-feedback-second',
    executionId: 'answer-execution-feedback-second',
    userId: 'user-feedback-second'
  });
  const feedbackPayloads = [];
  const feedbackItems = [];

  await page.route(/\/api\/sessions(?:\?.*)?$/, async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        data: {
          sessions: [
            {
              session_id: 'session-feedback-first',
              updated_at: '2026-09-07T08:00:00Z',
              message_count: 2
            },
            {
              session_id: 'session-feedback-second',
              updated_at: '2026-09-07T08:01:00Z',
              message_count: 2
            }
          ]
        }
      })
    });
  });
  await page.route(/\/api\/sessions\/session-feedback-first(?:\?.*)?$/, async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        data: {
          session_id: 'session-feedback-first',
          messages: [firstHistory.user, firstHistory.assistant]
        }
      })
    });
  });
  await page.route(/\/api\/sessions\/session-feedback-second(?:\?.*)?$/, async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        data: {
          session_id: 'session-feedback-second',
          messages: [secondHistory.user, secondHistory.assistant]
        }
      })
    });
  });
  await page.route('**/api/knowledge-feedback**', async (route) => {
    if (route.request().method() === 'GET') {
      const answerId = new URL(route.request().url()).searchParams.get('answer_id');
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({
          data: {
            items: feedbackItems.filter((item) => item.answer_id === answerId)
          }
        })
      });
      return;
    }
    const payload = JSON.parse(route.request().postData() || '{}');
    feedbackPayloads.push(payload);
    const feedback = {
      id: `feedback-${feedbackItems.length + 1}`,
      answer_id: payload.answer_id,
      entry_id: payload.entry_id || null,
      knowledge_edition: null,
      label: payload.label,
      outcome: 'insufficient_evidence_reply',
      created_at: '2026-09-07T08:10:00Z',
      expires_at: '2027-03-06T08:10:00Z',
      retention_days: 180,
      duplicate: false
    };
    feedbackItems.push(feedback);
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        data: feedback
      })
    });
  });
  await page.goto(`${baseUrl}chat`);

  const rail = page.getByRole('complementary', { name: '最近会话' });
  await rail.getByRole('button', { name: /^session-feedback-first/ }).click();
  const firstGapReport = page.getByLabel('助手消息').last().getByLabel('知识缺口报告');
  await firstGapReport.getByRole('button', { name: '报告知识缺口' }).click();
  await firstGapReport.getByLabel('补充说明（可选）').fill('第一条历史草稿不能泄露到下一条回答。');
  await firstGapReport.getByRole('button', { name: '预览缺口报告' }).click();
  await firstGapReport.getByLabel('缺口报告预览').waitFor();

  await rail.getByRole('button', { name: /^session-feedback-second/ }).click();
  const secondGapReport = page.getByLabel('助手消息').last().getByLabel('知识缺口报告');
  await secondGapReport.getByRole('button', { name: '报告知识缺口' }).waitFor();
  assert.equal(await secondGapReport.getByLabel('缺口报告预览').count(), 0);
  assert.equal(
    await secondGapReport.getByText('第一条历史草稿不能泄露到下一条回答。', { exact: true }).count(),
    0
  );

  await secondGapReport.getByRole('button', { name: '报告知识缺口' }).click();
  await secondGapReport.getByLabel('补充说明（可选）').fill('第二条历史回答的独立缺口。');
  await secondGapReport.getByRole('button', { name: '预览缺口报告' }).click();
  const preview = secondGapReport.getByLabel('缺口报告预览');
  await preview.waitFor();
  await preview.getByRole('button', { name: '确认提交' }).click();
  await secondGapReport.getByText('缺口报告已提交', { exact: true }).waitFor();
  assert.equal(
    await secondGapReport.getByText('第二条历史回答的独立缺口。', { exact: true }).isVisible(),
    true
  );

  assert.deepEqual(feedbackPayloads, [
    {
      answer_id: 'assistant-feedback-second',
      label: 'insufficient_evidence',
      note: '第二条历史回答的独立缺口。'
    }
  ]);
});

test('a Knowledge User previews, reloads, and deletes only their retained evidence feedback signal', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  const history = completedSupportedHistory();
  const answerId = history.assistant.id;
  let feedbackItems = [
    {
      id: 'feedback-retained-history',
      answer_id: answerId,
      entry_id: 'synthetic-workflow-001',
      knowledge_edition: 'publication:v1',
      label: 'helpful',
      created_at: '2026-09-07T08:00:00Z',
      expires_at: '2027-03-06T08:00:00Z',
      retention_days: 180,
      duplicate: false
    }
  ];

  await page.route(/\/api\/sessions(?:\?.*)?$/, async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        data: {
          sessions: [
            {
              session_id: 'session-retained-evidence-feedback',
              updated_at: '2026-09-07T08:00:00Z',
              message_count: 2
            }
          ]
        }
      })
    });
  });
  await page.route(/\/api\/sessions\/session-retained-evidence-feedback(?:\?.*)?$/, async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        data: {
          session_id: 'session-retained-evidence-feedback',
          messages: [history.user, history.assistant]
        }
      })
    });
  });
  await page.route('**/api/knowledge-feedback**', async (route) => {
    const request = route.request();
    if (request.method() === 'GET') {
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({ data: { items: feedbackItems } })
      });
      return;
    }
    if (request.method() === 'DELETE') {
      const signalId = request.url().split('/').at(-1);
      feedbackItems = feedbackItems.filter((item) => item.id !== signalId);
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({ data: { id: signalId, deleted: true } })
      });
      return;
    }
    const payload = JSON.parse(request.postData() || '{}');
    const submitted = {
      id: 'feedback-new-history',
      answer_id: payload.answer_id,
      entry_id: payload.entry_id,
      knowledge_edition: 'publication:v1',
      label: payload.label,
      created_at: '2026-09-07T08:10:00Z',
      expires_at: '2027-03-06T08:10:00Z',
      retention_days: 180,
      duplicate: false
    };
    feedbackItems = [submitted];
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ data: submitted })
    });
  });

  await page.goto(`${baseUrl}chat`);
  const rail = page.getByRole('complementary', { name: '最近会话' });
  await rail.getByRole('button', { name: /^session-retained-evidence-feedback/ }).click();
  const feedback = page.getByLabel('助手消息').last().getByLabel('知识反馈');
  await feedback.getByText('反馈已提交', { exact: true }).waitFor();
  await feedback.getByRole('button', { name: '删除反馈' }).click();
  await feedback.getByRole('button', { name: '预览反馈' }).waitFor();

  await feedback.getByRole('radio', { name: '有帮助' }).click();
  await feedback.getByText('反馈已提交', { exact: true }).waitFor();
  assert.equal(await feedback.getByText('有帮助', { exact: true }).isVisible(), true);
  await feedback.getByRole('button', { name: '删除反馈' }).click();
  await feedback.getByRole('button', { name: '预览反馈' }).waitFor();
  await feedback.getByLabel('补充说明（可选）').fill('这条有帮助反馈需要保留说明。');
  await feedback.getByRole('radio', { name: '有帮助' }).click();
  const helpfulPreview = feedback.getByLabel('反馈预览');
  await helpfulPreview.waitFor();
  assert.equal(
    await helpfulPreview.getByText('这条有帮助反馈需要保留说明。', { exact: true }).isVisible(),
    true
  );
  await helpfulPreview.getByRole('button', { name: '确认提交' }).click();
  await feedback.getByText('反馈已提交', { exact: true }).waitFor();

  await page.reload();
  await rail.getByRole('button', { name: /^session-retained-evidence-feedback/ }).click();
  const reloadedFeedback = page.getByLabel('助手消息').last().getByLabel('知识反馈');
  await reloadedFeedback.getByText('反馈已提交', { exact: true }).waitFor();
  await reloadedFeedback.getByRole('button', { name: '删除反馈' }).click();
  await reloadedFeedback.getByRole('button', { name: '预览反馈' }).waitFor();
});

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
  await rail.getByText('历史的部署问题').waitFor();
  assert.equal(await rail.getByText('历史的部署问题').isVisible(), true);
  assert.equal(await rail.getByText('未记录关闭结果').isVisible(), true);
  assert.equal(await rail.getByText('4 条消息').isVisible(), true);
  assert.equal(await rail.getByRole('button', { name: /^历史的部署问题/ }).getAttribute('aria-current'), null);
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

test('Conversation Workspace gives stored throttled and rejected sessions explicit terminal labels', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.route(/\/api\/sessions(?:\?.*)?$/, async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        data: {
          sessions: [
            {
              session_id: 'session-throttled',
              title: '受限流的知识问题',
              latest_execution_state: 'throttled',
              updated_at: '2026-09-07T08:00:00Z',
              message_count: 2
            },
            {
              session_id: 'session-rejected',
              title: '被拒绝的知识问题',
              latest_execution_state: 'rejected',
              updated_at: '2026-09-07T08:01:00Z',
              message_count: 2
            }
          ]
        }
      })
    });
  });
  await page.goto(`${baseUrl}chat`);

  const rail = page.getByRole('complementary', { name: '最近会话' });
  await rail.getByText('受限流的知识问题', { exact: true }).waitFor();
  assert.equal(await rail.getByText('已限流', { exact: true }).isVisible(), true);
  assert.equal(await rail.getByText('被拒绝的知识问题', { exact: true }).isVisible(), true);
  assert.equal(await rail.getByText('已拒绝', { exact: true }).isVisible(), true);
  assert.equal(await rail.getByText('未记录关闭结果', { exact: true }).count(), 0);
});

test('Conversation Workspace rejects an empty question with an explicit composer state', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await page.getByRole('button', { name: '发送' }).click();
  assert.equal(await page.getByRole('alert').innerText(), '请输入问题后再发送。');
});

test('the empty Conversation Workspace loads published coverage and pre-fills a representative decision query', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.goto(`${baseUrl}chat`);

  const coverage = page.getByLabel('已发布知识覆盖');
  await coverage.waitFor();
  await coverage.getByText('Agent Orchestration').waitFor();
  assert.equal(await coverage.getByText('Prefer deterministic workflows').isVisible(), true);
  assert.equal(await coverage.getByText('已知路径应由 deterministic workflow 控制。').isVisible(), true);
  assert.equal(await coverage.getByText('v1', { exact: true }).isVisible(), true);
  assert.equal(await coverage.getByText('2026-08-12', { exact: true }).isVisible(), true);
  assert.equal(await coverage.getByText('1 个来源', { exact: true }).isVisible(), true);
  assert.equal(await coverage.getByText(/candidate private|internal-/).count(), 0);

  await coverage.getByRole('button', { name: '什么时候使用 deterministic workflow？' }).click();
  assert.equal(
    await page.getByPlaceholder('请输入需要检索的问题').inputValue(),
    '什么时候使用 deterministic workflow？'
  );
});

test('the empty Conversation Workspace makes coverage-load failure explicit without fabricated topics', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.route(/\/api(?:\/v1)?\/knowledge-map(?:\?.*)?$/, async (route) => {
    await route.fulfill({
      status: 503,
      contentType: 'application/json',
      body: JSON.stringify({ code: 'KNOWLEDGE_MAP_UNAVAILABLE', message: 'unavailable' })
    });
  });
  await page.goto(`${baseUrl}chat`);

  await page.getByRole('alert').filter({ hasText: '无法加载已发布知识覆盖。' }).waitFor();
  assert.equal(await page.getByLabel('已发布知识覆盖').count(), 0);
  assert.equal(await page.getByText('Prefer deterministic workflows').count(), 0);
});

test('a supported answer renders the frozen decision sections and opens its governing citation path', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.route('**/api/chat/stream', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: completedSupportedEvents({
        answerId: 'assistant-supported',
        executionId: 'answer-execution-supported',
        qcsId: 'qcs-supported',
        question: '什么时候使用 deterministic workflow？'
      }).join('')
    });
  });
  await page.goto(`${baseUrl}chat`);

  await sendQuestion(page, '什么时候使用 deterministic workflow？');

  const answer = page.getByLabel('助手消息').last();
  const supportedAnswer = answer.getByLabel('已支持的知识回答');
  await supportedAnswer.getByText('Supported by published knowledge', { exact: true }).waitFor();
  assert.equal(await answer.getByText('Decision Summary', { exact: true }).isVisible(), true);
  assert.equal(await answer.getByText('Applicability Limits', { exact: true }).isVisible(), true);
  assert.equal(await answer.getByText('Alternatives', { exact: true }).isVisible(), true);
  assert.equal(
    await answer.getByText('Minimal Implementation or Acceptance Check', { exact: true }).isVisible(),
    true
  );

  await answer.getByRole('button', { name: '打开引用 S1' }).first().click();
  const entry = page.getByRole('complementary', { name: '来源摘录' });
  await entry.getByText('Governing Engineering Decision Entry', { exact: true }).waitFor();
  assert.equal(await entry.getByText('recommendation_or_reviewed_branches', { exact: true }).isVisible(), true);
  assert.match(await entry.getByText(/^[a-f0-9]{64}$/).first().innerText(), /^[a-f0-9]{64}$/);
  assert.equal(await entry.getByText('审查状态', { exact: true }).isVisible(), true);
  assert.equal(await entry.getByText('approved', { exact: true }).isVisible(), true);
  assert.equal(await entry.getByText('保证级别', { exact: true }).isVisible(), true);
  assert.equal(await entry.getByText('claim_linked', { exact: true }).isVisible(), true);
  assert.equal(await entry.getByText('适用条件', { exact: true }).isVisible(), true);
  assert.equal(await entry.getByText('execution_path equals known', { exact: true }).isVisible(), true);
  assert.equal(await entry.getByText('不适用条件', { exact: true }).isVisible(), true);
  assert.equal(await entry.getByText('autonomy equals unbounded', { exact: true }).isVisible(), true);
  assert.equal(await entry.getByRole('link', { name: '打开公开来源' }).isVisible(), true);
});

test('a supported answer preserves the frozen implementation label and unknown section heading', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  const question = '实施辅助必须保留冻结标签';
  const answerText = [
    '【Supported by published knowledge】',
    'Evidence-Bounded Implementation Aid',
    '',
    'This frozen implementation scope remains evidence-bounded. [S1]',
    '',
    '## Recommendation',
    'Use deterministic workflows for known execution paths. [S1]',
    '',
    '## Applicability Limits',
    'Use this only when termination conditions are explicit. [S1]',
    '',
    '## Alternatives',
    'Use bounded agent autonomy only for dynamic paths. [S1]',
    '',
    '## Minimal Implementation or Acceptance Check',
    'Verify the workflow terminates on an explicit condition. [S1]',
    '',
    '## Operational Notes',
    'Keep the approved deployment window unchanged. [S1]'
  ].join('\n');
  await page.route('**/api/chat/stream', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: completedSupportedEvents({
        answerId: 'assistant-supported-implementation-aid',
        executionId: 'answer-execution-supported-implementation-aid',
        qcsId: 'qcs-supported-implementation-aid',
        question,
        answerText
      }).join('')
    });
  });
  await page.goto(`${baseUrl}chat`);

  await sendQuestion(page, question);

  const answer = page.getByLabel('助手消息').last();
  await answer.getByText('Evidence-Bounded Implementation Aid', { exact: true }).waitFor();
  assert.equal(
    await answer.getByText('This frozen implementation scope remains evidence-bounded.', { exact: true }).isVisible(),
    true
  );
  assert.equal(await answer.getByText('Operational Notes', { exact: true }).isVisible(), true);
  assert.equal(await answer.getByText('Keep the approved deployment window unchanged.', { exact: true }).isVisible(), true);
  assert.equal(await answer.getByRole('button', { name: '打开引用 S1' }).count() > 0, true);
  assert.equal(await answer.getByText('【Supported by published knowledge】', { exact: true }).count(), 0);
});

test('a supported answer accepts a controlled locator and keeps it inside the governing entry panel', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.route('**/api/chat/stream', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: completedSupportedEvents({
        answerId: 'assistant-supported-controlled',
        executionId: 'answer-execution-supported-controlled',
        qcsId: 'qcs-supported-controlled',
        question: '受控来源如何展示？',
        sourceUrl: 'controlled://knowledge/reviewed-decision-001',
        sourceAccessScope: 'controlled_internal',
        includeConditionAndVersionScope: true
      }).join('')
    });
  });
  await page.goto(`${baseUrl}chat`);

  await sendQuestion(page, '受控来源如何展示？');

  const answer = page.getByLabel('助手消息').last();
  await answer.getByText('Missing Conditions and Version Scope', { exact: true }).waitFor();
  await answer.getByRole('button', { name: '打开引用 S1' }).first().click();
  const entry = page.getByRole('complementary', { name: '来源摘录' });
  await entry.getByText('受控来源定位符', { exact: true }).waitFor();
  assert.equal(
    await entry.getByText('controlled://knowledge/reviewed-decision-001', { exact: true }).isVisible(),
    true
  );
  assert.equal(await entry.getByRole('link', { name: '打开公开来源' }).count(), 0);
});

test('every frozen insufficiency reason remains non-supporting and provides refine and private gap actions', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  const reasonTitles = {
    no_eligible_published_evidence: 'No eligible published evidence',
    decision_not_covered: 'The requested decision is not covered',
    decisive_condition_missing: 'A decisive condition is missing',
    material_evidence_conflict: 'Published evidence has a material conflict',
    assurance_support_missing: 'Required assurance support is missing',
    evidence_budget_exceeded: 'Required evidence exceeds the bounded evidence set',
    knowledge_needs_review: 'Published knowledge needs review'
  };
  const reasons = Object.keys(reasonTitles);
  let turn = 0;
  await page.route('**/api/chat/stream', async (route) => {
    const request = JSON.parse(route.request().postData() || '{}');
    const reason = reasons[turn];
    const question = request.message;
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: [
        ...completedInsufficientEvents({
          answerId: `assistant-insufficient-${turn}`,
          executionId: `answer-execution-insufficient-${turn}`,
          qcsId: `qcs-insufficient-${turn}`,
          question,
          reason,
          answerText: `该闭合不足结果属于 ${reason}。`
        }),
        `event: content\ndata: ${JSON.stringify({ content: `该闭合不足结果属于 ${reason}。` })}\n\n`,
        'event: done\ndata: [DONE]\n\n'
      ].join('')
    });
    turn += 1;
  });
  await page.goto(`${baseUrl}chat`);
  await page.getByLabel('已发布知识覆盖').waitFor();

  for (const [index, reason] of reasons.entries()) {
    const question = `不足原因 ${index + 1}`;
    await sendQuestion(page, question);
    const answer = page.getByLabel('助手消息').last();
    const insufficient = answer.getByLabel('证据不足回复');
    await insufficient.waitFor();
    assert.equal(await insufficient.getByText('Insufficient Evidence Reply', { exact: true }).isVisible(), true);
    assert.equal(await insufficient.getByText(reasonTitles[reason], { exact: true }).isVisible(), true);
    assert.equal(await answer.getByLabel('证据摘要').count(), 0);
    assert.equal(await answer.getByRole('button', { name: /打开引用/ }).count(), 0);
    assert.equal(await answer.getByLabel('非支持性已发布覆盖').isVisible(), true);
  }

  const finalAnswer = page.getByLabel('助手消息').last();
  await finalAnswer.getByRole('button', { name: '细化问题' }).click();
  assert.equal(await page.getByPlaceholder('请输入需要检索的问题').inputValue(), '不足原因 7');
  const gapReport = finalAnswer.getByLabel('知识缺口报告');
  await gapReport.getByRole('button', { name: '报告知识缺口' }).waitFor();
  assert.equal(
    await gapReport.getByRole('button', { name: '报告知识缺口' }).isVisible(),
    true
  );
});

test('an insufficient reply prioritizes Chinese related non-supporting coverage over unrelated earlier topics', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.route('**/api/knowledge-map', async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        data: {
          total_entries: 4,
          themes: [
            {
              domain: 'engineering',
              label: 'Engineering',
              entries: [
                {
                  entry_id: 'rag-chunking',
                  title: '分块策略',
                  suggested_query: '如何配置分块策略？'
                },
                {
                  entry_id: 'retrieval-ranking',
                  title: '检索排序',
                  suggested_query: '如何配置检索排序？'
                },
                {
                  entry_id: 'memory-limits',
                  title: '记忆容量',
                  suggested_query: '如何配置记忆容量？'
                },
                {
                  entry_id: 'mcp-permissions',
                  title: '工具权限',
                  suggested_query: '如何配置工具权限？'
                }
              ]
            }
          ]
        }
      })
    });
  });
  await page.route('**/api/chat/stream', async (route) => {
    const request = JSON.parse(route.request().postData() || '{}');
    const answerText = '没有已发布决定能够支持工具权限的当前请求。';
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: [
        ...completedInsufficientEvents({
          answerId: 'assistant-related-coverage',
          executionId: 'answer-execution-related-coverage',
          qcsId: 'qcs-related-coverage',
          question: request.message,
          reason: 'decision_not_covered',
          answerText
        }),
        `event: content\ndata: ${JSON.stringify({ content: answerText })}\n\n`,
        'event: done\ndata: [DONE]\n\n'
      ].join('')
    });
  });
  await page.goto(`${baseUrl}chat`);
  await sendQuestion(page, '生产环境下应该如何配置工具权限？');

  const coverage = page.getByLabel('助手消息').last().getByLabel('非支持性已发布覆盖');
  await coverage.getByText('工具权限', { exact: true }).waitFor();
  assert.equal(await coverage.getByRole('button', { name: '如何配置工具权限？' }).isVisible(), true);
});

test('a narrow social reply is explicit and a knowledge request is never relabeled as social', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.goto(`${baseUrl}chat`);

  await sendQuestion(page, '你好');
  const socialReply = page.getByLabel('助手消息').last();
  const nonKnowledgeBase = socialReply.getByLabel('非知识库回复');
  await nonKnowledgeBase.waitFor();
  assert.equal(await nonKnowledgeBase.getByText('Non-Knowledge-Base Reply', { exact: true }).isVisible(), true);
  assert.equal(await socialReply.getByText('Decision Summary', { exact: true }).count(), 0);
  assert.equal(await socialReply.getByRole('button', { name: /打开引用/ }).count(), 0);

  await sendQuestion(page, '今天天气如何');
  const knowledgeRequest = page.getByLabel('助手消息').last();
  await knowledgeRequest.getByLabel('证据不足回复').waitFor();
  assert.equal(await knowledgeRequest.getByText('Non-Knowledge-Base Reply', { exact: true }).count(), 0);
  assert.equal(await knowledgeRequest.getByRole('button', { name: /打开引用/ }).count(), 0);
});

test('Generation Unavailable exposes neither a knowledge answer nor frozen provider text and retries safely', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.route('**/api/chat/stream', async (route) => {
    const request = JSON.parse(route.request().postData() || '{}');
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: completedGenerationUnavailableEvents({
        answerId: 'assistant-generation-unavailable',
        executionId: 'answer-execution-generation-unavailable',
        qcsId: 'qcs-generation-unavailable',
        question: request.message
      }).join('')
    });
  });
  await page.goto(`${baseUrl}chat`);
  await sendQuestion(page, '生成失败时如何展示？');

  const answer = page.getByLabel('助手消息').last();
  const unavailable = answer.getByLabel('生成不可用');
  await unavailable.waitFor();
  assert.equal(await unavailable.getByText('Generation Unavailable', { exact: true }).isVisible(), true);
  assert.equal(await answer.getByText('Decision Summary', { exact: true }).count(), 0);
  assert.equal(await answer.getByRole('button', { name: /打开引用/ }).count(), 0);
  assert.equal(await answer.getByLabel('证据摘要').count(), 0);
  assert.equal(
    await answer.getByText('This frozen provider response must not be rendered.', { exact: true }).count(),
    0
  );
  assert.equal(await answer.getByRole('button', { name: '重试' }).isVisible(), true);
});

test('closed stopped, failed, throttled, and rejected executions remain visibly distinct', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  const cases = [
    ['stopped', 'Stopped', 'ANSWER_EXECUTION_STOPPED'],
    ['failed', 'Failed', 'ANSWER_EXECUTION_FAILED'],
    ['throttled', 'Throttled', 'ANSWER_EXECUTION_THROTTLED'],
    ['rejected', 'Rejected', 'ANSWER_EXECUTION_REJECTED']
  ];
  let turn = 0;
  await page.route('**/api/chat/stream', async (route) => {
    const [state, _label, failureCode] = cases[turn];
    const request = JSON.parse(route.request().postData() || '{}');
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: terminalExecutionEvents({
        answerId: `assistant-${state}`,
        executionId: `answer-execution-${state}`,
        qcsId: `qcs-${state}`,
        question: request.message,
        state,
        failureCode
      }).join('')
    });
    turn += 1;
  });
  await page.goto(`${baseUrl}chat`);

  for (const [index, [_state, label]] of cases.entries()) {
    await sendQuestion(page, `终态 ${index + 1}`);
    const answer = page.getByLabel('助手消息').last();
    const terminal = answer.locator('.answer-outcome--terminal');
    await terminal.getByText(label, { exact: true }).waitFor();
    assert.equal(await answer.getByRole('button', { name: /打开引用/ }).count(), 0);
    assert.equal(await answer.getByText('Insufficient Evidence Reply', { exact: true }).count(), 0);
    assert.equal(await answer.getByRole('button', { name: '重试' }).isVisible(), true);
  }
});

test('a terminal execution with streamed content fails closed instead of discarding the protocol violation', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.route('**/api/chat/stream', async (route) => {
    const terminalEvents = terminalExecutionEvents({
      answerId: 'assistant-terminal-content',
      executionId: 'answer-execution-terminal-content',
      qcsId: 'qcs-terminal-content',
      question: '终态不得保留流式内容',
      state: 'stopped',
      failureCode: 'ANSWER_EXECUTION_STOPPED'
    });
    terminalEvents.splice(2, 0, 'event: content\ndata: {"content":"terminal content must remain untrusted"}\n\n');
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: terminalEvents.join('')
    });
  });
  await page.goto(`${baseUrl}chat`);

  await sendQuestion(page, '终态不得保留流式内容');

  const answer = page.getByLabel('助手消息').last();
  await expectClosedResultUnavailable(answer);
  assert.equal(await answer.getByText('terminal content must remain untrusted', { exact: true }).count(), 0);
});

test('historical withdrawal retains frozen identities while withholding excerpts and the original-source link', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  const history = completedSupportedHistory({ withdrawn: true });
  await page.route(/\/api\/sessions(?:\?.*)?$/, async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        data: {
          sessions: [
            {
              session_id: 'session-withdrawn-history',
              updated_at: '2026-09-07T08:00:00Z',
              message_count: 2
            }
          ]
        }
      })
    });
  });
  await page.route(/\/api\/sessions\/session-withdrawn-history(?:\?.*)?$/, async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        data: {
          session_id: 'session-withdrawn-history',
          messages: [history.user, history.assistant]
        }
      })
    });
  });
  await page.goto(`${baseUrl}chat`);

  const rail = page.getByRole('complementary', { name: '最近会话' });
  await rail.getByRole('button', { name: /^session-withdrawn-history/ }).click();
  const answer = page.getByLabel('助手消息').last();
  await answer.getByText('Supported by published knowledge', { exact: true }).last().waitFor();
  await answer.getByText(/This source has been withdrawn\./).first().waitFor();
  assert.equal(await answer.getByText('Known execution paths should use deterministic workflows.').count(), 0);
  assert.equal(await answer.getByRole('button', { name: '打开引用 S1' }).count(), 4);
  await answer.getByRole('button', { name: '打开引用 S1' }).first().click();
  const entry = page.getByRole('complementary', { name: '来源摘录' });
  await entry.getByText('Governing Engineering Decision Entry', { exact: true }).waitFor();
  assert.equal(
    await entry.getByText('This source has been withdrawn.', { exact: true }).isVisible(),
    true
  );
  assert.equal(await entry.getByText('recommendation_or_reviewed_branches', { exact: true }).isVisible(), true);
  assert.match(await entry.getByText(/^[9]{64}$/).first().innerText(), /^9{64}$/);
  assert.equal(await entry.getByRole('link', { name: '打开公开来源' }).count(), 0);
  assert.equal(await entry.getByText('Known execution paths should use deterministic workflows.').count(), 0);
  const record = answer.locator('details.frozen-answer-record');
  await record.locator('summary').click();
  assert.match(await record.getByText(/^[9]{64}$/).innerText(), /^9{64}$/);
});

test('refining a historical insufficiency restores its frozen Query Condition Set as an editable draft', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  const history = completedInsufficientHistory({
    question: '带条件的历史不足问题'
  });
  const conditions = [
    {
      condition_id: 'environment-staging',
      field: 'environment',
      operator: 'equals',
      value: 'staging'
    }
  ];
  history.user.answer_execution.query_condition_set.conditions = conditions;
  history.assistant.answer_execution.query_condition_set.conditions = conditions;
  const requestPayloads = [];

  await page.route(/\/api\/sessions(?:\?.*)?$/, async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        data: {
          sessions: [
            {
              session_id: 'session-conditioned-insufficiency',
              updated_at: '2026-09-07T08:00:00Z',
              message_count: 2
            }
          ]
        }
      })
    });
  });
  await page.route(/\/api\/sessions\/session-conditioned-insufficiency(?:\?.*)?$/, async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        data: {
          session_id: 'session-conditioned-insufficiency',
          messages: [history.user, history.assistant]
        }
      })
    });
  });
  await page.route('**/api/chat/stream', async (route) => {
    const request = JSON.parse(route.request().postData() || '{}');
    requestPayloads.push(request);
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: [
        ...completedInsufficientEvents({
          answerId: 'assistant-refined-condition',
          executionId: 'answer-execution-refined-condition',
          qcsId: 'qcs-refined-condition',
          question: request.message,
          conditions: request.query_conditions || [],
          answerText: '已完成的细化不足回答。'
        }),
        'event: content\ndata: {"content":"已完成的细化不足回答。"}\n\n',
        'event: done\ndata: [DONE]\n\n'
      ].join('')
    });
  });
  await page.goto(`${baseUrl}chat`);

  const rail = page.getByRole('complementary', { name: '最近会话' });
  await rail.getByRole('button', { name: /^session-conditioned-insufficiency/ }).click();
  const answer = page.getByLabel('助手消息').last();
  await answer.getByLabel('证据不足回复').waitFor();
  await answer.getByRole('button', { name: '细化问题' }).click();

  const composerConditions = page.locator('.composer').getByLabel('查询条件');
  assert.equal(await page.getByPlaceholder('请输入需要检索的问题').inputValue(), '带条件的历史不足问题');
  assert.equal(await composerConditions.getByLabel('条件 1 字段').inputValue(), 'environment');
  assert.equal(await composerConditions.getByLabel('条件 1 运算符').inputValue(), 'equals');
  assert.equal(await composerConditions.getByLabel('条件 1 值').inputValue(), 'staging');
  await composerConditions.getByLabel('条件 1 值').fill('production');
  await page.getByRole('button', { name: '发送' }).click();
  await page.getByLabel('助手消息').last().getByLabel('证据不足回复').waitFor();

  assert.deepEqual(requestPayloads, [
    {
      message: '带条件的历史不足问题',
      session_id: 'session-conditioned-insufficiency',
      query_conditions: [
        {
          condition_id: 'environment-staging',
          field: 'environment',
          operator: 'equals',
          value: 'production'
        }
      ]
    }
  ]);
  assert.equal(requestPayloads[0].inherit_conditions, undefined);
});

test('a refreshed historical withdrawal keeps the open source panel as a read-only identity record', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  let withdrawn = false;
  let historyRequests = 0;
  await page.route(/\/api\/sessions(?:\?.*)?$/, async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        data: {
          sessions: [
            {
              session_id: 'session-refresh-withdrawal',
              updated_at: '2026-09-07T08:00:00Z',
              message_count: 2
            }
          ]
        }
      })
    });
  });
  await page.route(/\/api\/sessions\/session-refresh-withdrawal(?:\?.*)?$/, async (route) => {
    historyRequests += 1;
    const history = completedSupportedHistory({ withdrawn });
    await route.fulfill({
      contentType: 'application/json',
      headers: { 'cache-control': 'no-store' },
      body: JSON.stringify({
        data: {
          session_id: 'session-refresh-withdrawal',
          messages: [history.user, history.assistant]
        }
      })
    });
  });
  await page.goto(`${baseUrl}chat`);

  const rail = page.getByRole('complementary', { name: '最近会话' });
  const session = rail.getByRole('button', { name: /^session-refresh-withdrawal/ });
  await session.click();
  const answer = page.getByLabel('助手消息').last();
  await answer.getByLabel('已支持的知识回答').waitFor();
  await answer.getByRole('button', { name: '打开引用 S1' }).first().click();
  const excerpt = page.getByRole('complementary', { name: '来源摘录' });
  await excerpt.getByText('Known execution paths should use deterministic workflows.', { exact: true }).waitFor();
  assert.equal(await excerpt.getByRole('link', { name: '打开公开来源' }).isVisible(), true);

  await page.waitForFunction(
    () =>
      Array.from(document.querySelectorAll('.session-select')).some(
        (button) =>
          button.textContent?.includes('session-refresh-withdrawal') &&
          !button.hasAttribute('disabled')
      )
  );
  withdrawn = true;
  const refreshedHistory = page.waitForResponse((response) =>
    response.url().includes('/api/sessions/session-refresh-withdrawal')
  );
  await session.click();
  await refreshedHistory;
  await answer.getByText(/This source has been withdrawn\./).first().waitFor();
  await excerpt.getByText('This source has been withdrawn.', { exact: true }).waitFor();

  assert.equal(historyRequests, 2);
  assert.equal(await excerpt.getByText('recommendation_or_reviewed_branches', { exact: true }).isVisible(), true);
  assert.match(await excerpt.getByText(/^[9]{64}$/).first().innerText(), /^9{64}$/);
  assert.equal(await excerpt.getByText('Known execution paths should use deterministic workflows.', { exact: true }).count(), 0);
  assert.equal(await excerpt.getByRole('link', { name: '打开公开来源' }).count(), 0);
});

test('historical assistant-persistence failure remains visible on its user turn without a fabricated assistant answer', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  const history = persistenceFailureHistory();
  await page.route(/\/api\/sessions(?:\?.*)?$/, async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        data: {
          sessions: [
            {
              session_id: 'session-persistence-failure-history',
              updated_at: '2026-09-07T08:00:00Z',
              message_count: 1
            }
          ]
        }
      })
    });
  });
  await page.route(/\/api\/sessions\/session-persistence-failure-history(?:\?.*)?$/, async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        data: {
          session_id: 'session-persistence-failure-history',
          messages: [history.user]
        }
      })
    });
  });
  await page.goto(`${baseUrl}chat`);

  const rail = page.getByRole('complementary', { name: '最近会话' });
  await rail.getByRole('button', { name: /^session-persistence-failure-history/ }).click();
  const user = page.getByLabel('用户消息').last();
  const failure = user.getByLabel('持久化失败');
  await failure.getByText('Failed', { exact: true }).waitFor();
  assert.equal(
    await failure.getByText('无法持久化回答；未形成回答或引用。', { exact: true }).isVisible(),
    true
  );
  assert.equal(await page.getByLabel('助手消息').count(), 0);
  assert.equal(await user.getByRole('button', { name: /打开引用/ }).count(), 0);
});

test('live assistant-persistence failure removes the temporary assistant and stays on its user turn', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  const question = '实时持久化失败不得留下临时助手消息';
  await page.route('**/api/chat/stream', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: assistantPersistenceFailureEvents({
        answerId: 'assistant-live-persistence-failure',
        executionId: 'answer-execution-live-persistence-failure',
        qcsId: 'qcs-live-persistence-failure',
        question
      }).join('')
    });
  });
  await page.goto(`${baseUrl}chat`);

  await sendQuestion(page, question);

  const user = page.getByLabel('用户消息').last();
  await user.getByLabel('持久化失败').waitFor();
  assert.equal(await page.getByLabel('助手消息').count(), 0);
  assert.equal(await user.getByRole('button', { name: /打开引用/ }).count(), 0);
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
  await expectClosedResultUnavailable(answer);
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
  await expectClosedResultUnavailable(answer);
  assert.equal(await answer.getByText('partial answer').count(), 0);
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
  await expectClosedResultUnavailable(answer);
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
  await failedAnswer.locator('.answer-outcome--terminal').getByText('Failed', { exact: true }).waitFor();
  await failedAnswer.getByRole('button', { name: '重试' }).click();
  const retriedAnswer = page.getByLabel('助手消息').last();
  await expectInsufficientEvidenceReply(retriedAnswer);

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
  await expectClosedResultUnavailable(answer);
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
    await expectClosedResultUnavailable(answer);
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
  await expectClosedResultUnavailable(answer);
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
  await expectClosedResultUnavailable(answer);
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
    await expectClosedResultUnavailable(answer);
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
  await expectClosedResultUnavailable(answer);
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
  await expectClosedResultUnavailable(answer);
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
  await expectClosedResultUnavailable(answer);
});

test('Conversation Workspace renders a frozen insufficient answer through the authenticated SSE workflow', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await sendQuestion(page, '部署前需要做什么？');

  const answer = page.getByLabel('助手消息').last();
  await expectInsufficientEvidenceReply(answer);
  assert.equal(
    await answer.getByText('未检索到足够相关的知识片段，请补充更具体的问题或关键词。').isVisible(),
    true
  );
  const completedBox = await answer.boundingBox();
  assert.equal(completedBox.height > 0, true);
});

test('reload replays a retained closed result without sending another chat request', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.goto(`${baseUrl}chat`);

  const firstStream = page.waitForRequest('**/api/chat/stream');
  await sendQuestion(page, '部署前需要做什么？');
  const request = await firstStream;
  const sessionId = JSON.parse(request.postData() || '{}').session_id;
  assert.match(sessionId, /^session_/);

  const firstAnswer = page.getByLabel('助手消息').last();
  await expectInsufficientEvidenceReply(firstAnswer);

  let replayStreamCalls = 0;
  await page.route('**/api/chat/stream', async (route) => {
    replayStreamCalls += 1;
    await route.abort();
  });
  await page.reload();

  const sessionButton = page
    .getByRole('complementary', { name: '最近会话' })
    .getByRole('button', { name: /^部署前需要做什么？/ });
  await sessionButton.click();
  await page.waitForFunction(
    (title) =>
      Array.from(document.querySelectorAll('.session-select')).some(
        (button) => button.textContent?.includes(title) && button.getAttribute('aria-current') === 'true'
      ),
    '部署前需要做什么？'
  );

  const reloadedAnswer = page.getByLabel('助手消息').last();
  await expectInsufficientEvidenceReply(reloadedAnswer);
  assert.equal(await reloadedAnswer.getByText('未检索到足够相关的知识片段，请补充更具体的问题或关键词。').isVisible(), true);
  assert.equal(await reloadedAnswer.getByRole('button', { name: /打开引用/ }).count(), 0);
  assert.equal(replayStreamCalls, 0);
});

test('stopping a real streamed answer recovers its same execution terminal record', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {
    env: { BROWSER_ACCEPTANCE_LLM_DELAY_MS: '4000' }
  });
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await sendQuestion(page, '部署前需要做什么？');
  const answer = page.getByLabel('助手消息').last();
  await page.getByRole('status').filter({ hasText: '正在检索' }).waitFor();
  await page.getByRole('button', { name: '停止' }).click();

  const terminal = answer.locator('.answer-outcome--terminal');
  await terminal.getByText('Stopped', { exact: true }).waitFor();
  assert.equal(await answer.getByText('该请求已停止，未形成完整回答。', { exact: true }).isVisible(), true);
  assert.equal(await answer.getByText('部署前需要完成变更审批。', { exact: true }).count(), 0);
});

test('identity-free cancellation remains explicitly stopped and retains only a safe retry', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.addInitScript(() => {
    const originalFetch = window.fetch.bind(window);
    window.fetch = async (input, init) => {
      const url = typeof input === 'string' ? input : input.url;
      if (!url.includes('/api/chat/stream')) return originalFetch(input, init);
      const body = new ReadableStream({
        start(controller) {
          init?.signal?.addEventListener(
            'abort',
            () => controller.error(new DOMException('Aborted', 'AbortError')),
            { once: true }
          );
        }
      });
      return new Response(body, {
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' }
      });
    };
  });
  await page.goto(`${baseUrl}chat`);
  await sendQuestion(page, '在身份到达前停止请求');
  await page.getByRole('button', { name: '停止' }).click();

  const answer = page.getByLabel('助手消息').last();
  await answer.getByLabel('本地停止').waitFor();
  assert.equal(await answer.getByLabel('本地停止').getByText('Stopped', { exact: true }).isVisible(), true);
  assert.equal(await answer.getByRole('button', { name: '重试' }).isVisible(), true);
  assert.equal(await answer.getByText('Closed result unavailable', { exact: true }).count(), 0);
});

test('identity-free request failure is explicitly failed and retries only explicit conditions', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  const requestPayloads = [];
  await page.route('**/api/chat/stream', async (route) => {
    requestPayloads.push(JSON.parse(route.request().postData() || '{}'));
    if (requestPayloads.length === 1) {
      await route.abort('failed');
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: [
        ...completedInsufficientEvents({
          answerId: 'assistant-identity-free-retry',
          executionId: 'answer-execution-identity-free-retry',
          qcsId: 'qcs-identity-free-retry',
          question: requestPayloads[1].message,
          conditions: requestPayloads[1].query_conditions,
          answerText: '重试后的闭合不足结果。'
        }),
        'event: content\ndata: {"content":"重试后的闭合不足结果。"}\n\n',
        'event: done\ndata: [DONE]\n\n'
      ].join('')
    });
  });
  await page.goto(`${baseUrl}chat`);
  const composerConditions = page.locator('.composer').getByLabel('查询条件');
  await composerConditions.getByRole('button', { name: '添加查询条件' }).click();
  await composerConditions.getByLabel('条件 1 字段').fill('environment');
  await composerConditions.getByLabel('条件 1 运算符').fill('equals');
  await composerConditions.getByLabel('条件 1 值').fill('production');
  await sendQuestion(page, '在身份到达前请求失败');

  const failedAnswer = page.getByLabel('助手消息').last();
  await failedAnswer.getByLabel('本地执行失败').waitFor();
  assert.equal(
    await failedAnswer.getByLabel('本地执行失败').getByText('Failed', { exact: true }).isVisible(),
    true
  );
  await failedAnswer.getByRole('button', { name: '重试' }).click();
  await page.getByLabel('助手消息').last().getByLabel('证据不足回复').waitFor();
  assert.deepEqual(requestPayloads[1].query_conditions, [
    {
      condition_id: requestPayloads[0].query_conditions[0].condition_id,
      field: 'environment',
      operator: 'equals',
      value: 'production'
    }
  ]);
  assert.notEqual(requestPayloads[1].inherit_conditions, true);
});

test('identity-free concurrency-limit SSE error remains throttled after its done frame', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.route('**/api/chat/stream', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: [
        'event: error\ndata: {"code":"CHAT_CONCURRENCY_LIMIT_REACHED","message":"concurrency limit reached"}\n\n',
        'event: done\ndata: [DONE]\n\n'
      ].join('')
    });
  });
  await page.goto(`${baseUrl}chat`);
  await sendQuestion(page, '限流必须显示为独立状态');

  const answer = page.getByLabel('助手消息').last();
  await answer.getByLabel('本地限流').waitFor();
  assert.equal(await answer.getByText('Throttled', { exact: true }).count() >= 1, true);
  assert.equal(await answer.getByRole('button', { name: '重试' }).isVisible(), true);
  assert.equal(await answer.getByText('Failed', { exact: true }).count(), 0);
});

test('a semantic frame after an identity-free local terminal done fails closed', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.route('**/api/chat/stream', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: [
        'event: error\ndata: {"code":"CHAT_CONCURRENCY_LIMIT_REACHED","message":"concurrency limit reached"}\n\n',
        'event: done\ndata: [DONE]\n\n',
        'event: content\ndata: {"content":"UNTRUSTED_AFTER_DONE"}\n\n'
      ].join('')
    });
  });
  await page.goto(`${baseUrl}chat`);
  await sendQuestion(page, '终态后语义帧不得被降级为可重试失败');

  const answer = page.getByLabel('助手消息').last();
  await expectClosedResultUnavailable(answer);
  assert.equal(await answer.getByText('UNTRUSTED_AFTER_DONE', { exact: true }).count(), 0);
  assert.equal(await answer.getByLabel('本地执行失败').count(), 0);
  assert.equal(await answer.getByLabel('本地限流').count(), 0);
});

for (const [name, terminalViolation] of [
  [
    'a semantic frame after a bare done',
    'event: content\ndata: {"content":"UNTRUSTED_AFTER_BARE_DONE"}\n\n'
  ],
  ['a repeated bare done', 'event: done\ndata: [DONE]\n\n']
]) {
  test(`${name} fails closed without retry authority`, { timeout: 30000 }, async (t) => {
    const { page, baseUrl } = await startAsKnowledgeUser(t, {});
    await page.route('**/api/chat/stream', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: ['event: done\ndata: [DONE]\n\n', terminalViolation].join('')
      });
    });
    await page.goto(`${baseUrl}chat`);
    await sendQuestion(page, '无执行身份的终态违规不得取得重试权限');

    const answer = page.getByLabel('助手消息').last();
    await expectClosedResultUnavailable(answer);
    assert.equal(await answer.getByLabel('本地执行失败').count(), 0);
  });
}

test('aborting after completed terminal fields fails closed without recovering a different completed history result', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  const question = '中止后不得用历史完成结果覆盖本轮';
  const partialCompletedStream = [
    ...completedInsufficientEvents({
      answerId: 'assistant-abort-completed-fields',
      executionId: 'answer-execution-abort-completed-fields',
      qcsId: 'qcs-abort-completed-fields',
      question
    }),
    'event: content\ndata: {"content":"本轮未形成终态。"}\n\n'
  ].join('');
  const recoveredHistory = completedInsufficientHistory({ question });
  let recoveryRequests = 0;
  await page.addInitScript(
    ({ stream }) => {
      const originalFetch = window.fetch.bind(window);
      window.fetch = async (input, init) => {
        const url = typeof input === 'string' ? input : input.url;
        if (!url.includes('/api/chat/stream')) {
          return originalFetch(input, init);
        }
        const encoder = new TextEncoder();
        const body = new ReadableStream({
          start(controller) {
            controller.enqueue(encoder.encode(stream));
            init?.signal?.addEventListener(
              'abort',
              () => controller.error(new DOMException('Aborted', 'AbortError')),
              { once: true }
            );
          }
        });
        return new Response(body, {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' }
        });
      };
    },
    { stream: partialCompletedStream }
  );
  await page.route(/\/api\/sessions\/session_[^/?]+(?:\?.*)?$/, async (route) => {
    recoveryRequests += 1;
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        data: {
          session_id: 'session-recovered-completed-history',
          messages: [recoveredHistory.user, recoveredHistory.assistant]
        }
      })
    });
  });
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await sendQuestion(page, question);
  await page.getByRole('status').filter({ hasText: '正在生成回答' }).waitFor();
  await page.getByRole('button', { name: '停止' }).click();

  const answer = page.getByLabel('助手消息').last();
  await expectClosedResultUnavailable(answer);
  assert.equal(recoveryRequests, 0);
  assert.equal(await answer.getByText('已完成的历史不足回答。', { exact: true }).count(), 0);
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
  await answer.getByLabel('本地执行失败').waitFor();
  assert.equal(await answer.getByText('Failed', { exact: true }).count() >= 1, true);
  assert.equal(await answer.getByRole('button', { name: '重试' }).isVisible(), true);
  assert.equal(await answer.getByText('partial answer').count(), 0);
});

test('a post-admission stream interruption fails closed without fabricated retry authority', { timeout: 30000 }, async (t) => {
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
  await expectClosedResultUnavailable(answer);
  assert.equal(requestPayloads.length, 1);
});

test('an assistant identity before interruption fails closed without fabricated retry authority', { timeout: 30000 }, async (t) => {
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
  await expectClosedResultUnavailable(answer);
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

  await page.getByRole('button', { name: /^历史的部署问题/ }).click();
  const firstHistoricalAnswer = page.getByLabel('助手消息').first();
  await expectClosedResultUnavailable(firstHistoricalAnswer);
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
    await expectClosedResultUnavailable(answer);
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
  await expectInsufficientEvidenceReply(firstAnswer);

  await sendQuestion(page, '部署前需要做什么？');
  const secondAnswer = page.getByLabel('助手消息').last();
  await expectInsufficientEvidenceReply(secondAnswer);
  assert.equal(await page.getByText('【生成不可用】生成服务暂不可用，请稍后重试。').count(), 0);
});

test('an evidence gate rejection is shown as insufficient evidence instead of a response failure', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.goto(`${baseUrl}chat`);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();

  await sendQuestion(page, 'zzzz 完全不存在的内容 987654321');
  await expectInsufficientEvidenceReply(page.getByLabel('助手消息').last());
  assert.equal(await page.getByText('未检索到足够相关的知识片段，请补充更具体的问题或关键词。').isVisible(), true);
  assert.equal(await page.getByText('Closed result unavailable').count(), 0);
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
  await rail.getByRole('button', { name: /^历史的部署问题/ }).click();
  const userMessage = page.getByLabel('用户消息');
  const assistantMessage = page.getByLabel('助手消息');
  await assistantMessage.first().waitFor();
  assert.match(await userMessage.first().innerText(), /你\s+历史的部署问题/);
  await expectClosedResultUnavailable(assistantMessage.first());

  await page.getByRole('button', { name: '新建会话' }).first().click();
  await page.getByRole('heading', { name: '从团队知识开始提问' }).waitFor();
  assert.equal(await page.getByText('标题').count(), 0);

  await rail.getByRole('button', { name: /^历史的部署问题/ }).click();
  await rail.getByRole('button', { name: '删除会话 历史的部署问题' }).click();
  const dialog = page.getByRole('dialog');
  await dialog.getByRole('button', { name: '删除' }).click();
  await rail.getByText('历史的部署问题').waitFor({ state: 'detached' });
  assert.equal(await rail.getByText('历史的部署问题').count(), 0);
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
  await page.getByText('当前没有可浏览的已发布决策。').waitFor();
  assert.equal(await page.getByText('当前没有可浏览的已发布决策。').isVisible(), true);
});

test('a completed frozen insufficiency remains readable without overflow at a 390-pixel viewport', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, { viewport: { width: 390, height: 844 } });
  await page.goto(`${baseUrl}chat`);
  await sendQuestion(page, '部署前需要做什么？');
  const answer = page.getByLabel('助手消息').last();
  await expectInsufficientEvidenceReply(answer);
  assert.equal(
    await answer.getByText('未检索到足够相关的知识片段，请补充更具体的问题或关键词。').isVisible(),
    true
  );
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
});

test('a real current-authority supported answer reloads without replaying chat and keeps citation navigation usable on narrow screens', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, { viewport: { width: 1440, height: 900 } });
  await page.goto(`${baseUrl}chat`);

  const composerConditions = page.locator('.composer').getByLabel('查询条件');
  await composerConditions.getByRole('button', { name: '添加查询条件' }).click();
  await composerConditions.getByLabel('条件 1 字段').fill('execution_path');
  await composerConditions.getByLabel('条件 1 运算符').fill('equals');
  await composerConditions.getByLabel('条件 1 值').fill('known');
  const initialStream = page.waitForRequest('**/api/chat/stream');
  await sendQuestion(page, '什么时候使用 deterministic workflow？');
  const sessionId = JSON.parse((await initialStream).postData() || '{}').session_id;
  assert.match(sessionId, /^session_/);

  const answer = page.getByLabel('助手消息').last();
  await answer.getByLabel('已支持的知识回答').waitFor();
  await answer.getByRole('button', { name: '打开引用 S1' }).first().click();
  const sourcePanel = page.getByRole('complementary', { name: '来源摘录' });
  await sourcePanel.getByText('已知路径应由 deterministic workflow 控制。', { exact: true }).waitFor();
  assert.equal(await sourcePanel.getByRole('link', { name: '打开公开来源' }).isVisible(), true);

  let replayStreamCalls = 0;
  await page.route('**/api/chat/stream', async (route) => {
    replayStreamCalls += 1;
    await route.abort();
  });
  await page.reload();
  await page
    .getByRole('complementary', { name: '最近会话' })
    .getByRole('button', { name: /^什么时候使用 deterministic workflow？/ })
    .click();

  const reloadedAnswer = page.getByLabel('助手消息').last();
  await reloadedAnswer.getByLabel('已支持的知识回答').waitFor();
  await page.setViewportSize({ width: 390, height: 844 });
  await reloadedAnswer.getByRole('button', { name: '打开引用 S1' }).first().click();
  await page.getByRole('complementary', { name: '来源摘录' }).waitFor();
  assert.equal(
    await page.evaluate(() => {
      const messageList = document.querySelector('.message-list');
      return Boolean(messageList) && messageList.scrollWidth <= messageList.clientWidth;
    }),
    true
  );
  await page.waitForFunction(() => document.documentElement.scrollWidth <= window.innerWidth);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
  assert.equal(replayStreamCalls, 0);
});

test('a Knowledge User sees a frozen insufficient result without fabricated excerpts or diagnostics', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, { viewport: { width: 1440, height: 900 } });
  await page.goto(`${baseUrl}chat`);
  await sendQuestion(page, '部署前需要做什么？');

  const answer = page.getByLabel('助手消息').last();
  await expectInsufficientEvidenceReply(answer);
  assert.equal(await page.getByText('RAG Trace').count(), 0);
  assert.equal(await page.getByLabel('检索诊断').count(), 0);
  assert.equal(await answer.getByLabel('知识反馈').count(), 0);
});

test('an ineligible retrieval hit does not turn into a Public Source Citation', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {
    viewport: { width: 1440, height: 900 },
    env: { BROWSER_ACCEPTANCE_SOURCE_LOST: '1' }
  });
  await page.goto(`${baseUrl}chat`);
  await sendQuestion(page, '什么时候使用 deterministic workflow？');

  const answer = page.getByLabel('助手消息').last();
  await expectInsufficientEvidenceReply(answer);
  assert.equal(await page.getByRole('complementary', { name: '来源摘录' }).count(), 0);
  assert.equal(await answer.getByText('Anthropic', { exact: true }).count(), 0);
  assert.equal(await answer.getByText(/internal-chunk|internal-document/).count(), 0);

  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForFunction(() => document.documentElement.scrollWidth <= window.innerWidth);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
});

test('historical messages without a closed execution never expose legacy evidence or answer text', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startAsKnowledgeUser(t, {});
  await page.goto(`${baseUrl}chat`);
  const historyButton = page
    .getByRole('complementary', { name: '最近会话' })
    .getByRole('button', { name: /^历史的部署问题/ });
  await historyButton.click();
  await page.waitForFunction(
    () =>
      Array.from(document.querySelectorAll('.session-select')).some(
        (button) =>
          button.textContent?.includes('历史的部署问题') &&
          button.getAttribute('aria-current') === 'true'
      )
  );

  const historicalAnswers = page.getByLabel('助手消息');
  await historicalAnswers.first().waitFor();
  assert.equal(await historicalAnswers.count(), 3);
  for (let index = 0; index < await historicalAnswers.count(); index += 1) {
    await expectClosedResultUnavailable(historicalAnswers.nth(index));
  }
  assert.equal(await page.getByText('历史回答有可核对来源。').count(), 0);
  assert.equal(await page.getByText('历史来源摘录。').count(), 0);
  assert.equal(await page.getByRole('complementary', { name: '来源摘录' }).count(), 0);
  assert.equal(await page.getByLabel('检索诊断').count(), 0);
});

test('the chat workspace exposes bounded retrieval diagnostics only to an administrator', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, {});
  await loginAdmin(page, baseUrl);

  await sendQuestion(page, '部署前需要做什么？');

  await expectInsufficientEvidenceReply(page.getByLabel('助手消息').last());
  const diagnostics = page.getByLabel('检索诊断');
  await diagnostics.waitFor();
  assert.equal(await diagnostics.getByText('仅系统管理员', { exact: true }).isVisible(), true);
  await diagnostics.locator('summary').click();
  assert.equal(await diagnostics.getByText(/召回候选|重排候选|门禁拒绝/).count() > 0, true);
});

test('an administrator does not receive diagnostics from an unverified historical assistant record', { timeout: 30000 }, async (t) => {
  const { page, baseUrl } = await startWorkbench(t, {});
  await loginAdmin(page, baseUrl);
  const historyButton = page
    .getByRole('complementary', { name: '最近会话' })
    .getByRole('button', { name: /^未命名会话/ });
  await historyButton.click();
  await page.waitForFunction(
    () =>
      Array.from(document.querySelectorAll('.session-select')).some(
        (button) =>
          button.textContent?.includes('未命名会话') &&
          button.getAttribute('aria-current') === 'true'
      )
  );

  await expectClosedResultUnavailable(page.getByLabel('助手消息').last());
  assert.equal(await page.getByLabel('检索诊断').count(), 0);
});
