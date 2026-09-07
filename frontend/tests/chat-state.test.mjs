import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {
  extractRejectReason,
  formatStreamError,
  findRecoveredClosedExecution,
  getDoneStatus,
  getProviderStatus,
  resolveRetryTurn,
  validateClosedAssistantProjection,
  validateCompletedStreamProjection
} from '../src/store/chat-state.js';
import {
  getAnswerExecutionPresentation,
  getInsufficientEvidencePresentation,
  isRenderableFrozenSupportedAnswer,
  parseFrozenDecisionAnswer
} from '../src/app/answer-execution-presentation.js';

const contractVectors = JSON.parse(
  fs.readFileSync(new URL('../../docs/contracts/answer-execution-contract-vectors.json', import.meta.url), 'utf-8')
);

const completedStreamMessage = ({
  outcome = 'insufficient_evidence_reply',
  executionOutcome = outcome,
  state = 'completed',
  evidenceSummary = { coverage: 'insufficient', source_count: 0, sources: [] },
  insufficientEvidenceReply = {
    outcome: 'insufficient_evidence_reply',
    reason: 'decision_not_covered',
    query_condition_set_identity: 'qcs-1'
  }
} = {}) => {
  const reply = outcome === 'insufficient_evidence_reply' ? structuredClone(insufficientEvidenceReply) : null;
  const message = {
    id: 'assistant-1',
    content: 'A frozen terminal reply.',
    outcome,
    answer_execution: {
      id: 'answer-execution-1',
      assistant_message_id: 'assistant-1',
      state,
      question: 'What applies?',
      answer_text: 'A frozen terminal reply.',
      query_condition_set: {
        identity: 'qcs-1',
        normalized_question: 'What applies?',
        conditions: []
      },
      condition_provenance: { mode: 'question_normalized' },
      outcome: executionOutcome,
      evidence_set_identity: null,
      item_identities: [],
      snapshot_ids: [],
      knowledge_version_identities: [],
      insufficient_evidence_reply: reply,
      evidence_summary: structuredClone(evidenceSummary)
    },
    evidence_summary: evidenceSummary
  };
  if (reply !== null) {
    message.insufficient_evidence_reply = structuredClone(reply);
  }
  return message;
};

const conditionRecordsForVector = (vector) => {
  if (Number.isInteger(vector.condition_count)) {
    return Array.from({ length: vector.condition_count }, (_, index) => ({
      condition_id: `f${index}-v`,
      field: `f${index}`,
      operator: 'equals',
      value: 'v'
    }));
  }
  if (
    Number.isInteger(vector.condition_id_chars) &&
    Number.isInteger(vector.field_chars) &&
    typeof vector.operator === 'string' &&
    Number.isInteger(vector.value_chars)
  ) {
    const field = 'f'.repeat(vector.field_chars);
    const value = 'v'.repeat(vector.value_chars);
    return [
      {
        condition_id: `${field}-${value}`,
        field,
        operator: vector.operator,
        value
      }
    ];
  }
  throw new Error('invalid condition contract vector');
};

test('Answer Execution contract vectors keep JavaScript QCS, reason, and presentation seams aligned', () => {
  assert.equal(contractVectors.schema, 'answer_execution_contract_vectors/v1');
  assert.deepEqual(contractVectors.query_condition_limits, {
    max_conditions: 32,
    max_condition_id_chars: 160,
    max_condition_field_chars: 160,
    max_condition_operator_chars: 64,
    max_condition_value_chars: 512
  });
  const unicodeVector = contractVectors.unicode_code_point_condition_vector;
  const unicodeCondition = {
    condition_id: unicodeVector.condition.condition_id,
    field: unicodeVector.condition.field,
    operator: unicodeVector.condition.operator,
    value: unicodeVector.condition.value_character.repeat(unicodeVector.condition.value_code_points)
  };
  const unicodeMessage = completedStreamMessage();
  unicodeMessage.answer_execution.question = unicodeVector.question;
  unicodeMessage.answer_execution.query_condition_set = {
    identity: 'qcs-unicode-code-points',
    normalized_question: unicodeVector.question,
    conditions: [unicodeCondition]
  };
  unicodeMessage.answer_execution.condition_provenance = { mode: 'explicit' };
  unicodeMessage.answer_execution.insufficient_evidence_reply.query_condition_set_identity =
    'qcs-unicode-code-points';
  unicodeMessage.insufficient_evidence_reply.query_condition_set_identity = 'qcs-unicode-code-points';
  assert.doesNotThrow(() =>
    validateCompletedStreamProjection(unicodeMessage, {
      question: unicodeVector.question,
      query_conditions: [unicodeCondition],
      inherit_conditions: false
    })
  );
  assert.deepEqual(
    resolveRetryTurn(
      [
        {
          role: 'user',
          content: unicodeVector.question,
          requested_query_conditions: [unicodeCondition],
          answer_execution: null
        }
      ],
      0
    ),
    {
      question: unicodeVector.question,
      query_conditions: [unicodeCondition],
      inherit_conditions: false
    }
  );

  for (const vector of contractVectors.question_condition_vectors) {
    const message = completedStreamMessage();
    const qcsIdentity = `qcs-vector-${vector.conditions.length}`;
    message.answer_execution.question = vector.question;
    message.answer_execution.query_condition_set = {
      identity: qcsIdentity,
      normalized_question: vector.question,
      conditions: structuredClone(vector.conditions)
    };
    message.answer_execution.insufficient_evidence_reply.query_condition_set_identity = qcsIdentity;
    message.insufficient_evidence_reply.query_condition_set_identity = qcsIdentity;
    assert.doesNotThrow(() =>
      validateCompletedStreamProjection(message, {
        question: vector.question,
        query_conditions: undefined,
        inherit_conditions: false
      })
    );
  }

  for (const vector of contractVectors.invalid_question_condition_vectors) {
    const message = completedStreamMessage();
    message.answer_execution.question = vector.question;
    message.answer_execution.query_condition_set = {
      identity: 'qcs-invalid-contract-vector',
      normalized_question: vector.question,
      conditions: conditionRecordsForVector(vector)
    };
    message.answer_execution.insufficient_evidence_reply.query_condition_set_identity =
      'qcs-invalid-contract-vector';
    message.insufficient_evidence_reply.query_condition_set_identity = 'qcs-invalid-contract-vector';
    assert.throws(
      () =>
        validateCompletedStreamProjection(message, {
          question: vector.question,
          query_conditions: undefined,
          inherit_conditions: false
        }),
      /condition limit/
    );
    assert.throws(
      () =>
        resolveRetryTurn(
          [
            {
              role: 'user',
              content: vector.question,
              answer_execution: {
                state: 'failed',
                question: vector.question,
                query_condition_set: message.answer_execution.query_condition_set
              }
            }
          ],
          0
        ),
      /condition limit/
    );
  }

  for (const vector of contractVectors.insufficient_evidence_presentations) {
    assert.deepEqual(getInsufficientEvidencePresentation(vector.reason), {
      title: vector.title,
      detail: vector.detail
    });
  }

  for (const vector of contractVectors.execution_presentations) {
    assert.deepEqual(
      getAnswerExecutionPresentation({
        state: vector.state,
        ...(vector.outcome ? { outcome: vector.outcome } : {})
      }),
      {
        kind: vector.kind,
        label: vector.label,
        retryable: vector.retryable
      }
    );
  }
});

test('extractRejectReason detects retrieve reject gates', () => {
  const reason = extractRejectReason({
    step: 'retrieve',
    detail: { gate_passed: false, gate_reason: 'low_reject' }
  });
  assert.equal(reason, 'low_reject');
});

test('extractRejectReason ignores non-reject steps', () => {
  assert.equal(
    extractRejectReason({
      step: 'retrieve',
      detail: { gate_passed: true, gate_reason: 'high_accept' }
    }),
    ''
  );
  assert.equal(extractRejectReason({ step: 'generate', detail: {} }), '');
});

test('formatStreamError formats known auth errors first', () => {
  assert.equal(formatStreamError({ code: 'AUTH_FORBIDDEN', message: 'Forbidden' }), '无权限执行当前问答');
  assert.equal(formatStreamError({ status: 401, message: 'Unauthorized' }), '登录状态已失效，请重新登录');
  assert.equal(formatStreamError('upstream unavailable'), 'upstream unavailable');
  assert.equal(formatStreamError({ message: 'boom' }), 'boom');
});

test('getDoneStatus uses the explicit completed outcome instead of evidence-summary inference', () => {
  assert.equal(getDoneStatus({ outcome: 'insufficient_evidence_reply', rejected: false }), '证据不足');
  assert.equal(getDoneStatus({ outcome: 'evidence_gated_answer', rejected: true }), '已完成');
  assert.equal(getDoneStatus({ outcome: 'non_knowledge_base_reply' }), '已完成');
  assert.equal(getDoneStatus({ outcome: 'generation_unavailable' }), '生成不可用');
});

test('Answer Execution presentation reads only its explicit state and closed outcome', () => {
  assert.deepEqual(
    getAnswerExecutionPresentation({
      state: 'completed',
      outcome: 'evidence_gated_answer',
      content: 'This text says failed.',
      failed: true,
      rejected: true
    }),
    {
      kind: 'supported',
      label: 'Supported by published knowledge',
      retryable: false
    }
  );

  assert.deepEqual(
    getAnswerExecutionPresentation({
      state: 'completed',
      outcome: 'insufficient_evidence_reply',
      content: 'This text says supported.',
      failed: false,
      rejected: false
    }),
    {
      kind: 'insufficient',
      label: 'Insufficient Evidence Reply',
      retryable: false
    }
  );

  assert.deepEqual(
    getAnswerExecutionPresentation({ state: 'completed', outcome: 'non_knowledge_base_reply' }),
    {
      kind: 'non-knowledge-base',
      label: 'Non-Knowledge-Base Reply',
      retryable: false
    }
  );

  assert.deepEqual(
    getAnswerExecutionPresentation({ state: 'completed', outcome: 'generation_unavailable' }),
    {
      kind: 'generation-unavailable',
      label: 'Generation Unavailable',
      retryable: true
    }
  );
});

test('Answer Execution presentation keeps terminal states distinct from all completed outcomes', () => {
  assert.deepEqual(
    getAnswerExecutionPresentation({ state: 'stopped', outcome: 'evidence_gated_answer' }),
    { kind: 'stopped', label: 'Stopped', retryable: true }
  );
  assert.deepEqual(
    getAnswerExecutionPresentation({ state: 'failed', outcome: 'insufficient_evidence_reply' }),
    { kind: 'failed', label: 'Failed', retryable: true }
  );
  assert.deepEqual(
    getAnswerExecutionPresentation({ state: 'throttled', outcome: 'generation_unavailable' }),
    { kind: 'throttled', label: 'Throttled', retryable: true }
  );
  assert.deepEqual(
    getAnswerExecutionPresentation({ state: 'rejected', outcome: 'non_knowledge_base_reply' }),
    { kind: 'rejected', label: 'Rejected', retryable: true }
  );
});

test('Insufficient Evidence presentation names only the frozen reason family', () => {
  assert.equal(
    getInsufficientEvidencePresentation('decisive_condition_missing').title,
    'A decisive condition is missing'
  );
  assert.equal(
    getInsufficientEvidencePresentation('knowledge_needs_review').title,
    'Published knowledge needs review'
  );
  assert.equal(getInsufficientEvidencePresentation('unknown_reason'), null);
});

test('frozen supported answer sections retain their text and citation markers for rendering', () => {
  const sections = parseFrozenDecisionAnswer(
    [
      '## Recommendation',
      'Use the reviewed workflow. [S1]',
      '',
      '## Applicability Limits',
      'Only for known paths. [S1]',
      '',
      '## Alternatives',
      'Escalate unknown paths. [S2]',
      '',
      '## Minimal Implementation or Acceptance Check',
      'Verify an explicit termination condition. [S1]'
    ].join('\n'),
    ['S1', 'S2']
  );

  assert.deepEqual(
    sections.map((section) => ({ id: section.id, label: section.label })),
    [
      { id: 'decision-summary', label: 'Decision Summary' },
      { id: 'applicability-limits', label: 'Applicability Limits' },
      { id: 'alternatives', label: 'Alternatives' },
      { id: 'minimum-check', label: 'Minimal Implementation or Acceptance Check' }
    ]
  );
  assert.deepEqual(sections[0].blocks[0], [
    { type: 'text', value: 'Use the reviewed workflow. ' },
    { type: 'citation', citationId: 'S1' }
  ]);
  assert.deepEqual(sections[2].blocks[0], [
    { type: 'text', value: 'Escalate unknown paths. ' },
    { type: 'citation', citationId: 'S2' }
  ]);
});

test('frozen supported answer presentation preserves unrecognized content without reclassifying its outcome', () => {
  const sources = [
    {
      citation_id: 'S1',
      citation_identity: 'a'.repeat(64),
      entry_id: 'workflow-entry-001',
      entry_title: 'Reviewed workflow boundary',
      section_id: 'recommendation_or_reviewed_branches',
      snapshot_id: 'b'.repeat(64),
      publication_version: 'v1',
      source_url: 'https://example.com/decision'
    },
    {
      citation_id: 'S2',
      citation_identity: 'c'.repeat(64),
      entry_id: 'workflow-entry-002',
      entry_title: 'Reviewed alternative boundary',
      section_id: 'alternatives',
      snapshot_id: 'd'.repeat(64),
      publication_version: 'v1',
      source_url: 'https://example.com/alternative'
    }
  ];
  const answerText = [
    '【Supported by published knowledge】',
    'Evidence-Bounded Implementation Aid',
    '',
    '## Recommendation',
    'Use the reviewed workflow. [S1]',
    '',
    '## Applicability Limits',
    'Only for known paths. [S1]',
    '',
    '## Alternatives',
    'Escalate unknown paths. [S1]',
    '',
    '## Minimal Implementation or Acceptance Check',
    'Verify an explicit termination condition. [S1]',
    '',
    '## Operational Notes',
    'Preserve the existing deployment window. [S1]'
  ].join('\n');

  const sections = parseFrozenDecisionAnswer(answerText, sources.map((source) => source.citation_id));
  assert.deepEqual(
    sections.map((section) => ({ id: section.id, label: section.label })),
    [
      { id: 'frozen-preamble', label: '' },
      { id: 'decision-summary', label: 'Decision Summary' },
      { id: 'applicability-limits', label: 'Applicability Limits' },
      { id: 'alternatives', label: 'Alternatives' },
      { id: 'minimum-check', label: 'Minimal Implementation or Acceptance Check' },
      { id: 'frozen-section-0', label: 'Operational Notes' }
    ]
  );
  assert.deepEqual(sections[0].blocks, [[
    { type: 'text', value: 'Evidence-Bounded Implementation Aid' }
  ]]);
  assert.deepEqual(sections.at(-1).blocks, [
    [
      { type: 'text', value: 'Preserve the existing deployment window. ' },
      { type: 'citation', citationId: 'S1' }
    ]
  ]);
  assert.equal(
    sections
      .flatMap((section) => section.blocks)
      .flat()
      .some((fragment) => fragment.type === 'text' && fragment.value.includes('【Supported by published knowledge】')),
    false
  );
  assert.equal(isRenderableFrozenSupportedAnswer(answerText, sources), true);
  assert.equal(
    isRenderableFrozenSupportedAnswer(
      answerText.replace(
        'Preserve the existing deployment window. [S1]',
        'Preserve the existing deployment window.'
      ),
      sources
    ),
    true
  );
  assert.equal(
    isRenderableFrozenSupportedAnswer(
      answerText.replace('Evidence-Bounded Implementation Aid', 'Uncited presentation commentary'),
      sources
    ),
    true
  );
  assert.equal(
    isRenderableFrozenSupportedAnswer(
      answerText.replace(
        'Evidence-Bounded Implementation Aid',
        'Evidence-Bounded Implementation Aid\n\nThis frozen implementation scope remains evidence-bounded. [S1]'
      ),
      sources
    ),
    true
  );
});

test('a closed supported answer does not become a contract failure for uncited presentation copy', () => {
  const source = {
    citation_id: 'S1',
    citation_identity: 'a'.repeat(64),
    entry_id: 'workflow-entry-001',
    entry_title: 'Reviewed workflow boundary',
    section_id: 'recommendation_or_reviewed_branches',
    snapshot_id: 'b'.repeat(64),
    publication_version: 'v1',
    source_url: 'https://example.com/decision'
  };
  const answerText = [
    'A concise presentation introduction.',
    '',
    '## Recommendation',
    'Use the reviewed workflow. [S1]',
    '',
    '## Applicability Limits',
    'Only for known paths. [S1]',
    '',
    '## Alternatives',
    'Escalate unknown paths. [S1]',
    '',
    '## Minimal Implementation or Acceptance Check',
    'Verify an explicit termination condition. [S1]'
  ].join('\n');

  assert.equal(isRenderableFrozenSupportedAnswer(answerText, [source]), true);
});

test('a supported presentation requires a frozen governing citation source but does not inspect answer text', () => {
  const answerText = [
    '## Recommendation',
    'Use the reviewed workflow. [S1]',
    '',
    '## Applicability Limits',
    'Only for known paths. [S1]',
    '',
    '## Alternatives',
    'Escalate unknown paths. [S1]',
    '',
    '## Minimal Implementation or Acceptance Check',
    'Verify an explicit termination condition. [S1]'
  ].join('\n');
  const source = {
    citation_id: 'S1',
    citation_identity: 'a'.repeat(64),
    entry_id: 'workflow-entry-001',
    entry_title: 'Reviewed workflow boundary',
    section_id: 'recommendation_or_reviewed_branches',
    snapshot_id: 'b'.repeat(64),
    publication_version: 'v1',
    source_url: 'https://example.com/decision'
  };

  assert.equal(isRenderableFrozenSupportedAnswer(answerText, [source]), true);
  assert.equal(isRenderableFrozenSupportedAnswer(answerText.replaceAll('[S1]', '[S9]'), [source]), true);
  assert.equal(
    isRenderableFrozenSupportedAnswer(answerText, [{ ...source, section_id: 'applicability' }]),
    false
  );
});

test('a supported presentation permits the frozen optional condition and version scope in either language', () => {
  const source = {
    citation_id: 'S1',
    citation_identity: 'a'.repeat(64),
    entry_id: 'workflow-entry-001',
    entry_title: 'Reviewed workflow boundary',
    section_id: 'recommendation_or_reviewed_branches',
    snapshot_id: 'b'.repeat(64),
    publication_version: 'v1',
    source_url: 'https://example.com/decision'
  };
  const english = [
    '## Recommendation',
    'Use the reviewed workflow. [S1]',
    '',
    '## Applicability Limits',
    'Only for known paths. [S1]',
    '',
    '## Alternatives',
    'Escalate unknown paths. [S1]',
    '',
    '## Minimal Implementation or Acceptance Check',
    'Verify an explicit termination condition. [S1]',
    '',
    '## Missing Conditions and Version Scope',
    'Confirm the deployment version before applying this path. [S1]'
  ].join('\n');
  const chinese = [
    '## 建议',
    '使用已复核的流程。 [S1]',
    '',
    '## 适用边界',
    '仅适用于已知路径。 [S1]',
    '',
    '## 备选方案',
    '未知路径需要升级处理。 [S1]',
    '',
    '## 最小实现或验收检查',
    '验证明确的终止条件。 [S1]',
    '',
    '## 缺失条件与版本范围',
    '在采用前确认部署版本。 [S1]'
  ].join('\n');

  assert.equal(isRenderableFrozenSupportedAnswer(english, [source]), true);
  assert.equal(isRenderableFrozenSupportedAnswer(chinese, [source]), true);
});

test('a supported presentation accepts only a scoped controlled source locator', () => {
  const answerText = [
    '## Recommendation',
    'Use the reviewed workflow. [S1]',
    '',
    '## Applicability Limits',
    'Only for known paths. [S1]',
    '',
    '## Alternatives',
    'Escalate unknown paths. [S1]',
    '',
    '## Minimal Implementation or Acceptance Check',
    'Verify an explicit termination condition. [S1]'
  ].join('\n');
  const controlledSource = {
    citation_id: 'S1',
    citation_identity: 'a'.repeat(64),
    entry_id: 'workflow-entry-001',
    entry_title: 'Reviewed workflow boundary',
    section_id: 'recommendation_or_reviewed_branches',
    snapshot_id: 'b'.repeat(64),
    publication_version: 'v1',
    source_access_scope: 'controlled_internal',
    source_url: 'controlled://knowledge/reviewed-decision-001'
  };

  assert.equal(isRenderableFrozenSupportedAnswer(answerText, [controlledSource]), true);
  assert.equal(
    isRenderableFrozenSupportedAnswer(answerText, [{ ...controlledSource, source_access_scope: 'public' }]),
    false
  );
  assert.equal(
    isRenderableFrozenSupportedAnswer(answerText, [{
      ...controlledSource,
      source_url: 'https://example.com/contradictory-controlled-source'
    }]),
    false
  );
});

test('closed-session recovery accepts only a new user execution bound to the interrupted turn', () => {
  const submittedTurn = {
    question: '恢复必须绑定同一轮问题',
    query_conditions: undefined,
    inherit_conditions: false,
    known_executions: []
  };
  const priorExecution = {
    id: 'execution-prior',
    state: 'completed',
    question: submittedTurn.question,
    query_condition_set: {
      identity: 'qcs-prior',
      normalized_question: submittedTurn.question,
      conditions: []
    },
    condition_provenance: { mode: 'question_normalized' }
  };
  const recoveredExecution = {
    id: 'execution-current',
    state: 'stopped',
    question: submittedTurn.question,
    query_condition_set: {
      identity: 'qcs-current',
      normalized_question: submittedTurn.question,
      conditions: []
    },
    condition_provenance: { mode: 'question_normalized' },
    assistant_message_id: 'assistant-current',
    failure_code: 'ANSWER_EXECUTION_STOPPED'
  };
  const incompleteCompletedExecution = {
    id: 'execution-incomplete-completed',
    state: 'completed',
    question: submittedTurn.question,
    query_condition_set: {
      identity: 'qcs-incomplete-completed',
      normalized_question: submittedTurn.question,
      conditions: []
    },
    condition_provenance: { mode: 'question_normalized' }
  };
  const oldHistory = [
    { role: 'user', answer_execution: priorExecution },
    { role: 'assistant', answer_execution: priorExecution }
  ];

  assert.equal(
    findRecoveredClosedExecution(oldHistory, submittedTurn, new Set([priorExecution.id])),
    null
  );
  assert.equal(
    findRecoveredClosedExecution(
      [...oldHistory, { role: 'user', answer_execution: recoveredExecution }],
      submittedTurn,
      new Set([priorExecution.id])
    ),
    null
  );
  assert.equal(
    findRecoveredClosedExecution(
      [...oldHistory, { role: 'user', answer_execution: incompleteCompletedExecution }],
      submittedTurn,
      new Set([priorExecution.id])
    ),
    null
  );
  const recoveredHistory = [
    ...oldHistory,
    {
      role: 'user',
      content: submittedTurn.question,
      answer_execution: recoveredExecution
    },
    {
      id: 'assistant-current',
      role: 'assistant',
      content: '',
      answer_execution: recoveredExecution
    }
  ];
  assert.equal(
    findRecoveredClosedExecution(
      recoveredHistory,
      submittedTurn,
      new Set([priorExecution.id])
    ),
    null
  );
  assert.equal(
    findRecoveredClosedExecution(
      recoveredHistory,
      submittedTurn,
      new Set([priorExecution.id]),
      'assistant-other'
    ),
    null
  );
  assert.equal(
    findRecoveredClosedExecution(
      recoveredHistory,
      submittedTurn,
      new Set([priorExecution.id]),
      'assistant-current'
    )?.id,
    recoveredExecution.id
  );
});

test('validateCompletedStreamProjection requires one coherent frozen terminal contract', () => {
  assert.doesNotThrow(() => validateCompletedStreamProjection(completedStreamMessage()));
  assert.throws(
    () =>
      validateCompletedStreamProjection(
        completedStreamMessage({
          outcome: 'evidence_gated_answer',
          executionOutcome: 'insufficient_evidence_reply',
          evidenceSummary: {
            coverage: 'sufficient',
            source_count: 1,
            sources: [{ snapshot_id: 'snapshot-1' }]
          }
        })
      ),
    /outcome/
  );
  assert.throws(
    () => validateCompletedStreamProjection({ content: 'partial response', outcome: 'evidence_gated_answer' }),
    /execution/
  );
  assert.throws(
    () => validateCompletedStreamProjection(completedStreamMessage({ state: 'stopped' })),
    /completed/
  );
});

test('validateClosedAssistantProjection accepts only a closed completed or terminal execution', () => {
  assert.doesNotThrow(() => validateClosedAssistantProjection(completedStreamMessage()));

  const terminal = {
    id: 'assistant-stopped',
    content: '',
    answer_execution: {
      id: 'answer-execution-stopped',
      assistant_message_id: 'assistant-stopped',
      state: 'stopped',
      question: 'What applies?',
      query_condition_set: {
        identity: 'qcs-stopped',
        normalized_question: 'What applies?',
        conditions: []
      },
      condition_provenance: { mode: 'question_normalized' },
      failure_code: 'ANSWER_EXECUTION_STOPPED'
    }
  };
  assert.doesNotThrow(() => validateClosedAssistantProjection(terminal));

  assert.throws(
    () =>
      validateClosedAssistantProjection({
        id: 'legacy-assistant',
        content: 'A legacy answer has no closed execution.'
      }),
    /closed answer execution/
  );

  assert.throws(
    () =>
      validateClosedAssistantProjection({
        ...terminal,
        content: 'A terminal execution must not retain answer text.'
      }),
    /terminal/
  );

  assert.throws(
    () =>
      validateClosedAssistantProjection({
        ...terminal,
        outcome: 'insufficient_evidence_reply',
        evidence_summary: { coverage: 'insufficient', source_count: 0, sources: [] }
      }),
    /terminal/
  );
});

test('validateCompletedStreamProjection rejects malformed QCS, assistant bindings, and evidence drift', () => {
  const malformedQcs = completedStreamMessage();
  malformedQcs.answer_execution.query_condition_set = {
    identity: 'qcs-1',
    normalized_question: 'A substituted question',
    conditions: [null]
  };
  assert.throws(() => validateCompletedStreamProjection(malformedQcs), /Query Condition Set/);

  const wrongAssistantBinding = completedStreamMessage();
  wrongAssistantBinding.answer_execution.assistant_message_id = 'assistant-substituted';
  assert.throws(() => validateCompletedStreamProjection(wrongAssistantBinding), /assistant message/);

  const frozenSummary = {
    coverage: 'sufficient',
    source_count: 1,
    sources: [
      {
        citation_id: 'S1',
        citation_identity: 'citation-1',
        snapshot_id: 'snapshot-1',
        publication_version: 'v1',
        excerpt: 'The frozen source excerpt.'
      }
    ]
  };
  const evidenceBound = completedStreamMessage({
    outcome: 'evidence_gated_answer',
    evidenceSummary: frozenSummary
  });
  evidenceBound.answer_execution.evidence_set_identity = 'evidence-set-1';
  evidenceBound.answer_execution.item_identities = ['item-1'];
  evidenceBound.answer_execution.snapshot_ids = ['snapshot-1'];
  evidenceBound.answer_execution.knowledge_version_identities = ['knowledge-version-1'];
  evidenceBound.evidence_summary = {
    ...frozenSummary,
    sources: [{ ...frozenSummary.sources[0], excerpt: 'A substituted source excerpt.' }]
  };
  assert.throws(() => validateCompletedStreamProjection(evidenceBound), /evidence summary/);
});

test('validateCompletedStreamProjection binds the terminal execution to the submitted turn', () => {
  assert.doesNotThrow(() =>
    validateCompletedStreamProjection(completedStreamMessage(), {
      question: 'What applies?',
      query_conditions: undefined,
      inherit_conditions: false
    })
  );

  assert.throws(
    () =>
      validateCompletedStreamProjection(completedStreamMessage(), {
        question: 'A different submitted question',
        query_conditions: undefined,
        inherit_conditions: false
      }),
    /submitted question/
  );

  const conditions = [
    {
      condition_id: 'environment-production',
      field: 'environment',
      operator: 'equals',
      value: 'production'
    }
  ];
  const explicitProjection = completedStreamMessage({
    insufficientEvidenceReply: {
      outcome: 'insufficient_evidence_reply',
      reason: 'decision_not_covered',
      query_condition_set_identity: 'qcs-production'
    }
  });
  explicitProjection.answer_execution.query_condition_set = {
    identity: 'qcs-production',
    normalized_question: 'What applies?',
    conditions
  };
  explicitProjection.answer_execution.condition_provenance = { mode: 'explicit' };
  explicitProjection.answer_execution.insufficient_evidence_reply.query_condition_set_identity = 'qcs-production';
  explicitProjection.insufficient_evidence_reply.query_condition_set_identity = 'qcs-production';

  assert.throws(
    () =>
      validateCompletedStreamProjection(explicitProjection, {
        question: 'What applies?',
        query_conditions: [{ ...conditions[0], value: 'staging' }],
        inherit_conditions: false
      }),
    /submitted Query Condition Set/
  );

  const implicitProjection = completedStreamMessage();
  implicitProjection.answer_execution.query_condition_set.conditions = [
    {
      condition_id: 'environment-staging',
      field: 'environment',
      operator: 'equals',
      value: 'staging'
    }
  ];
  assert.throws(
    () =>
      validateCompletedStreamProjection(implicitProjection, {
        question: 'What applies?',
        query_conditions: undefined,
        inherit_conditions: false
      }),
    /submitted Query Condition Set/
  );

  const inheritedProjection = completedStreamMessage();
  inheritedProjection.answer_execution.query_condition_set.conditions = [
    {
      condition_id: 'environment-production',
      field: 'environment',
      operator: 'equals',
      value: 'production'
    }
  ];
  inheritedProjection.answer_execution.condition_provenance = {
    mode: 'inherited',
    source_execution_id: 'foreign-session-execution'
  };
  assert.throws(
    () =>
      validateCompletedStreamProjection(inheritedProjection, {
        question: 'What applies?',
        query_conditions: undefined,
        inherit_conditions: true,
        known_executions: []
      }),
    /inherited Query Condition Set/
  );

  const mixedLanguageProjection = completedStreamMessage();
  const mixedLanguageQuestion = '请在environment=production下说明适用决策';
  mixedLanguageProjection.answer_execution.question = mixedLanguageQuestion;
  mixedLanguageProjection.answer_execution.query_condition_set.normalized_question = mixedLanguageQuestion;
  assert.doesNotThrow(() =>
    validateCompletedStreamProjection(mixedLanguageProjection, {
      question: mixedLanguageQuestion,
      query_conditions: undefined,
      inherit_conditions: false
    })
  );

  for (const [question, field, value] of [
    ['请解释 environment=production.', 'environment', 'production'],
    ['Use release=2026.09.', 'release', '2026.09'],
    ['请按 target=release/v1/执行', 'target', 'release/v1/'],
    ['Use region=İSTANBUL.', 'region', 'i̇stanbul'],
    ['Use mode=ſAFE.', 'mode', 'safe']
  ]) {
    const punctuationProjection = completedStreamMessage();
    const qcsIdentity = `qcs-${field}`;
    const condition = {
      condition_id: `${field}-${value}`,
      field,
      operator: 'equals',
      value
    };
    punctuationProjection.answer_execution.question = question;
    punctuationProjection.answer_execution.query_condition_set = {
      identity: qcsIdentity,
      normalized_question: question,
      conditions: [condition]
    };
    punctuationProjection.answer_execution.insufficient_evidence_reply.query_condition_set_identity = qcsIdentity;
    punctuationProjection.insufficient_evidence_reply.query_condition_set_identity = qcsIdentity;
    assert.doesNotThrow(() =>
      validateCompletedStreamProjection(punctuationProjection, {
        question,
        query_conditions: undefined,
        inherit_conditions: false
      })
    );
  }
});

test('validateCompletedStreamProjection matches Python word-boundary normalization for punctuation-only values', () => {
  const message = completedStreamMessage();
  message.answer_execution.question = 'Can I use path=/?';
  message.answer_execution.query_condition_set.normalized_question = 'Can I use path=/?';

  assert.doesNotThrow(() =>
    validateCompletedStreamProjection(message, {
      question: 'Can I use path=/?',
      query_conditions: undefined,
      inherit_conditions: false
    })
  );
});

test('validateCompletedStreamProjection requires the frozen structured insufficiency reply', () => {
  const missingReply = completedStreamMessage();
  delete missingReply.insufficient_evidence_reply;
  delete missingReply.answer_execution.insufficient_evidence_reply;
  assert.throws(() => validateCompletedStreamProjection(missingReply), /structured insufficiency reply/);

  const contradictoryReply = completedStreamMessage();
  contradictoryReply.insufficient_evidence_reply.reason = 'knowledge_needs_review';
  assert.throws(() => validateCompletedStreamProjection(contradictoryReply), /structured insufficiency reply/);

  const inventedReason = completedStreamMessage();
  inventedReason.insufficient_evidence_reply.reason = 'provider_timeout';
  inventedReason.answer_execution.insufficient_evidence_reply.reason = 'provider_timeout';
  assert.throws(() => validateCompletedStreamProjection(inventedReason), /structured insufficiency reply/);
});

test('resolveRetryTurn uses the selected user turn and never re-inherits after admission', () => {
  const frozenConditions = [
    {
      condition_id: 'environment-production',
      field: 'environment',
      operator: 'equals',
      value: 'production'
    }
  ];
  const failedUserTurn = {
    role: 'user',
    content: 'Which reviewed operating decision applies?',
    answer_execution: {
      state: 'failed',
      query_condition_set: {
        identity: 'qcs-production',
        normalized_question: 'Which reviewed operating decision applies?',
        conditions: frozenConditions
      }
    },
    requested_inherit_conditions: true
  };
  const failedAssistantTurn = {
    role: 'assistant',
    content: '请求失败：execution failed',
    failed: true
  };

  assert.deepEqual(resolveRetryTurn([failedUserTurn], 0), {
    question: failedUserTurn.content,
    query_conditions: frozenConditions,
    inherit_conditions: false
  });
  assert.deepEqual(resolveRetryTurn([failedUserTurn, failedAssistantTurn], 1), {
    question: failedUserTurn.content,
    query_conditions: frozenConditions,
    inherit_conditions: false
  });

  const stagingConditions = [
    {
      condition_id: 'environment-staging',
      field: 'environment',
      operator: 'equals',
      value: 'staging'
    }
  ];
  const firstUserTurn = {
    role: 'user',
    content: 'Which reviewed production decision applies?',
    answer_execution: {
      id: 'answer-execution-production',
      state: 'completed',
      question: 'Which reviewed production decision applies?',
      query_condition_set: {
        identity: 'qcs-production',
        normalized_question: 'Which reviewed production decision applies?',
        conditions: frozenConditions
      }
    }
  };
  const secondUserTurn = {
    role: 'user',
    content: 'Which reviewed staging decision applies?',
    answer_execution: {
      id: 'answer-execution-staging',
      state: 'completed',
      question: 'Which reviewed staging decision applies?',
      query_condition_set: {
        identity: 'qcs-staging',
        normalized_question: 'Which reviewed staging decision applies?',
        conditions: stagingConditions
      }
    }
  };
  const secondAssistantTurn = {
    role: 'assistant',
    content: 'staging answer',
    answer_execution: {
      ...secondUserTurn.answer_execution,
      question: secondUserTurn.content
    }
  };
  const firstAssistantTurn = {
    role: 'assistant',
    content: 'production answer',
    answer_execution: {
      ...firstUserTurn.answer_execution,
      question: firstUserTurn.content
    }
  };
  assert.deepEqual(
    resolveRetryTurn(
      [firstUserTurn, secondUserTurn, secondAssistantTurn, firstAssistantTurn],
      3
    ),
    {
      question: firstUserTurn.content,
      query_conditions: frozenConditions,
      inherit_conditions: false
    }
  );
  assert.throws(
    () =>
      resolveRetryTurn(
        [
          firstUserTurn,
          {
            ...firstAssistantTurn,
            answer_execution: {
              ...secondUserTurn.answer_execution,
              id: firstUserTurn.answer_execution.id,
              question: secondUserTurn.content
            }
          }
        ],
        1
      ),
    /retry source answer execution binding/
  );

  assert.throws(
    () =>
      resolveRetryTurn(
        [
          {
            role: 'user',
            content: 'An admitted failure must not inherit again.',
            answer_execution: { state: 'failed' },
            requested_inherit_conditions: true
          }
        ],
        0
      ),
    /frozen Query Condition Set/
  );
  assert.throws(
    () =>
      resolveRetryTurn(
        [
          {
            role: 'user',
            content: 'An interrupted inherited request cannot ask the current session again.',
            answer_execution: null,
            requested_inherit_conditions: true
          }
        ],
        0
      ),
    /cannot re-inherit/
  );
  assert.throws(
    () =>
      resolveRetryTurn(
        [
          {
            role: 'user',
            content: 'A contradictory stream cannot retain retry authority.',
            answer_execution: null,
            requested_query_conditions: frozenConditions,
            retry_authority_invalid: true
          }
        ],
        0
      ),
    /recovered admitted execution/
  );
});

test('getProviderStatus returns direct provider label', () => {
  const status = getProviderStatus({ runtime: { final_provider: 'ark', fallback_hops: 0 } });
  assert.equal(status, '模型提供方：ark');
});

test('getProviderStatus returns fallback label with hops', () => {
  const status = getProviderStatus({ runtime: { final_provider: 'openai', fallback_hops: 1 } });
  assert.equal(status, '已切换到 openai（1 次回退）');
});

test('getProviderStatus returns empty when provider missing', () => {
  assert.equal(getProviderStatus({ runtime: {} }), '');
  assert.equal(getProviderStatus(null), '');
});
