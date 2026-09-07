import test from 'node:test';
import assert from 'node:assert/strict';
import {
  extractRejectReason,
  formatStreamError,
  getDoneStatus,
  getProviderStatus,
  resolveRetryTurn,
  validateCompletedStreamProjection
} from '../src/store/chat-state.js';

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
