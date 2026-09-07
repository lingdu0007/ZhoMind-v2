export const extractRejectReason = (step) => {
  if (!step || typeof step !== 'object') return '';
  if (step.step !== 'retrieve') return '';

  const detail = step.detail || {};
  const gateReason = detail.gate_reason || '';
  if (detail.gate_passed === false && String(gateReason).includes('reject')) {
    return String(gateReason);
  }
  return '';
};

export const formatStreamError = (error) => {
  if (!error) return '请求失败，请稍后重试';
  if (typeof error === 'string') return error;

  const code = error.code || '';
  if (code === 'AUTH_FORBIDDEN') return '无权限执行当前问答';
  if (code === 'AUTH_INVALID_TOKEN' || error.status === 401) return '登录状态已失效，请重新登录';

  return error.message || '请求失败，请稍后重试';
};

export const getDoneStatus = (assistantMsg) => {
  if (assistantMsg?.outcome === 'insufficient_evidence_reply') return '证据不足';
  if (assistantMsg?.outcome === 'generation_unavailable') return '生成不可用';
  return '已完成';
};

const completedOutcomes = new Set([
  'evidence_gated_answer',
  'insufficient_evidence_reply',
  'non_knowledge_base_reply',
  'generation_unavailable'
]);

const evidenceBoundOutcomes = new Set([
  'evidence_gated_answer',
  'generation_unavailable'
]);

const answerExecutionStates = new Set([
  'admitted',
  'queued',
  'running',
  'stopped',
  'failed',
  'throttled',
  'rejected',
  'completed'
]);

const terminalExecutionStates = new Set(['stopped', 'failed', 'throttled', 'rejected']);

const insufficientEvidenceReasons = new Set([
  'no_eligible_published_evidence',
  'decision_not_covered',
  'decisive_condition_missing',
  'material_evidence_conflict',
  'assurance_support_missing',
  'evidence_budget_exceeded',
  'knowledge_needs_review'
]);
const queryConditionLimits = {
  maxConditions: 32,
  maxConditionIdChars: 160,
  maxConditionFieldChars: 160,
  maxConditionOperatorChars: 64,
  maxConditionValueChars: 512
};

const explicitQuestionCondition =
  /(?<![\p{L}\p{N}_])([A-Za-zİıſK][A-Za-z0-9_.İıſK-]{0,79})\s*=\s*([A-Za-z0-9_.:/İıſK-]{1,160})/gu;
const unicodeWordCharacter = /[\p{L}\p{N}_]/u;
const pythonConditionCasefold = (value) =>
  value.replaceAll('ſ', 's').replaceAll('K', 'k').toLowerCase();

const isRecord = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);

const requireText = (value, label) => {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`${label} is missing from the frozen answer execution`);
  }
  return value;
};

const requireCanonicalText = (value, label) => {
  const text = requireText(value, label);
  if (text !== text.trim()) {
    throw new Error(`${label} is not canonical in the frozen answer execution`);
  }
  return text;
};

const cloneConditions = (conditions) =>
  conditions.map((condition) => ({
    condition_id: condition.condition_id,
    field: condition.field,
    operator: condition.operator,
    value: condition.value
  }));

const codePointLength = (value) => Array.from(value).length;

const requireConditionRecords = (conditions, label) => {
  if (!Array.isArray(conditions)) {
    throw new Error(`${label} has no condition list`);
  }
  if (conditions.length > queryConditionLimits.maxConditions) {
    throw new Error(`${label} exceeds the condition limit`);
  }
  const conditionIds = new Set();
  const conditionKeys = new Set();
  return conditions.map((condition) => {
    if (!isRecord(condition)) {
      throw new Error(`${label} contains a malformed condition`);
    }
    const normalized = {
      condition_id: requireCanonicalText(condition.condition_id, `${label} condition identity`),
      field: requireCanonicalText(condition.field, `${label} condition field`),
      operator: requireCanonicalText(condition.operator, `${label} condition operator`),
      value: requireCanonicalText(condition.value, `${label} condition value`)
    };
    if (
      codePointLength(normalized.condition_id) > queryConditionLimits.maxConditionIdChars ||
      codePointLength(normalized.field) > queryConditionLimits.maxConditionFieldChars ||
      codePointLength(normalized.operator) > queryConditionLimits.maxConditionOperatorChars ||
      codePointLength(normalized.value) > queryConditionLimits.maxConditionValueChars
    ) {
      throw new Error(`${label} exceeds the condition limit`);
    }
    const conditionKey = `${normalized.field}\u0000${normalized.operator}`;
    if (conditionIds.has(normalized.condition_id) || conditionKeys.has(conditionKey)) {
      throw new Error(`${label} contains duplicate conditions`);
    }
    conditionIds.add(normalized.condition_id);
    conditionKeys.add(conditionKey);
    return normalized;
  });
};

const requireFrozenQueryConditions = (qcs, question) => {
  if (!isRecord(qcs)) {
    throw new Error('answer execution has no frozen Query Condition Set');
  }
  requireCanonicalText(qcs.identity, 'Query Condition Set identity');
  if (requireCanonicalText(qcs.normalized_question, 'Query Condition Set normalized question') !== question) {
    throw new Error('Query Condition Set contradicts the frozen answer execution question');
  }
  return requireConditionRecords(qcs.conditions, 'Query Condition Set');
};

const valueAtPythonWordBoundary = (value, followingCharacter) => {
  let candidate = value;
  let nextCharacter = followingCharacter;
  while (candidate) {
    const finalCharacter = candidate.at(-1);
    if (unicodeWordCharacter.test(finalCharacter) !== unicodeWordCharacter.test(nextCharacter || '')) {
      return candidate;
    }
    nextCharacter = finalCharacter;
    candidate = candidate.slice(0, -1);
  }
  return '';
};

const conditionsNormalizedFromQuestion = (question) =>
  requireConditionRecords(
    Array.from(question.matchAll(explicitQuestionCondition), (match) => {
      const [, field, rawValue] = match;
      const normalizedField = pythonConditionCasefold(field);
      const followingCharacter = question.at((match.index || 0) + match[0].length);
      const normalizedValue = pythonConditionCasefold(
        valueAtPythonWordBoundary(rawValue, followingCharacter)
      );
      if (!normalizedValue) return null;
      return {
        condition_id: `${normalizedField}-${normalizedValue}`,
        field: normalizedField,
        operator: 'equals',
        value: normalizedValue
      };
    }).filter((condition) => condition !== null),
    'submitted question Query Condition Set'
  );

const requireIdentityList = (value, label) => {
  if (!Array.isArray(value) || value.some((item) => typeof item !== 'string' || !item)) {
    throw new Error(`frozen ${label} identities are malformed`);
  }
  return value;
};

const requireSummary = (value) => {
  if (
    !isRecord(value) ||
    typeof value.coverage !== 'string' ||
    !Number.isInteger(value.source_count) ||
    value.source_count < 0 ||
    !Array.isArray(value.sources) ||
    value.source_count !== value.sources.length
  ) {
    throw new Error('frozen evidence summary is malformed');
  }
  return value;
};

const requireNoEvidenceProjection = (execution, itemIds, snapshotIds, knowledgeVersionIds) => {
  if (
    execution.evidence_set_identity !== null ||
    itemIds.length !== 0 ||
    snapshotIds.length !== 0 ||
    knowledgeVersionIds.length !== 0
  ) {
    throw new Error('non-evidence completed outcome retained frozen evidence');
  }
};

const requireEvidenceProjection = (execution, itemIds, snapshotIds, knowledgeVersionIds) => {
  requireText(execution.evidence_set_identity, 'evidence set identity');
  if (
    itemIds.length === 0 ||
    snapshotIds.length === 0 ||
    knowledgeVersionIds.length === 0 ||
    itemIds.length !== snapshotIds.length
  ) {
    throw new Error('evidence-bound completed outcome has incomplete frozen identities');
  }
};

const rejectEvidencePreview = (summary) => {
  if (summary.evidence_preview !== undefined && summary.evidence_preview !== null) {
    throw new Error('completed outcome has an unexpected evidence preview');
  }
};

const canonicalFrozenValue = (value) => {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (Array.isArray(value)) return value.map(canonicalFrozenValue);
  if (isRecord(value)) {
    return Object.keys(value)
      .sort()
      .reduce((normalized, key) => {
        normalized[key] = canonicalFrozenValue(value[key]);
        return normalized;
      }, {});
  }
  throw new Error('frozen execution projection contains an unsupported value');
};

const sameFrozenProjection = (left, right) =>
  JSON.stringify(canonicalFrozenValue(left)) === JSON.stringify(canonicalFrozenValue(right));

const requireStructuredInsufficiencyReply = (value, queryConditionSetIdentity, label) => {
  if (!isRecord(value) || Object.keys(value).length !== 3 || value.outcome !== 'insufficient_evidence_reply') {
    throw new Error(`${label} is malformed`);
  }
  const reason = requireCanonicalText(value.reason, `${label} reason`);
  const identity = requireCanonicalText(value.query_condition_set_identity, `${label} Query Condition Set identity`);
  if (!insufficientEvidenceReasons.has(reason) || identity !== queryConditionSetIdentity) {
    throw new Error(`${label} is malformed`);
  }
  return {
    outcome: value.outcome,
    reason,
    query_condition_set_identity: identity
  };
};

const requireInheritedSourceConditions = (expectedTurn, sourceExecutionId) => {
  if (!Array.isArray(expectedTurn.known_executions)) {
    throw new Error('submitted inherited Query Condition Set has no private source execution');
  }
  const sourceExecution = expectedTurn.known_executions.find(
    (candidate) =>
      isRecord(candidate) &&
      candidate.id === sourceExecutionId &&
      candidate.state === 'completed'
  );
  if (!sourceExecution) {
    throw new Error('frozen answer execution contradicts the submitted inherited Query Condition Set');
  }
  const sourceQuestion = requireCanonicalText(
    sourceExecution.question,
    'inherited source answer execution question'
  );
  return requireFrozenQueryConditions(
    sourceExecution.query_condition_set,
    sourceQuestion
  );
};

const requireSubmittedTurnBinding = (expectedTurn, { question, conditions, provenance }) => {
  if (expectedTurn === undefined) return;
  if (!isRecord(expectedTurn)) {
    throw new Error('submitted turn binding is malformed');
  }
  if (requireCanonicalText(expectedTurn.question, 'submitted question') !== question) {
    throw new Error('frozen answer execution contradicts the submitted question');
  }
  const inheritConditions = expectedTurn.inherit_conditions === true;
  const submittedConditions = expectedTurn.query_conditions;
  if (inheritConditions) {
    if (submittedConditions !== undefined || provenance.mode !== 'inherited') {
      throw new Error('frozen answer execution contradicts the submitted Query Condition Set provenance');
    }
    const inheritedConditions = requireInheritedSourceConditions(
      expectedTurn,
      provenance.source_execution_id
    );
    if (!sameFrozenProjection(inheritedConditions, conditions)) {
      throw new Error('frozen answer execution contradicts the submitted inherited Query Condition Set');
    }
    return;
  }
  if (submittedConditions === undefined) {
    if (
      provenance.mode !== 'question_normalized' ||
      !sameFrozenProjection(conditionsNormalizedFromQuestion(question), conditions)
    ) {
      throw new Error('frozen answer execution contradicts the submitted Query Condition Set provenance');
    }
    return;
  }
  const normalizedSubmittedConditions = requireConditionRecords(
    submittedConditions,
    'submitted Query Condition Set'
  );
  if (
    provenance.mode !== 'explicit' ||
    !sameFrozenProjection(normalizedSubmittedConditions, conditions)
  ) {
    throw new Error('frozen answer execution contradicts the submitted Query Condition Set');
  }
};

export const validateExecutionTurnBinding = (execution, expectedTurn = undefined) => {
  if (!isRecord(execution)) {
    throw new Error('answer execution is malformed');
  }
  requireText(execution.id, 'answer execution identity');
  if (!answerExecutionStates.has(execution.state)) {
    throw new Error('answer execution state is malformed');
  }
  const question = requireCanonicalText(execution.question, 'answer execution question');
  const conditions = requireFrozenQueryConditions(execution.query_condition_set, question);
  const provenance = execution.condition_provenance;
  if (
    !isRecord(provenance) ||
    !['explicit', 'inherited', 'question_normalized'].includes(provenance.mode)
  ) {
    throw new Error('answer execution condition provenance is malformed');
  }
  if (
    provenance.mode === 'inherited' &&
    (typeof provenance.source_execution_id !== 'string' || !provenance.source_execution_id)
  ) {
    throw new Error('inherited answer execution provenance is incomplete');
  }
  if (
    provenance.mode !== 'inherited' &&
    Object.keys(provenance).length !== 1
  ) {
    throw new Error('answer execution condition provenance is malformed');
  }
  requireSubmittedTurnBinding(expectedTurn, { question, conditions, provenance });
  return { question, conditions, provenance };
};

const terminalExecutionHasCompletedFields = (execution) =>
  Boolean(
    execution.outcome !== undefined ||
      execution.answer_text !== undefined ||
      execution.evidence_set_identity !== undefined ||
      execution.item_identities !== undefined ||
      execution.snapshot_ids !== undefined ||
      execution.knowledge_version_identities !== undefined ||
      execution.evidence_summary !== undefined ||
      execution.insufficient_evidence_reply !== undefined
  );

export const isClosedAnswerExecution = (execution) =>
  execution?.state === 'completed' || terminalExecutionStates.has(execution?.state);

const validateTerminalFailureCode = (execution, context) => {
  if (typeof execution.failure_code !== 'string' || !execution.failure_code.trim()) {
    throw new Error(`${context} has no failure code`);
  }
};

export const validateUserExecutionProjection = (execution, expectedTurn = undefined) => {
  validateExecutionTurnBinding(execution, expectedTurn);
  if (execution.state === 'completed') {
    if (typeof execution.assistant_message_id !== 'string' || !execution.assistant_message_id) {
      throw new Error('completed user answer execution has no assistant message binding');
    }
    return { state: execution.state };
  }
  if (!terminalExecutionStates.has(execution.state)) return { state: execution.state };
  if (terminalExecutionHasCompletedFields(execution)) {
    throw new Error('terminal user answer execution retained a completed outcome');
  }
  validateTerminalFailureCode(execution, 'terminal user answer execution');
  const assistantMessageId = execution.assistant_message_id;
  if (assistantMessageId === undefined || assistantMessageId === null) {
    if (
      execution.state !== 'failed' ||
      execution.failure_code !== 'ANSWER_EXECUTION_PERSISTENCE_FAILED'
    ) {
      throw new Error('terminal user answer execution has no assistant message binding');
    }
    return { state: execution.state };
  }
  if (typeof assistantMessageId !== 'string' || !assistantMessageId) {
    throw new Error('terminal user answer execution assistant message binding is malformed');
  }
  return { state: execution.state };
};

export const resolveRetryTurn = (messages, messageIndex) => {
  if (!Array.isArray(messages) || !Number.isInteger(messageIndex)) return null;
  const selectedMessage = messages[messageIndex];
  let userMessage = selectedMessage?.role === 'user' ? selectedMessage : null;
  let boundExecution = null;
  if (!userMessage && selectedMessage?.answer_execution !== undefined && selectedMessage?.answer_execution !== null) {
    const selectedExecution = selectedMessage.answer_execution;
    if (
      !isRecord(selectedExecution) ||
      typeof selectedExecution.id !== 'string' ||
      !selectedExecution.id
    ) {
      throw new Error('retry source answer execution binding is malformed');
    }
    const sourceTurns = messages.filter(
      (message) =>
        message?.role === 'user' &&
        isRecord(message.answer_execution) &&
        message.answer_execution.id === selectedExecution.id
    );
    if (sourceTurns.length !== 1) {
      throw new Error('retry source answer execution binding is unavailable');
    }
    [userMessage] = sourceTurns;
    boundExecution = selectedExecution;
  }
  if (!userMessage) {
    userMessage = messages
      .slice(0, messageIndex)
      .reverse()
      .find((message) => message?.role === 'user');
  }
  if (!userMessage || typeof userMessage.content !== 'string' || !userMessage.content.trim()) {
    return null;
  }
  if (userMessage.retry_authority_invalid === true) {
    throw new Error('retry requires a recovered admitted execution');
  }

  if (boundExecution !== null) {
    const sourceExecution = userMessage.answer_execution;
    const sourceQuestion = requireCanonicalText(
      sourceExecution.question,
      'retry source user answer execution question'
    );
    const boundQuestion = requireCanonicalText(
      boundExecution.question,
      'retry selected answer execution question'
    );
    const sourceConditions = requireFrozenQueryConditions(
      sourceExecution.query_condition_set,
      sourceQuestion
    );
    const boundConditions = requireFrozenQueryConditions(
      boundExecution.query_condition_set,
      boundQuestion
    );
    if (
      sourceQuestion !== userMessage.content ||
      sourceQuestion !== boundQuestion ||
      requireCanonicalText(
        sourceExecution.query_condition_set.identity,
        'retry source user Query Condition Set identity'
      ) !==
        requireCanonicalText(
          boundExecution.query_condition_set.identity,
          'retry selected Query Condition Set identity'
        ) ||
      !sameFrozenProjection(sourceConditions, boundConditions)
    ) {
      throw new Error('retry source answer execution binding is contradictory');
    }
    return {
      question: sourceQuestion,
      query_conditions: cloneConditions(boundConditions),
      inherit_conditions: false
    };
  }

  if (userMessage.answer_execution !== null && userMessage.answer_execution !== undefined) {
    if (!isRecord(userMessage.answer_execution)) {
      throw new Error('admitted answer execution is malformed');
    }
    return {
      question: userMessage.content,
      query_conditions: cloneConditions(
        requireFrozenQueryConditions(
          userMessage.answer_execution.query_condition_set,
          userMessage.content
        )
      ),
      inherit_conditions: false
    };
  }

  const requestedConditions = userMessage.requested_query_conditions;
  if (requestedConditions !== undefined && !Array.isArray(requestedConditions)) {
    throw new Error('pre-admission retry conditions are malformed');
  }
  if (userMessage.requested_inherit_conditions === true) {
    throw new Error('retry cannot re-inherit conditions before the admitted execution is recovered');
  }
  return {
    question: userMessage.content,
    query_conditions: Array.isArray(requestedConditions)
      ? requestedConditions.map((condition) => ({
          condition_id: condition?.condition_id || '',
          field: condition?.field || '',
          operator: condition?.operator || '',
          value: condition?.value || ''
        }))
      : undefined,
    inherit_conditions: false
  };
};

export const validateCompletedStreamProjection = (message, expectedTurn = undefined) => {
  if (!isRecord(message)) {
    throw new Error('completed stream projection is malformed');
  }
  requireText(message.id, 'assistant message identity');
  requireText(message.content, 'assistant message text');

  const execution = message.answer_execution;
  if (!isRecord(execution)) {
    throw new Error('completed stream projection has no answer execution');
  }
  if (execution.state !== 'completed') {
    throw new Error('answer execution is not completed');
  }
  requireText(execution.id, 'answer execution identity');
  if (requireText(execution.assistant_message_id, 'answer execution assistant message identity') !== message.id) {
    throw new Error('answer execution assistant message identity contradicts the stream message');
  }
  const { question, conditions: frozenConditions } = validateExecutionTurnBinding(execution, expectedTurn);
  if (requireText(execution.answer_text, 'answer execution answer text') !== message.content) {
    throw new Error('answer execution text contradicts the stream message');
  }

  if (!completedOutcomes.has(message.outcome)) {
    throw new Error('completed stream projection has no accepted outcome');
  }
  if (execution.outcome !== message.outcome) {
    throw new Error('completed stream outcome contradicts the frozen answer execution');
  }

  const itemIds = requireIdentityList(execution.item_identities, 'evidence item');
  const snapshotIds = requireIdentityList(execution.snapshot_ids, 'snapshot');
  const knowledgeVersionIds = requireIdentityList(
    execution.knowledge_version_identities,
    'knowledge version'
  );
  const summary = requireSummary(message.evidence_summary);
  const frozenSummary = requireSummary(execution.evidence_summary);
  if (!sameFrozenProjection(summary, frozenSummary)) {
    throw new Error('stream evidence summary contradicts the frozen answer execution');
  }

  if (message.outcome === 'insufficient_evidence_reply') {
    const frozenReply = requireStructuredInsufficiencyReply(
      execution.insufficient_evidence_reply,
      execution.query_condition_set.identity,
      'frozen structured insufficiency reply'
    );
    const streamedReply = requireStructuredInsufficiencyReply(
      message.insufficient_evidence_reply,
      execution.query_condition_set.identity,
      'streamed structured insufficiency reply'
    );
    if (!sameFrozenProjection(frozenReply, streamedReply)) {
      throw new Error('streamed structured insufficiency reply contradicts the frozen answer execution');
    }
  } else if (
    execution.insufficient_evidence_reply !== null &&
    execution.insufficient_evidence_reply !== undefined
  ) {
    throw new Error('non-insufficient completed outcome retained a structured insufficiency reply');
  } else if (
    message.insufficient_evidence_reply !== null &&
    message.insufficient_evidence_reply !== undefined
  ) {
    throw new Error('streamed non-insufficient completed outcome retained a structured insufficiency reply');
  }

  if (!evidenceBoundOutcomes.has(message.outcome)) {
    requireNoEvidenceProjection(execution, itemIds, snapshotIds, knowledgeVersionIds);
    if (
      summary.source_count !== 0 ||
      summary.sources.length !== 0 ||
      (message.outcome === 'insufficient_evidence_reply' && summary.coverage !== 'insufficient') ||
      (message.outcome === 'non_knowledge_base_reply' && summary.coverage !== 'unavailable')
    ) {
      throw new Error('non-evidence completed outcome contradicts its frozen evidence summary');
    }
    rejectEvidencePreview(summary);
    return;
  }

  requireEvidenceProjection(execution, itemIds, snapshotIds, knowledgeVersionIds);
  if (message.outcome === 'generation_unavailable') {
    if (summary.coverage !== 'unavailable' || summary.source_count !== 0 || summary.sources.length !== 0) {
      throw new Error('generation-unavailable outcome exposed evidence instead of a closed failure');
    }
    rejectEvidencePreview(summary);
    return;
  }

  if (
    summary.coverage !== 'sufficient' ||
    summary.source_count !== snapshotIds.length ||
    summary.sources.some(
      (source, index) => !isRecord(source) || source.snapshot_id !== snapshotIds[index]
    )
  ) {
    throw new Error('evidence-gated outcome contradicts the frozen evidence snapshots');
  }
  rejectEvidencePreview(summary);
};

const terminalMessageHasCompletedFields = (message, execution) =>
  Boolean(
    terminalExecutionHasCompletedFields(execution) ||
      (message.outcome !== undefined && message.outcome !== null && message.outcome !== '') ||
      (message.evidence_summary !== undefined && message.evidence_summary !== null) ||
      (message.insufficient_evidence_reply !== undefined && message.insufficient_evidence_reply !== null)
  );

export const validateClosedAssistantProjection = (message, expectedTurn = undefined) => {
  if (!isRecord(message)) {
    throw new Error('assistant message is malformed');
  }
  const execution = message.answer_execution;
  if (!isRecord(execution)) {
    throw new Error('assistant message has no closed answer execution');
  }

  if (execution.state === 'completed') {
    validateCompletedStreamProjection(message, expectedTurn);
    return { state: 'completed' };
  }

  if (!terminalExecutionStates.has(execution.state)) {
    throw new Error('assistant message has no closed terminal answer execution');
  }
  validateExecutionTurnBinding(execution, expectedTurn);
  if (terminalMessageHasCompletedFields(message, execution)) {
    throw new Error('terminal answer execution retained a completed outcome');
  }
  if (message.content !== '') {
    throw new Error('terminal answer execution retained answer text');
  }
  validateTerminalFailureCode(execution, 'terminal answer execution');

  const assistantMessageId = execution.assistant_message_id;
  if (assistantMessageId === undefined || assistantMessageId === null) {
    if (
      execution.state !== 'failed' ||
      execution.failure_code !== 'ANSWER_EXECUTION_PERSISTENCE_FAILED'
    ) {
      throw new Error('terminal answer execution has no assistant message binding');
    }
    return { state: execution.state };
  }
  if (
    typeof assistantMessageId !== 'string' ||
    !assistantMessageId ||
    typeof message.id !== 'string' ||
    !message.id ||
    assistantMessageId !== message.id
  ) {
    throw new Error('terminal answer execution assistant message binding is malformed');
  }
  return { state: execution.state };
};

export const validateBoundUserExecutionProjection = (execution, messages, expectedTurn = undefined) => {
  const projection = validateUserExecutionProjection(execution, expectedTurn);
  const assistantMessageId = execution.assistant_message_id;
  if (assistantMessageId === undefined || assistantMessageId === null) {
    return projection;
  }
  if (!Array.isArray(messages)) {
    throw new Error('bound user answer execution has no projected assistant messages');
  }
  const matchingAssistants = messages.filter(
    (message) => message?.role === 'assistant' && message?.id === assistantMessageId
  );
  if (
    matchingAssistants.length !== 1 ||
    matchingAssistants[0]?.answer_execution?.id !== execution.id
  ) {
    throw new Error('bound user answer execution has no matching assistant projection');
  }
  validateClosedAssistantProjection(matchingAssistants[0], expectedTurn);
  if (!sameFrozenProjection(matchingAssistants[0].answer_execution, execution)) {
    throw new Error('bound user and assistant answer executions disagree');
  }
  return projection;
};

export const validateHistoryUserExecutionProjection = (
  message,
  messages,
  expectedTurn = undefined
) => {
  if (!isRecord(message) || message.role !== 'user') {
    throw new Error('stored user message is malformed');
  }
  const storedQuestion = requireCanonicalText(message.content, 'stored user message');
  const executionQuestion = requireCanonicalText(
    message.answer_execution?.question,
    'answer execution question'
  );
  if (storedQuestion !== executionQuestion) {
    throw new Error('stored user message contradicts the frozen answer execution question');
  }
  return validateBoundUserExecutionProjection(message.answer_execution, messages, expectedTurn);
};

export const findRecoveredClosedExecution = (
  messages,
  expectedTurn,
  knownExecutionIds = new Set(),
  expectedAssistantMessageId = ''
) => {
  if (
    !Array.isArray(messages) ||
    typeof expectedAssistantMessageId !== 'string' ||
    !expectedAssistantMessageId ||
    expectedAssistantMessageId !== expectedAssistantMessageId.trim()
  ) {
    return null;
  }
  const known =
    knownExecutionIds instanceof Set
      ? knownExecutionIds
      : new Set(Array.isArray(knownExecutionIds) ? knownExecutionIds : []);

  for (const message of [...messages].reverse()) {
    const execution = message?.role === 'user' ? message.answer_execution : null;
    if (
      !isClosedAnswerExecution(execution) ||
      typeof execution.id !== 'string' ||
      !execution.id ||
      execution.assistant_message_id !== expectedAssistantMessageId ||
      known.has(execution.id)
    ) {
      continue;
    }
    try {
      validateHistoryUserExecutionProjection(message, messages, expectedTurn);
      return execution;
    } catch {
      // A recovered history item must independently satisfy the submitted-turn contract.
    }
  }
  return null;
};

export const getProviderStatus = (trace) => {
  const runtime = trace?.runtime || {};
  const provider = runtime.final_provider;
  const hops = Number(runtime.fallback_hops || 0);
  if (!provider) return '';
  if (hops > 0) return `已切换到 ${provider}（${hops} 次回退）`;
  return `模型提供方：${provider}`;
};
