import test from 'node:test';
import assert from 'node:assert/strict';
import { createSSEParser, normalizeSSEFrame } from '../src/api/sse.js';

const collectEvents = (chunks) => {
  const frames = [];
  const parser = createSSEParser((frame) => {
    frames.push(normalizeSSEFrame(frame));
  });
  chunks.forEach((chunk) => parser.feed(chunk));
  parser.finish();
  return frames;
};

test('parse standard SSE event/data pairs without leaking event lines into content', () => {
  const frames = collectEvents([
    'event: meta\n',
    'data: {"request_id":"req_1"}\n\n',
    'event: rag_step\n',
    'data: {"step":"retrieve"}\n\n',
    'event: content\n',
    'data: {"delta":"你"}\n\n',
    'event: content\ndata: {"delta":"好"}\n\n',
    'event: evidence_summary\ndata: {"evidence_summary":{"coverage":"sufficient","source_count":1,"sources":[]}}\n\n',
    'event: trace\ndata: {"trace":{"k":"v"}}\n\n',
    'event: done\ndata: [DONE]\n\n'
  ]);

  const types = frames.map((item) => item.type);
  assert.deepEqual(types, ['unknown', 'rag_step', 'content', 'content', 'evidence_summary', 'trace', 'done']);

  const contentChunks = frames.filter((item) => item.type === 'content').map((item) => item.content);
  assert.deepEqual(contentChunks, ['你', '好']);
  contentChunks.forEach((chunk) => {
    assert.equal(chunk.includes('event:'), false);
  });
});

test('parse an Evidence Summary frame as structured answer support', () => {
  const [event] = collectEvents([
    'event: evidence_summary\n',
    'data: {"evidence_summary":{"coverage":"unavailable","source_count":0,"sources":[]}}\n\n'
  ]);

  assert.deepEqual(event, {
    type: 'evidence_summary',
    evidence_summary: { coverage: 'unavailable', source_count: 0, sources: [] }
  });
});

test('parse an answer identity without treating it as generated content', () => {
  const events = collectEvents(['event: answer_identity\ndata: {"answer_id":"answer-123"}\n\n']);

  assert.deepEqual(events, [{ type: 'answer_identity', answer_id: 'answer-123' }]);
});

test('parse a frozen answer execution and explicit completed outcome without inference', () => {
  const events = collectEvents([
    'event: answer_execution\ndata: {"answer_execution":{"id":"answer_execution:1","state":"completed","query_condition_set":{"identity":"qcs:1","conditions":[]}}}\n\n',
    'event: outcome\ndata: {"outcome":"insufficient_evidence_reply"}\n\n'
  ]);

  assert.deepEqual(events, [
    {
      type: 'answer_execution',
      answer_execution: {
        id: 'answer_execution:1',
        state: 'completed',
        query_condition_set: { identity: 'qcs:1', conditions: [] }
      }
    },
    { type: 'outcome', outcome: 'insufficient_evidence_reply' }
  ]);
});

test('parse a structured insufficiency reply as an explicit terminal field', () => {
  const [event] = collectEvents([
    'event: insufficient_evidence_reply\n',
    'data: {"insufficient_evidence_reply":{"outcome":"insufficient_evidence_reply","reason":"decision_not_covered","query_condition_set_identity":"qcs:1"}}\n\n'
  ]);

  assert.deepEqual(event, {
    type: 'insufficient_evidence_reply',
    insufficient_evidence_reply: {
      outcome: 'insufficient_evidence_reply',
      reason: 'decision_not_covered',
      query_condition_set_identity: 'qcs:1'
    }
  });
});

test('parse a role-scoped Retrieval Diagnostics frame as structured administrator data', () => {
  const [event] = collectEvents([
    'event: retrieval_diagnostics\n',
    'data: {"retrieval_diagnostics":{"timeline":[{"step":"retrieve"}],"candidate_counts":{"retrieved":2,"reranked":1}}}\n\n'
  ]);

  assert.deepEqual(event, {
    type: 'retrieval_diagnostics',
    retrieval_diagnostics: {
      timeline: [{ step: 'retrieve' }],
      candidate_counts: { retrieved: 2, reranked: 1 }
    }
  });
});

test('fallback supports non-standard plain json line stream', () => {
  const frames = collectEvents(['{"delta":"hello"}\n', '{"delta":" world"}\n']);
  assert.deepEqual(
    frames.map((item) => item.type),
    ['content', 'content']
  );
  assert.equal(frames[0].content, 'hello');
  assert.equal(frames[1].content, ' world');
});

test('parse stage progress frames with stage and message fields', () => {
  const frames = collectEvents([
    'event: stage\n',
    'data: {"stage":"retrieval","message":"正在检索知识库并核验证据…"}\n\n',
    'event: stage\n',
    'data: {"stage":"generating","message":"证据核验通过，正在生成回答…"}\n\n'
  ]);

  assert.deepEqual(
    frames.map((item) => ({ type: item.type, stage: item.stage, message: item.message })),
    [
      { type: 'stage', stage: 'retrieval', message: '正在检索知识库并核验证据…' },
      { type: 'stage', stage: 'generating', message: '证据核验通过，正在生成回答…' }
    ]
  );
});

test('reject unterminated, empty, and malformed SSE done terminals', () => {
  assert.deepEqual(collectEvents(['event: done\ndata: [DONE]']), [
    {
      type: 'protocol_error',
      error: 'stream ended before an SSE frame separator'
    }
  ]);
  assert.deepEqual(collectEvents(['event: done\n\n']), [
    {
      type: 'protocol_error',
      error: 'stream emitted a malformed terminal done event'
    }
  ]);
  assert.deepEqual(normalizeSSEFrame({ event: 'done', data: '{"state":"failed"}' }), {
    type: 'protocol_error',
    error: 'stream emitted a malformed terminal done event'
  });
  assert.deepEqual(normalizeSSEFrame({ event: 'done', data: '"[DONE]"' }), {
    type: 'protocol_error',
    error: 'stream emitted a malformed terminal done event'
  });
  assert.deepEqual(normalizeSSEFrame({ event: 'stage', data: '[DONE]' }), {
    type: 'protocol_error',
    error: 'stream emitted a terminal marker on a non-done event'
  });
  assert.deepEqual(normalizeSSEFrame({ event: 'error', data: '"[DONE]"' }), {
    type: 'protocol_error',
    error: 'stream emitted a terminal marker on a non-done event'
  });
});

test('preserve empty and unterminated semantic frames as protocol errors', () => {
  assert.deepEqual(collectEvents(['event: outcome\n\n']), [
    {
      type: 'protocol_error',
      error: 'stream emitted a semantic event without data'
    }
  ]);
  assert.deepEqual(
    collectEvents(['event: outcome\ndata: {"outcome":"insufficient_evidence_reply"}']),
    [
      {
        type: 'protocol_error',
        error: 'stream ended before an SSE frame separator'
      }
    ]
  );
});

test('reject duplicate JSON keys before terminal projection can infer a completion', () => {
  assert.deepEqual(
    normalizeSSEFrame({
      event: 'outcome',
      data: '{"outcome":"evidence_gated_answer","outcome":"non_knowledge_base_reply"}'
    }),
    {
      type: 'protocol_error',
      error: 'stream emitted a JSON payload with duplicate keys'
    }
  );
  assert.deepEqual(
    normalizeSSEFrame({
      event: 'answer_execution',
      data:
        '{"answer_execution":{"id":"answer_execution:1","state":"completed","query_condition_set":{"identity":"qcs:1","identity":"qcs:2","conditions":[]}}}'
    }),
    {
      type: 'protocol_error',
      error: 'stream emitted a JSON payload with duplicate keys'
    }
  );
});
