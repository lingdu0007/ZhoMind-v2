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
    'event: done\ndata: "[DONE]"\n\n'
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
