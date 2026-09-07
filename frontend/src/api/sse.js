const DONE_MARKER = '[DONE]';
const PARSER_ERROR_EVENT = '__sse_parser_error__';
const EVENTS_REQUIRING_DATA = new Set([
  'answer_identity',
  'answer_execution',
  'outcome',
  'insufficient_evidence_reply',
  'evidence_summary',
  'retrieval_diagnostics',
  'error'
]);

class DuplicateJSONKeyError extends Error {}

const skipJSONWhitespace = (source, index) => {
  let cursor = index;
  while (source[cursor] === ' ' || source[cursor] === '\n' || source[cursor] === '\r' || source[cursor] === '\t') {
    cursor += 1;
  }
  return cursor;
};

const consumeJSONString = (source, index) => {
  let cursor = index + 1;
  while (cursor < source.length) {
    if (source[cursor] === '\\') {
      cursor += 2;
    } else if (source[cursor] === '"') {
      return cursor + 1;
    } else {
      cursor += 1;
    }
  }
  throw new SyntaxError('unterminated JSON string');
};

const consumeJSONValue = (source, index) => {
  let cursor = skipJSONWhitespace(source, index);
  if (source[cursor] === '{') {
    cursor = skipJSONWhitespace(source, cursor + 1);
    const keys = new Set();
    if (source[cursor] === '}') return cursor + 1;
    while (true) {
      if (source[cursor] !== '"') throw new SyntaxError('JSON object key must be a string');
      const keyStart = cursor;
      cursor = consumeJSONString(source, cursor);
      const key = JSON.parse(source.slice(keyStart, cursor));
      if (keys.has(key)) throw new DuplicateJSONKeyError('JSON object contains a duplicate key');
      keys.add(key);
      cursor = skipJSONWhitespace(source, cursor);
      if (source[cursor] !== ':') throw new SyntaxError('JSON object key has no value');
      cursor = consumeJSONValue(source, cursor + 1);
      cursor = skipJSONWhitespace(source, cursor);
      if (source[cursor] === '}') return cursor + 1;
      if (source[cursor] !== ',') throw new SyntaxError('JSON object members are malformed');
      cursor = skipJSONWhitespace(source, cursor + 1);
    }
  }
  if (source[cursor] === '[') {
    cursor = skipJSONWhitespace(source, cursor + 1);
    if (source[cursor] === ']') return cursor + 1;
    while (true) {
      cursor = consumeJSONValue(source, cursor);
      cursor = skipJSONWhitespace(source, cursor);
      if (source[cursor] === ']') return cursor + 1;
      if (source[cursor] !== ',') throw new SyntaxError('JSON array members are malformed');
      cursor = skipJSONWhitespace(source, cursor + 1);
    }
  }
  if (source[cursor] === '"') return consumeJSONString(source, cursor);
  const primitive = source
    .slice(cursor)
    .match(/^(?:true|false|null|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)/);
  if (!primitive) throw new SyntaxError('JSON value is malformed');
  return cursor + primitive[0].length;
};

const assertUniqueJSONKeys = (source) => {
  const end = consumeJSONValue(source, skipJSONWhitespace(source, 0));
  if (skipJSONWhitespace(source, end) !== source.length) {
    throw new SyntaxError('JSON payload has trailing data');
  }
};

const parsePayload = (raw) => {
  if (!raw) return { value: null, hasDuplicateKeys: false };
  if (raw === DONE_MARKER) return { value: DONE_MARKER, hasDuplicateKeys: false };

  try {
    const value = JSON.parse(raw);
    assertUniqueJSONKeys(raw);
    return { value, hasDuplicateKeys: false };
  } catch (error) {
    if (error instanceof DuplicateJSONKeyError) {
      return { value: null, hasDuplicateKeys: true };
    }
    return { value: raw, hasDuplicateKeys: false };
  }
};

export const normalizeSSEFrame = (frame) => {
  const eventName = frame?.event || 'message';
  const rawData = frame?.data ?? '';

  if (eventName === PARSER_ERROR_EVENT) {
    return {
      type: 'protocol_error',
      error: rawData || 'stream ended before an SSE frame separator'
    };
  }
  if (eventName === 'done') {
    if (rawData === DONE_MARKER) {
      return { type: 'done' };
    }
    return {
      type: 'protocol_error',
      error: 'stream emitted a malformed terminal done event'
    };
  }
  if (!rawData && EVENTS_REQUIRING_DATA.has(eventName)) {
    return {
      type: 'protocol_error',
      error: 'stream emitted a semantic event without data'
    };
  }
  const parsedPayload = parsePayload(rawData);
  if (parsedPayload.hasDuplicateKeys) {
    return {
      type: 'protocol_error',
      error: 'stream emitted a JSON payload with duplicate keys'
    };
  }
  const payload = parsedPayload.value;
  if (payload === DONE_MARKER) {
    return {
      type: 'protocol_error',
      error: 'stream emitted a terminal marker on a non-done event'
    };
  }

  if (eventName === 'content') {
    if (typeof payload === 'string') {
      return { type: 'content', content: payload };
    }
    return { type: 'content', content: payload?.content || payload?.delta || '' };
  }

  if (eventName === 'answer_identity') {
    return { type: 'answer_identity', answer_id: payload?.answer_id || '' };
  }

  if (eventName === 'answer_execution') {
    return { type: 'answer_execution', answer_execution: payload?.answer_execution ?? payload };
  }

  if (eventName === 'outcome') {
    return { type: 'outcome', outcome: payload?.outcome || '' };
  }

  if (eventName === 'insufficient_evidence_reply') {
    return {
      type: 'insufficient_evidence_reply',
      insufficient_evidence_reply: payload?.insufficient_evidence_reply ?? payload
    };
  }

  if (eventName === 'evidence_summary') {
    return { type: 'evidence_summary', evidence_summary: payload?.evidence_summary ?? payload };
  }

  if (eventName === 'retrieval_diagnostics') {
    return { type: 'retrieval_diagnostics', retrieval_diagnostics: payload?.retrieval_diagnostics ?? payload };
  }

  if (eventName === 'stage') {
    return { type: 'stage', stage: payload?.stage || '', message: payload?.message ?? payload?.detail ?? '' };
  }

  if (eventName === 'rag_step') {
    return { type: 'rag_step', data: payload?.step ?? payload };
  }

  if (eventName === 'trace') {
    return { type: 'trace', data: payload?.trace ?? payload };
  }

  if (eventName === 'error') {
    return { type: 'error', error: payload?.error || payload };
  }

  // 兼容后端直接推送 JSON 行，而不是标准 event/data 对。
  if (payload && typeof payload === 'object') {
    if (payload.type === 'content') return { type: 'content', content: payload.content || payload.delta || '' };
    if (payload.type === 'answer_identity') {
      return { type: 'answer_identity', answer_id: payload.answer_id || '' };
    }
    if (payload.type === 'answer_execution') {
      return { type: 'answer_execution', answer_execution: payload.answer_execution ?? payload.data ?? payload };
    }
    if (payload.type === 'outcome') {
      return { type: 'outcome', outcome: payload.outcome || payload.data?.outcome || '' };
    }
    if (payload.type === 'insufficient_evidence_reply') {
      return {
        type: 'insufficient_evidence_reply',
        insufficient_evidence_reply:
          payload.insufficient_evidence_reply ?? payload.data?.insufficient_evidence_reply ?? payload.data ?? payload
      };
    }
    if (payload.type === 'evidence_summary') {
      return { type: 'evidence_summary', evidence_summary: payload.evidence_summary ?? payload.data ?? payload };
    }
    if (payload.type === 'retrieval_diagnostics') {
      return { type: 'retrieval_diagnostics', retrieval_diagnostics: payload.retrieval_diagnostics ?? payload.data ?? payload };
    }
    if (payload.type === 'stage') {
      return { type: 'stage', stage: payload.stage || '', message: payload.message ?? payload.detail ?? '' };
    }
    if (payload.type === 'rag_step') return { type: 'rag_step', data: payload.step ?? payload.data ?? payload };
    if (payload.type === 'trace') return { type: 'trace', data: payload.trace ?? payload.data ?? payload };
    if (payload.type === 'error') return { type: 'error', error: payload.error || payload.detail || payload };
    if (Object.prototype.hasOwnProperty.call(payload, 'delta') || Object.prototype.hasOwnProperty.call(payload, 'content')) {
      return { type: 'content', content: payload.content || payload.delta || '' };
    }
  }

  if (typeof payload === 'string' && payload) {
    return { type: 'content', content: payload };
  }

  return { type: 'unknown', data: payload };
};

export const createSSEParser = (onFrame) => {
  let lineBuffer = '';
  let currentEvent = 'message';
  let dataLines = [];

  const flushEvent = () => {
    if (!dataLines.length) {
      if (currentEvent !== 'message') {
        onFrame({ event: currentEvent, data: '' });
      }
      currentEvent = 'message';
      dataLines = [];
      return;
    }

    onFrame({
      event: currentEvent || 'message',
      data: dataLines.join('\n')
    });
    currentEvent = 'message';
    dataLines = [];
  };

  const consumeLine = (line) => {
    if (!line) {
      flushEvent();
      return;
    }

    if (line.startsWith(':')) {
      return;
    }

    const separator = line.indexOf(':');
    if (separator === -1) {
      // 非标准 SSE 行，按 message data 兼容。
      onFrame({ event: 'message', data: line });
      return;
    }

    const field = line.slice(0, separator);
    const value = line.slice(separator + 1).replace(/^ /, '');

    if (field === 'event') {
      currentEvent = value || 'message';
      return;
    }

    if (field === 'data') {
      dataLines.push(value);
      return;
    }

    // 未知字段按原始消息兼容，避免把 JSON 行误丢弃。
    onFrame({ event: 'message', data: line });
  };

  const feed = (textChunk) => {
    if (!textChunk) return;
    lineBuffer += textChunk;
    const lines = lineBuffer.split(/\r?\n/);
    lineBuffer = lines.pop() ?? '';
    lines.forEach(consumeLine);
  };

  const finish = () => {
    // SSE frames are complete only after their blank-line separator. EOF
    // cannot synthesize a terminal from an unterminated final frame.
    if (lineBuffer || currentEvent !== 'message' || dataLines.length) {
      onFrame({
        event: PARSER_ERROR_EVENT,
        data: 'stream ended before an SSE frame separator'
      });
    }
    lineBuffer = '';
    currentEvent = 'message';
    dataLines = [];
  };

  return { feed, finish };
};
