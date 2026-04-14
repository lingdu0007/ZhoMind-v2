const DONE_MARKER = '[DONE]';

const parsePayload = (raw) => {
  if (!raw) return null;
  if (raw === DONE_MARKER) return DONE_MARKER;

  try {
    return JSON.parse(raw);
  } catch {
    return raw;
  }
};

export const normalizeSSEFrame = (frame) => {
  const eventName = frame?.event || 'message';
  const payload = parsePayload(frame?.data || '');

  if (eventName === 'done' || payload === DONE_MARKER) {
    return { type: 'done' };
  }

  if (eventName === 'content') {
    if (typeof payload === 'string') {
      return { type: 'content', content: payload };
    }
    return { type: 'content', content: payload?.content || payload?.delta || '' };
  }

  if (eventName === 'rag_step') {
    return { type: 'rag_step', data: payload?.step ?? payload };
  }

  if (eventName === 'trace') {
    return { type: 'trace', data: payload?.trace ?? payload };
  }

  if (eventName === 'error') {
    return { type: 'error', error: payload?.error || payload?.message || payload?.detail || payload };
  }

  // 兼容后端直接推送 JSON 行，而不是标准 event/data 对。
  if (payload && typeof payload === 'object') {
    if (payload.type === 'done') return { type: 'done' };
    if (payload.type === 'content') return { type: 'content', content: payload.content || payload.delta || '' };
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
      currentEvent = 'message';
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
    if (lineBuffer) {
      consumeLine(lineBuffer);
      lineBuffer = '';
    }
    flushEvent();
  };

  return { feed, finish };
};
