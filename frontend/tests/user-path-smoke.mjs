import assert from 'node:assert/strict';

const API_BASE = process.env.API_BASE || 'http://127.0.0.1:8000/api/v1';

const request = async ({ method, path, token, body, isForm }) => {
  const headers = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  if (!isForm && body !== undefined) headers['Content-Type'] = 'application/json';

  const response = await fetch(`${API_BASE}${path}`, {
    method,
    headers,
    body: body === undefined ? undefined : isForm ? body : JSON.stringify(body)
  });

  const text = await response.text();
  let json = null;
  try {
    json = text ? JSON.parse(text) : null;
  } catch {
    json = text;
  }

  return { status: response.status, body: json };
};

const waitJobTerminal = async (token, jobId, timeoutMs = 30000) => {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const resp = await request({
      method: 'GET',
      path: `/documents/jobs/${jobId}`,
      token
    });
    if (resp.status !== 200) return resp;
    if (['succeeded', 'failed', 'canceled'].includes(resp.body?.status)) return resp;
    await new Promise((r) => setTimeout(r, resp.body?.status === 'queued' ? 1200 : 800));
  }
  throw new Error(`job timeout: ${jobId}`);
};

const run = async () => {
  const login = await request({
    method: 'POST',
    path: '/auth/login',
    body: { username: 'admin', password: 'admin-token' }
  });
  assert.equal(login.status, 200, 'login failed');
  const token = login.body?.access_token;
  assert.ok(token, 'missing access_token');

  const uploadOkForm = new FormData();
  uploadOkForm.append('file', new Blob(['line1\nline2\nline3'], { type: 'text/plain' }), 'smoke-ok.txt');
  const uploadOk = await request({
    method: 'POST',
    path: '/documents/upload',
    token,
    body: uploadOkForm,
    isForm: true
  });
  assert.equal(uploadOk.status, 202, 'upload ok should be accepted');

  const uploadFailForm = new FormData();
  uploadFailForm.append('file', new Blob([' \n\t\n '], { type: 'text/plain' }), 'smoke-fail.txt');
  const uploadFail = await request({
    method: 'POST',
    path: '/documents/upload',
    token,
    body: uploadFailForm,
    isForm: true
  });
  assert.equal(uploadFail.status, 202, 'upload fail sample should be accepted');

  const okJob = await waitJobTerminal(token, uploadOk.body.job_id);
  assert.equal(okJob.status, 200);
  assert.equal(okJob.body.status, 'succeeded');

  const failJob = await waitJobTerminal(token, uploadFail.body.job_id);
  assert.equal(failJob.status, 200);
  assert.equal(failJob.body.status, 'failed');
  assert.ok(String(failJob.body.message || '').length > 0, 'failed job should carry message');

  const chatSync = await request({
    method: 'POST',
    path: '/chat',
    token,
    body: { message: '你好', session_id: `smoke_${Date.now()}` }
  });
  assert.equal(chatSync.status, 200, 'chat sync failed');
  assert.ok(chatSync.body?.message?.content, 'chat sync empty answer');

  const streamResp = await fetch(`${API_BASE}/chat/stream`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ message: '你好', session_id: `smoke_stream_${Date.now()}` })
  });
  assert.equal(streamResp.status, 200, 'chat stream failed');
  const streamText = await streamResp.text();
  assert.ok(streamText.includes('event: content'), 'stream should contain content event');
  assert.ok(streamText.includes('event: done'), 'stream should contain done event');

  console.log(
    JSON.stringify(
      {
        ok: true,
        api_base: API_BASE,
        evidences: {
          login_status: login.status,
          ok_job: okJob.body,
          fail_job: failJob.body
        }
      },
      null,
      2
    )
  );
};

run().catch((error) => {
  console.error(
    JSON.stringify(
      {
        ok: false,
        api_base: API_BASE,
        error: error.message
      },
      null,
      2
    )
  );
  process.exit(1);
});
