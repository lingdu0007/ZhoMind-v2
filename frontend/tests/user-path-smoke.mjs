import assert from 'node:assert/strict';

const API_BASE = process.env.API_BASE || 'http://127.0.0.1:8000/api/v1';

const unwrapEnvelopeData = (payload) => {
  if (payload && typeof payload === 'object' && !Array.isArray(payload) && 'data' in payload) {
    return payload.data;
  }
  return payload;
};

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

  return {
    status: response.status,
    body: json,
    data: unwrapEnvelopeData(json)
  };
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
    if (['succeeded', 'failed', 'canceled'].includes(resp.data?.status)) return resp;
    await new Promise((r) => setTimeout(r, resp.data?.status === 'queued' ? 1200 : 800));
  }
  throw new Error(`job timeout: ${jobId}`);
};

const run = async () => {
  const stamp = Date.now();

  const username = `smoke_user_${stamp}`;
  const password = 'smoke-pass-123';

  const register = await request({
    method: 'POST',
    path: '/auth/register',
    body: { username, password, role: 'user' }
  });
  assert.equal(register.status, 200, 'register failed');

  const login = await request({
    method: 'POST',
    path: '/auth/login',
    body: { username, password }
  });
  assert.equal(login.status, 200, 'login failed');
  const token = login.data?.access_token;
  assert.ok(token, 'missing access_token');

  const userUploadForm = new FormData();
  userUploadForm.append('file', new Blob(['line1'], { type: 'text/plain' }), `smoke-user-${stamp}.txt`);
  const userUpload = await request({
    method: 'POST',
    path: '/documents/upload',
    token,
    body: userUploadForm,
    isForm: true
  });
  assert.equal(userUpload.status, 403, 'user upload should be forbidden');

  const adminUsername = `smoke_admin_${stamp}`;
  const adminCode = process.env.SMOKE_ADMIN_CODE;
  const adminRegisterBody = adminCode
    ? { username: adminUsername, password, role: 'admin', admin_code: adminCode }
    : { username: adminUsername, password, role: 'admin' };

  const registerAdmin = await request({
    method: 'POST',
    path: '/auth/register',
    body: adminRegisterBody
  });
  assert.equal(registerAdmin.status, 200, 'admin register failed');

  const loginAdmin = await request({
    method: 'POST',
    path: '/auth/login',
    body: { username: adminUsername, password }
  });
  assert.equal(loginAdmin.status, 200, 'admin login failed');
  const adminToken = loginAdmin.data?.access_token;
  assert.ok(adminToken, 'missing admin access_token');

  const uploadOkForm = new FormData();
  uploadOkForm.append('file', new Blob(['line1\nline2\nline3'], { type: 'text/plain' }), `smoke-ok-${stamp}.txt`);
  const uploadOk = await request({
    method: 'POST',
    path: '/documents/upload',
    token: adminToken,
    body: uploadOkForm,
    isForm: true
  });
  assert.equal(uploadOk.status, 200, 'admin upload should succeed');

  const uploadEdgeForm = new FormData();
  uploadEdgeForm.append('file', new Blob([' \n\t\n '], { type: 'text/plain' }), `smoke-edge-${stamp}.txt`);
  const uploadEdge = await request({
    method: 'POST',
    path: '/documents/upload',
    token: adminToken,
    body: uploadEdgeForm,
    isForm: true
  });
  assert.equal(uploadEdge.status, 200, 'edge upload should succeed');

  const okJob = await waitJobTerminal(adminToken, uploadOk.data.job_id);
  assert.equal(okJob.status, 200);
  assert.equal(okJob.data.status, 'succeeded');

  const edgeJob = await waitJobTerminal(adminToken, uploadEdge.data.job_id);
  assert.equal(edgeJob.status, 200);
  assert.equal(edgeJob.data.status, 'succeeded');

  const chatSync = await request({
    method: 'POST',
    path: '/chat',
    token,
    body: { message: '你好', session_id: `smoke_${Date.now()}` }
  });
  assert.equal(chatSync.status, 200, 'chat sync failed');
  assert.ok(chatSync.data?.message?.content, 'chat sync empty answer');

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
          user_upload_status: userUpload.status,
          admin_upload_job: okJob.body,
          admin_edge_job: edgeJob.body
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
