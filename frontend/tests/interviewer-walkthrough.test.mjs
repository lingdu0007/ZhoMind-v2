import assert from 'node:assert/strict';
import test from 'node:test';
import { registerKnowledgeUserViaApi, startWorkbench } from './acceptance-env.mjs';

const ENTRY_ID = 'pae-acceptance-workflow-001';
const SOURCE_URL = 'https://www.anthropic.com/engineering/building-effective-agents';
const DIRECT_QUERY = '什么时候应该使用 deterministic workflow？';
const PARAPHRASE_QUERY = '固定 steps 的任务还需要 autonomous Agent loop 吗？';
const BOUNDARY_QUERY = '量子烹饪协议的强制超时时间是多少？';

const syntheticAgentEntry = `---
entry_id: ${ENTRY_ID}
title: Bound a deterministic workflow before adding agent autonomy
domain: workflow-vs-agent
review_status: approved
review_date: 2026-08-12
applicable_versions:
  - framework-neutral
approved_summary: 固定 steps 与 termination condition 已知时，优先使用 deterministic workflow。
suggested_query: ${DIRECT_QUERY}
sources:
  - title: Building effective agents
    authority: Anthropic
    url: ${SOURCE_URL}
    version: 2026-08-12
    availability: verified
---
# Decision Question

固定 steps 与 termination condition 已知时，是否应该使用 deterministic workflow？

## Recommendation

已知路径应由 deterministic workflow 控制；只有依赖 runtime observation 的局部步骤才进入 bounded Agent loop。

## Applicability

该建议适用于固定 steps、明确 termination condition 与可重放输入。

## Alternatives and Trade-offs

autonomous Agent loop 可处理 runtime observation，但必须增加 step budget 与 tool budget。

## Validation

验证固定输入的步骤、tool 参数和终止结果可以重放。
`;

const apiRequest = async (api, path, { token, method = 'GET', body, headers = {} } = {}) => {
  const response = await fetch(`${api.baseUrl}${path}`, {
    method,
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(body && !(body instanceof FormData) ? { 'Content-Type': 'application/json' } : {}),
      ...headers
    },
    body: body instanceof FormData ? body : body ? JSON.stringify(body) : undefined
  });
  const payload = await response.json();
  return { response, data: payload.data || payload };
};

const loginOperator = async (api) => {
  const { response, data } = await apiRequest(api, '/auth/login', {
    method: 'POST',
    body: { username: 'operator', password: 'safe-password' }
  });
  assert.equal(response.status, 200);
  return data.access_token;
};

const publishThroughCandidateBuild = async (api, adminToken) => {
  const uploadBody = new FormData();
  uploadBody.set('chunk_strategy', 'agent');
  uploadBody.set('file', new Blob([syntheticAgentEntry], { type: 'text/markdown' }), `${ENTRY_ID}.md`);
  const upload = await apiRequest(api, '/documents/upload', {
    token: adminToken,
    method: 'POST',
    body: uploadBody
  });
  assert.equal(upload.response.status, 200);

  const deadline = Date.now() + 15000;
  let job;
  while (Date.now() < deadline) {
    const result = await apiRequest(api, `/documents/jobs/${upload.data.job_id}`, { token: adminToken });
    assert.equal(result.response.status, 200);
    job = result.data;
    if (['succeeded', 'failed', 'canceled'].includes(job.status)) break;
    await new Promise((resolveWait) => setTimeout(resolveWait, 50));
  }
  assert.equal(job?.status, 'succeeded', job?.message);
  assert.match(job.message, /awaiting publication/);

  const candidate = await apiRequest(api, `/documents/${upload.data.document_id}/chunks?page=1&page_size=20`, {
    token: adminToken
  });
  assert.equal(candidate.response.status, 200);
  assert.equal(candidate.data.generation_state, 'candidate');
  assert.equal(candidate.data.items.every((item) => item.metadata.entry_id === ENTRY_ID), true);

  const publication = await apiRequest(api, `/documents/${upload.data.document_id}/publish`, {
    token: adminToken,
    method: 'POST'
  });
  assert.equal(publication.response.status, 200);
  assert.equal(publication.data.published_generation, candidate.data.generation);
  assert.equal(publication.data.candidate_generation, null);
  return publication.data;
};

const parseSse = (text) => {
  const events = [];
  let event = '';
  for (const line of text.split(/\r?\n/)) {
    if (line.startsWith('event: ')) event = line.slice(7);
    if (line.startsWith('data: ')) {
      const raw = line.slice(6);
      events.push({ event, data: raw === '[DONE]' ? raw : JSON.parse(raw) });
    }
  }
  return events;
};

test('published Agent evidence without Claim-Evidence Links closes as explicit insufficient evidence', { timeout: 90000 }, async (t) => {
  const { page, baseUrl, api } = await startWorkbench(t, {
    built: true,
    env: {
      BROWSER_ACCEPTANCE_SEED: 'minimal',
      BROWSER_ACCEPTANCE_PUBLIC_SOURCE_URL: SOURCE_URL
    }
  });
  const adminToken = await loginOperator(api);
  const userToken = await registerKnowledgeUserViaApi(api, 'interviewer-user');

  const beforePublication = await apiRequest(api, '/knowledge-map', { token: userToken });
  assert.equal(beforePublication.response.status, 200);
  assert.equal(beforePublication.data.total_entries, 0);

  const publication = await publishThroughCandidateBuild(api, adminToken);
  assert.equal(publication.chunk_strategy, 'agent');
  const afterPublication = await apiRequest(api, '/knowledge-map', { token: userToken });
  assert.equal(afterPublication.response.status, 200);
  assert.equal(afterPublication.data.total_entries, 0);

  await page.addInitScript(({ token }) => localStorage.setItem('access_token', token), { token: userToken });
  await page.goto(`${baseUrl}knowledge`);
  await page.getByRole('heading', { name: '知识地图' }).waitFor();
  await page.locator('section.knowledge-map[aria-busy="false"]').waitFor();
  assert.equal(
    await page.getByRole('article', { name: 'Bound a deterministic workflow before adding agent autonomy' }).count(),
    0
  );
  await page.goto(`${baseUrl}chat`);
  await page.getByPlaceholder('请输入需要检索的问题').fill(DIRECT_QUERY);
  await page.getByRole('button', { name: '发送' }).click();
  const browserAnswer = page.getByLabel('助手消息').last();
  const browserInsufficiency = browserAnswer.getByLabel('证据不足回复');
  await browserInsufficiency.waitFor();
  assert.equal(
    await browserInsufficiency.getByText('Insufficient Evidence Reply', { exact: true }).isVisible(),
    true
  );
  assert.equal(
    await browserAnswer.getByText('未检索到足够相关的知识片段，请补充更具体的问题或关键词。').isVisible(),
    true
  );
  assert.equal(await browserAnswer.getByLabel('证据摘要').count(), 0);
  assert.equal(await browserAnswer.getByRole('button', { name: /打开引用/ }).count(), 0);
  assert.equal(await browserAnswer.getByLabel('知识反馈').count(), 0);

  const normal = await apiRequest(api, '/chat', {
    token: userToken,
    method: 'POST',
    body: { message: DIRECT_QUERY, session_id: 'walkthrough-normal' }
  });
  assert.equal(normal.response.status, 200);
  assert.equal(normal.data.outcome, 'insufficient_evidence_reply');
  assert.equal(normal.data.message.evidence_summary.coverage, 'insufficient');
  assert.equal(normal.data.message.evidence_summary.source_count, 0);
  const normalHistory = await apiRequest(api, '/sessions/walkthrough-normal', { token: userToken });
  assert.equal(normalHistory.response.status, 200);
  const normalHistoryAnswer = normalHistory.data.messages.find((item) => item.type === 'assistant');
  assert.equal(normalHistoryAnswer.id, normal.data.message.id);
  assert.equal(normalHistoryAnswer.outcome, normal.data.outcome);
  assert.equal(normalHistoryAnswer.content, normal.data.message.content);
  assert.deepEqual(normalHistoryAnswer.evidence_summary, normal.data.message.evidence_summary);

  const streamedResponse = await fetch(`${api.baseUrl}/chat/stream`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${userToken}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: DIRECT_QUERY, session_id: 'walkthrough-stream' })
  });
  assert.equal(streamedResponse.status, 200);
  const streamEvents = parseSse(await streamedResponse.text());
  const streamOutcome = streamEvents.find((item) => item.event === 'outcome')?.data.outcome;
  const streamSummary = streamEvents.find((item) => item.event === 'evidence_summary')?.data.evidence_summary;
  assert.equal(streamOutcome, 'insufficient_evidence_reply');
  assert.deepEqual(streamSummary, normal.data.message.evidence_summary);
  const streamAnswerId = streamEvents.find((item) => item.event === 'answer_identity')?.data.answer_id;
  assert.ok(streamAnswerId);
  const streamContent = streamEvents
    .filter((item) => item.event === 'content')
    .map((item) => item.data.content)
    .join('');

  const history = await apiRequest(api, '/sessions/walkthrough-stream', { token: userToken });
  assert.equal(history.response.status, 200);
  const historyAnswer = history.data.messages.find((item) => item.type === 'assistant');
  assert.equal(historyAnswer.outcome, streamOutcome);
  assert.deepEqual(historyAnswer.evidence_summary, streamSummary);
  assert.equal(historyAnswer.id, streamAnswerId);
  assert.equal(historyAnswer.content, streamContent);

  const paraphrase = await apiRequest(api, '/chat', {
    token: userToken,
    method: 'POST',
    body: { message: PARAPHRASE_QUERY, session_id: 'walkthrough-paraphrase' }
  });
  assert.equal(paraphrase.response.status, 200);
  assert.equal(paraphrase.data.outcome, 'insufficient_evidence_reply');
  assert.equal(paraphrase.data.message.evidence_summary.source_count, 0);

  const boundary = await apiRequest(api, '/chat', {
    token: userToken,
    method: 'POST',
    body: { message: BOUNDARY_QUERY, session_id: 'walkthrough-boundary' }
  });
  assert.equal(boundary.response.status, 200);
  assert.equal(boundary.data.outcome, 'insufficient_evidence_reply');
  assert.equal(boundary.data.message.evidence_summary.coverage, 'insufficient');

  const deactivation = await apiRequest(api, '/members/interviewer-user/deactivate', {
    token: adminToken,
    method: 'POST'
  });
  assert.equal(deactivation.response.status, 200);
  assert.equal(deactivation.data.is_active, false);
  const revokedIdentity = await apiRequest(api, '/auth/me', { token: userToken });
  assert.equal(revokedIdentity.response.status, 401);
});
