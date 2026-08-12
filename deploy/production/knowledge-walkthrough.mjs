import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFile, writeFile } from 'node:fs/promises';
import { basename, resolve } from 'node:path';

const required = (name) => {
  const value = process.env[name]?.trim();
  assert.ok(value, `${name} is required`);
  return value;
};

const siteAddress = process.env.DEPLOY_CADDY_SITE_ADDRESS?.trim();
const baseUrl = (process.env.KNOWLEDGE_WALKTHROUGH_BASE_URL || `https://${siteAddress}`).replace(/\/$/, '');
const apiUrl = (process.env.KNOWLEDGE_WALKTHROUGH_API_URL || `${baseUrl}/api`).replace(/\/$/, '');
const sourceRevision = required('SOURCE_REVISION');
const entryPath = resolve(required('KNOWLEDGE_WALKTHROUGH_ENTRY_PATH'));
const entryId = required('KNOWLEDGE_WALKTHROUGH_ENTRY_ID');
const directQuery = required('KNOWLEDGE_WALKTHROUGH_DIRECT_QUERY');
const paraphraseQuery = required('KNOWLEDGE_WALKTHROUGH_PARAPHRASE_QUERY');
const boundaryQuery = required('KNOWLEDGE_WALKTHROUGH_BOUNDARY_QUERY');
const outputPath = resolve(required('KNOWLEDGE_WALKTHROUGH_OUTPUT'));
const generationProvider = required('RAG_PRIMARY_LLM_PROVIDER');
const generationModel = required('MODEL');
const embeddingModel = required('EMBEDDING_MODEL');
const runId = process.env.KNOWLEDGE_WALKTHROUGH_RUN_ID || `knowledge-walkthrough-${new Date().toISOString().replace(/[-:.]/g, '')}`;
const startedAt = new Date().toISOString();

const repositoryRoot = resolve(import.meta.dirname, '../..');
const require = createRequire(resolve(repositoryRoot, 'frontend/package.json'));
const { chromium } = require('@playwright/test');

const request = async (path, { token, method = 'GET', body } = {}) => {
  const response = await fetch(`${apiUrl}${path}`, {
    method,
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(body && !(body instanceof FormData) ? { 'Content-Type': 'application/json' } : {})
    },
    body: body instanceof FormData ? body : body ? JSON.stringify(body) : undefined
  });
  const payload = await response.json();
  return { response, data: payload.data || payload };
};

const login = async (username, password) => {
  const result = await request('/auth/login', { method: 'POST', body: { username, password } });
  assert.equal(result.response.status, 200, `login failed with ${result.response.status}`);
  return result.data.access_token;
};

const parseSse = (text) => {
  const events = [];
  let event = '';
  for (const line of text.split(/\r?\n/)) {
    if (line.startsWith('event: ')) event = line.slice(7);
    if (line.startsWith('data: ')) {
      const data = line.slice(6);
      events.push({ event, data: data === '[DONE]' ? data : JSON.parse(data) });
    }
  }
  return events;
};

const results = [];
const record = (check, outcome = 'non_knowledge_base_reply') => {
  results.push({ case_id: `${runId}-${check}`, check, outcome, passed: true, limitations: [] });
};

const adminToken = await login(required('BOOTSTRAP_ADMIN_USERNAME'), required('BOOTSTRAP_ADMIN_PASSWORD'));
record('bootstrap_administrator');

const entryContent = await readFile(entryPath);
const form = new FormData();
form.set('chunk_strategy', 'agent');
form.set('file', new Blob([entryContent], { type: 'text/markdown' }), basename(entryPath));
const upload = await request('/documents/upload', { token: adminToken, method: 'POST', body: form });
assert.equal(upload.response.status, 200);

const jobDeadline = Date.now() + 120000;
let job;
while (Date.now() < jobDeadline) {
  const result = await request(`/documents/jobs/${upload.data.job_id}`, { token: adminToken });
  assert.equal(result.response.status, 200);
  job = result.data;
  if (['succeeded', 'failed', 'canceled'].includes(job.status)) break;
  await new Promise((resolveWait) => setTimeout(resolveWait, 500));
}
assert.equal(job?.status, 'succeeded', 'Candidate Build failed');
assert.match(job.message, /awaiting publication/);
const candidate = await request(`/documents/${upload.data.document_id}/chunks?page=1&page_size=200`, { token: adminToken });
assert.equal(candidate.response.status, 200);
assert.equal(candidate.data.generation_state, 'candidate');
assert.equal(candidate.data.items.every((item) => item.metadata.entry_id === entryId), true);
const publication = await request(`/documents/${upload.data.document_id}/publish`, { token: adminToken, method: 'POST' });
assert.equal(publication.response.status, 200);
assert.equal(publication.data.published_generation, candidate.data.generation);
record('candidate_publication');

const invitation = await request('/members/invitations', { token: adminToken, method: 'POST', body: {} });
assert.equal(invitation.response.status, 200);
const username = `interviewer-${Date.now().toString(36)}`;
const password = crypto.randomUUID();
const registration = await request('/auth/register', {
  method: 'POST',
  body: { username, password, invitation_code: invitation.data.invitation_code }
});
assert.equal(registration.response.status, 200);
const userToken = registration.data.access_token;
record('team_invitation');

const knowledgeMap = await request('/knowledge-map', { token: userToken });
assert.equal(knowledgeMap.response.status, 200);
assert.equal(knowledgeMap.data.themes.some((theme) => theme.entries.some((entry) => entry.entry_id === entryId)), true);
record('knowledge_map');

const normal = await request('/chat', {
  token: userToken,
  method: 'POST',
  body: { message: directQuery, session_id: `${runId}-normal` }
});
assert.equal(normal.response.status, 200);
assert.equal(normal.data.outcome, 'evidence_gated_answer');
const normalSummary = normal.data.message.evidence_summary;
assert.equal(normalSummary.sources.some((source) => source.entry_id === entryId), true);
record('normal_answer', normal.data.outcome);

const streamResponse = await fetch(`${apiUrl}/chat/stream`, {
  method: 'POST',
  headers: { Authorization: `Bearer ${userToken}`, 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: directQuery, session_id: `${runId}-stream` })
});
assert.equal(streamResponse.status, 200);
const events = parseSse(await streamResponse.text());
const streamSummary = events.find((item) => item.event === 'evidence_summary')?.data.evidence_summary;
assert.deepEqual(streamSummary, normalSummary);
assert.equal(events.some((item) => item.event === 'answer_identity'), true);
record('sse_answer', 'evidence_gated_answer');

const history = await request(`/sessions/${runId}-stream`, { token: userToken });
assert.equal(history.response.status, 200);
const historyAnswer = history.data.messages.find((item) => item.type === 'assistant');
assert.deepEqual(historyAnswer.evidence_summary, normalSummary);
assert.equal(historyAnswer.content, normal.data.message.content);
record('history_citations', 'evidence_gated_answer');

const paraphrase = await request('/chat', {
  token: userToken,
  method: 'POST',
  body: { message: paraphraseQuery, session_id: `${runId}-paraphrase` }
});
assert.equal(paraphrase.response.status, 200);
assert.equal(paraphrase.data.outcome, 'evidence_gated_answer');
record('paraphrase_answer', paraphrase.data.outcome);

const boundary = await request('/chat', {
  token: userToken,
  method: 'POST',
  body: { message: boundaryQuery, session_id: `${runId}-boundary` }
});
assert.equal(boundary.response.status, 200);
assert.equal(boundary.data.outcome, 'insufficient_evidence_reply');
assert.equal(boundary.data.message.evidence_summary.coverage, 'insufficient');
record('insufficient_evidence', boundary.data.outcome);

const source = normalSummary.sources.find((item) => item.entry_id === entryId);
assert.ok(source?.source_url && source?.excerpt);
const feedback = await request('/knowledge-feedback', {
  token: userToken,
  method: 'POST',
  body: { answer_id: normal.data.message.id, entry_id: entryId, label: 'helpful' }
});
assert.equal(feedback.response.status, 200);
record('knowledge_feedback');

const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  await page.addInitScript(({ token }) => localStorage.setItem('access_token', token), { token: userToken });
  await page.goto(`${baseUrl}/knowledge`, { waitUntil: 'networkidle' });
  const entry = page.locator(`[aria-label]`).filter({ hasText: entryId }).first();
  await entry.waitFor();
  await entry.getByRole('button', { name: '基于此条提问' }).click();
  await page.getByRole('button', { name: '发送' }).click();
  const answer = page.getByLabel('助手消息').last();
  await answer.getByRole('status').filter({ hasText: '已完成' }).waitFor();
  await answer.getByRole('button', { name: /查看来源/ }).first().click();
  const citation = page.getByRole('complementary', { name: '来源摘录' });
  await citation.waitFor();
  assert.equal(await citation.getByRole('link', { name: '打开公开来源' }).getAttribute('href'), source.source_url);
  record('public_source_citation');
} finally {
  await browser.close();
}

const deactivation = await request(`/members/${username}/deactivate`, { token: adminToken, method: 'POST' });
assert.equal(deactivation.response.status, 200);
assert.equal(deactivation.data.is_active, false);
assert.equal((await request('/auth/me', { token: userToken })).response.status, 401);
record('deactivation');

const finishedAt = new Date().toISOString();
await writeFile(outputPath, `${JSON.stringify({
  schema_version: 1,
  run_id: runId,
  environment: 'production-compose',
  source_revision: sourceRevision,
  knowledge_edition_manifest: 'knowledge-base-pilot-20260812/knowledge-edition.json',
  provider_identity: generationProvider,
  model_identities: { generation: generationModel, embedding: embeddingModel },
  observation_window: { started_at: startedAt, finished_at: finishedAt },
  limits: [
    'One reviewed Pilot entry was sampled through the production path.',
    'No question, answer, excerpt, user identity, credential, host, or private corpus text is recorded.',
    'The result applies only to this source revision, model identity, and observation window.'
  ],
  results
}, null, 2)}\n`);

process.stdout.write(`Production Interviewer Walkthrough passed: ${runId}\n`);
