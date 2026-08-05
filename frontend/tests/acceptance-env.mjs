// Shared disposable-API acceptance environment for browser journeys.
//
// Every journey here talks to a real FastAPI application (tests/browser_acceptance_api.py)
// over normal HTTP network traffic through the Vite dev/preview server proxy. Roles come
// from the server (auth/me); no route is mocked.
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, rm } from 'node:fs/promises';
import net from 'node:net';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { chromium } from 'playwright';
import { createServer, preview } from 'vite';

const DEFAULT_ENV = {
  JWT_SECRET: 'browser-acceptance-secret',
  BOOTSTRAP_ADMIN_USERNAME: 'operator',
  BOOTSTRAP_ADMIN_PASSWORD: 'safe-password',
  SYSTEM_SETTINGS_DRAFT_ENABLED: 'true',
  SYSTEM_SETTINGS_APPLICATION_ENABLED: 'true',
  SYSTEM_SETTINGS_ENCRYPTION_KEY: 'MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY=',
  DENSE_EMBEDDING_DIM: '0',
  EMBEDDING_API_KEY: '',
  EMBEDDING_BASE_URL: '',
  EMBEDDING_MODEL: '',
  MILVUS_URI: '',
  MILVUS_TOKEN: '',
  RAG_PRIMARY_LLM_PROVIDER: 'browser-acceptance',
  RAG_LLM_FALLBACK_PROVIDERS: ''
};

const reservePort = () =>
  new Promise((resolvePort, reject) => {
    const listener = net.createServer();
    listener.once('error', reject);
    listener.listen(0, '127.0.0.1', () => {
      const address = listener.address();
      listener.close((error) => (error ? reject(error) : resolvePort(address.port)));
    });
  });

/**
 * Start one disposable application API environment backed by a temporary
 * SQLite file, an in-memory Redis, and a deterministic LLM adapter.
 * Extra environment variables may steer deterministic scenarios, e.g.
 * `BROWSER_ACCEPTANCE_LLM_DELAY_MS`, `BROWSER_ACCEPTANCE_FAIL_FIRST`,
 * `BROWSER_ACCEPTANCE_SETTINGS_FAILED`, or `BROWSER_ACCEPTANCE_SEED=minimal`.
 */
export const startApiEnvironment = async (t, { env = {} } = {}) => {
  const tempDirectory = await mkdtemp(join(tmpdir(), 'zhomind-browser-api-'));
  const port = await reservePort();
  const baseUrl = `http://127.0.0.1:${port}/api/v1`;
  const backendDirectory = resolve(process.cwd(), '../backend');
  const output = [];
  // --no-sync: the backend environment is installed by the surrounding gate
  // step (`uv sync --frozen`); do not let `uv run` rewrite dependency state.
  const apiProcess = spawn(
    'uv',
    ['run', '--no-sync', 'python', 'tests/browser_acceptance_api.py', '--host', '127.0.0.1', '--port', String(port)],
    {
      cwd: backendDirectory,
      // Own process group so a failed journey can never leave the uv wrapper or
      // its python child behind to exhaust the runner's memory.
      detached: true,
      env: {
        ...process.env,
        PYTHONPATH: backendDirectory,
        DATABASE_URL: `sqlite+aiosqlite:///${join(tempDirectory, 'acceptance.db')}`,
        ...DEFAULT_ENV,
        ...env
      },
      stdio: ['ignore', 'pipe', 'pipe']
    }
  );
  apiProcess.stdout.on('data', (chunk) => output.push(chunk.toString()));
  apiProcess.stderr.on('data', (chunk) => output.push(chunk.toString()));

  t.after(async () => {
    if (apiProcess.exitCode === null) {
      try {
        process.kill(-apiProcess.pid, 'SIGTERM');
      } catch {
        apiProcess.kill('SIGTERM');
      }
      await Promise.race([once(apiProcess, 'exit'), new Promise((resolveWait) => setTimeout(resolveWait, 5000))]);
      if (apiProcess.exitCode === null) {
        try {
          process.kill(-apiProcess.pid, 'SIGKILL');
        } catch {
          apiProcess.kill('SIGKILL');
        }
      }
    }
    await rm(tempDirectory, { recursive: true, force: true });
  });

  const deadline = Date.now() + 20000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${baseUrl}/health`);
      if (response.ok) return { baseUrl, port };
    } catch {
      // The runner has not bound its socket yet.
    }
    if (apiProcess.exitCode !== null) break;
    await new Promise((resolveWait) => setTimeout(resolveWait, 100));
  }

  throw new Error(`isolated application API environment did not become healthy:\n${output.join('')}`);
};

/**
 * Start a workbench: disposable API + Vite dev or preview server + a Playwright
 * page. The Vite proxy forwards /api to the disposable API over real network
 * traffic.
 */
export const startWorkbench = async (t, { built = false, env = {}, viewport = { width: 1440, height: 900 } } = {}) => {
  const api = await startApiEnvironment(t, { env });
  const previousProxyTarget = process.env.ZHOMIND_API_PROXY_TARGET;
  process.env.ZHOMIND_API_PROXY_TARGET = `http://127.0.0.1:${api.port}`;
  // Vite resolves `port: 0` from the project config (5173), so reserve a
  // distinct random port per journey; parallel test files must never collide.
  const webPort = await reservePort();
  const server = built
    ? await preview({ preview: { host: '127.0.0.1', port: webPort, strictPort: true } })
    : await createServer({ server: { host: '127.0.0.1', port: webPort, strictPort: true } });
  if (!built) await server.listen();

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport });
  const page = await context.newPage();
  page.setDefaultTimeout(20000);

  t.after(async () => {
    await browser.close();
    await server.close();
    if (previousProxyTarget === undefined) delete process.env.ZHOMIND_API_PROXY_TARGET;
    else process.env.ZHOMIND_API_PROXY_TARGET = previousProxyTarget;
  });

  return { page, baseUrl: server.resolvedUrls.local[0], api };
};

export const createTeamInvitation = async (api) => {
  const login = await fetch(`${api.baseUrl}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: 'operator', password: 'safe-password' })
  });
  if (login.status !== 200) throw new Error(`operator login failed: ${login.status}`);
  const token = (await login.json()).data.access_token;
  const invitation = await fetch(`${api.baseUrl}/members/invitations`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({})
  });
  if (invitation.status !== 200) throw new Error(`invitation creation failed: ${invitation.status}`);
  return (await invitation.json()).data.invitation_code;
};

export const loginAdmin = async (page, baseUrl, { username = 'operator' } = {}) => {
  await page.goto(`${baseUrl}auth`);
  await page.getByRole('heading', { name: '身份验证' }).waitFor();
  await page.getByLabel('用户名').fill(username);
  await page.getByLabel('密码').fill('safe-password');
  await page.getByRole('button', { name: '登录' }).click();
  await page.waitForURL(/\/chat$/);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();
};

export const registerKnowledgeUser = async (page, baseUrl, api, username) => {
  await page.goto(`${baseUrl}auth`);
  await page.getByRole('heading', { name: '身份验证' }).waitFor();
  await page.getByRole('tab', { name: '注册' }).click();
  await page.getByLabel('用户名').fill(username);
  await page.getByLabel('密码').fill('safe-password');
  await page.getByLabel('团队邀请码').fill(await createTeamInvitation(api));
  await page.getByRole('button', { name: '完成注册' }).click();
  await page.waitForURL(/\/chat$/);
  await page.getByRole('heading', { name: '对话工作区' }).waitFor();
};

/** Register a Knowledge User through the API and return its bearer token. */
export const registerKnowledgeUserViaApi = async (api, username) => {
  const invitation = await createTeamInvitation(api);
  const response = await fetch(`${api.baseUrl}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password: 'safe-password', invitation_code: invitation })
  });
  if (response.status !== 200) throw new Error(`knowledge user registration failed: ${response.status}`);
  return (await response.json()).data.access_token;
};
