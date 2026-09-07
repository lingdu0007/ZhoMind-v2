import assert from 'node:assert/strict';
import test from 'node:test';
import { startWorkbench } from './acceptance-env.mjs';

const createServerStub = () => {
  let closed = false;
  return {
    resolvedUrls: { local: ['http://127.0.0.1:5173/'] },
    async listen() {},
    async close() {
      closed = true;
    },
    get closed() {
      return closed;
    }
  };
};

const startApiStub = async () => ({ port: 45678 });

test('workbench cleanup closes Vite and restores the proxy target when browser launch fails', async (t) => {
  const server = createServerStub();
  const previousProxyTarget = process.env.ZHOMIND_API_PROXY_TARGET;
  process.env.ZHOMIND_API_PROXY_TARGET = 'http://existing-proxy.test';

  try {
    await assert.rejects(
      startWorkbench(t, {
        startApi: startApiStub,
        createServerForWorkbench: async () => server,
        launchBrowser: async () => {
          throw new Error('forced browser launch failure');
        }
      }),
      /forced browser launch failure/
    );
    assert.equal(server.closed, true);
    assert.equal(process.env.ZHOMIND_API_PROXY_TARGET, 'http://existing-proxy.test');
  } finally {
    if (previousProxyTarget === undefined) delete process.env.ZHOMIND_API_PROXY_TARGET;
    else process.env.ZHOMIND_API_PROXY_TARGET = previousProxyTarget;
  }
});

test('workbench cleanup closes a launched browser when context setup fails', async (t) => {
  const server = createServerStub();
  let browserClosed = false;

  await assert.rejects(
    startWorkbench(t, {
      startApi: startApiStub,
      createServerForWorkbench: async () => server,
      launchBrowser: async () => ({
        async newContext() {
          throw new Error('forced context setup failure');
        },
        async close() {
          browserClosed = true;
        }
      })
    }),
    /forced context setup failure/
  );

  assert.equal(browserClosed, true);
  assert.equal(server.closed, true);
  assert.equal(process.env.ZHOMIND_API_PROXY_TARGET, undefined);
});
