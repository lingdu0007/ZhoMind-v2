import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';

const read = (path) => fs.readFileSync(path, 'utf-8');

test('env base url keeps same-origin /api entry', () => {
  assert.match(read('./.env.development'), /VITE_API_BASE_URL=\/api/);
  assert.match(read('./.env.production'), /VITE_API_BASE_URL=\/api/);
});

test('vite proxy rewrites /api to /api/v1', () => {
  const content = read('./vite.config.js');
  assert.ok(content.includes("rewrite: (path) => path.replace(/^\\/api/, '/api/v1')"));
});

test('http adapter has explicit direct backend /api/v1 fallback', () => {
  const content = read('./src/api/http.js');
  assert.match(content, /http:\/\/127\.0\.0\.1:8000\/api\/v1/);
});
