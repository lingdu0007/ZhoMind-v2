# ZhoMind-v2

## Minimal Local Run

Run these commands from the repository root:

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` before startup. At minimum, fill the blank required values such as `JWT_SECRET`, `ADMIN_INVITE_CODE`, provider API keys, and any provider-specific base URLs or tokens you actually use.

Start the backend stack with your local ignored runtime config:

```bash
docker compose up -d backend
```

Run the smoke script after the backend is healthy. If `backend/.env` sets `ADMIN_INVITE_CODE`, export the same value through `SMOKE_ADMIN_CODE` first:

```bash
SMOKE_ADMIN_CODE='<same as backend/.env ADMIN_INVITE_CODE>' node frontend/tests/user-path-smoke.mjs
```

## Opt-In Milvus Dense E2E Verification

Use this path only when you want real dense indexing against a local Milvus instance. Default backend tests still isolate dense mode off unless you explicitly enable this opt-in flow.

```bash
docker compose up -d etcd minio milvus
curl -fsS http://127.0.0.1:9091/healthz
cd backend
RUN_MILVUS_E2E=1 MILVUS_URI=http://127.0.0.1:19530 EMBEDDING_API_KEY=test-milvus-e2e-key EMBEDDING_BASE_URL=http://test-milvus-e2e.local/v1 EMBEDDING_MODEL=test-milvus-e2e DENSE_EMBEDDING_DIM=3 uv run pytest -q tests/integration/test_milvus_dense_e2e.py
```
