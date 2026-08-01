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

## Retrieval Evidence Smoke

The Retrieval Smoke uses the existing administrator upload and document-build flow, then calls the existing dense retrieval implementation directly. It does not call a chat model.

Keep the local provider configuration in the ignored `backend/.env`. The smoke requires a non-default `JWT_SECRET`, `ADMIN_INVITE_CODE`, and active Qwen embedding configuration (`EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, and `DENSE_EMBEDDING_DIM`). It never prints or writes those values.

Run it from the repository root:

```bash
./retrieval-evidence smoke
```

The command requires a clean local experiment branch, then starts a unique Compose project with separate ports and runtime volumes for that run. It stops that Compose project on exit, so repeated runs start with independent databases and indexes. The command writes a non-sensitive manifest to `../../evidence/runs/<run-id>/manifest.json`, reporting the document build, live Qwen embedding, the hashed embedding contract identity, Milvus indexing, and proof that the retrieved dense candidate belongs to the just-ingested document. It exits nonzero for any failed acceptance check.

## Cited Generation Smoke

With the same ignored local configuration plus one active Ark provider (`ARK_API_KEY`, `BASE_URL`, and `MODEL`), run:

```bash
./retrieval-evidence generation-smoke
```

This creates a disposable published source, verifies an actual provider answer and both normal and SSE cited-response contracts, and writes only non-sensitive pass/fail fields and citation counts to its manifest. It never write or print provider credentials, prompts, answers, or excerpts. Provider-outage no-fallback behavior is covered by the backend contract tests rather than sending a real question to an intentionally invalid provider.
