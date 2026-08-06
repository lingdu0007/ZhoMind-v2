# ZhoMind-v2

## Deterministic PR Gate

Every pull request runs the public gate in `.github/workflows/pr-gate.yml`. A fresh checkout reproduces it from the committed locks (`backend/uv.lock` and `frontend/package-lock.json`) without any provider credentials; the live-Milvus dense E2E test remains opt-in and is reported as an explicit skip, never as live evidence.

Run the same checks locally (backend checks require the committed lock):

```bash
# Backend: locked install, lint, types, complete deterministic suite
cd backend
uv sync --frozen
uv run ruff check .
uv run pyright
uv run pytest -q -ra

# Frontend: unit tests, production build, disposable-API browser acceptance
cd ../frontend
npm ci
npm run test:unit
npm run build
npx playwright install chromium
npm run test:browser

# Repository-wide: bilingual parity, bundle validation, and secret scan (pure Python 3, stdlib only)
cd ..
python3 scripts/check-docs-parity.py
python3 scripts/validate-evidence-bundle.py --all
python3 scripts/scan-secrets.py
```

The browser journeys spawn the real disposable FastAPI acceptance app through `uv run --no-sync`, so keep `backend` synced before running them.

## Public Evidence Bundle

The Portfolio Release publishes one versioned, non-sensitive Public Evidence Bundle. The contract lives in `public-evidence/contract/` (manifest and typed section schemas, bilingual contract document); `public-evidence/example/` is the minimal complete example. The deterministic validator `scripts/validate-evidence-bundle.py` enforces the field allowlists, provenance cross-references, artifact hashes, bilingual report parity, and sensitive-shape exclusion in the PR Gate. See `public-evidence/contract/README.md` for the full contract.

## Minimal Local Run

Run these commands from the repository root:

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` before startup. At minimum, fill the blank required values such as `JWT_SECRET`, Bootstrap Administrator credentials, provider API keys, and any provider-specific base URLs or tokens you actually use.

Start the backend stack with your local ignored runtime config:

```bash
docker compose up -d backend
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

Keep the local provider configuration in the ignored `backend/.env`. The smoke requires a non-default `JWT_SECRET`, Bootstrap Administrator credentials, and active Qwen embedding configuration (`EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, and `DENSE_EMBEDDING_DIM`). It never prints or writes those values.

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

This creates a disposable published source, verifies one `evidence_gated_answer` through authenticated normal chat and SSE, then proves the same citation set in conversation history. Direct retrieval remains a separate health diagnostic and is not used as a citation oracle. The command writes only non-sensitive pass/fail fields and citation counts to its manifest; it never writes or prints provider credentials, prompts, answers, or excerpts. Provider-outage no-fallback behavior is covered by the backend contract tests rather than sending a real question to an intentionally invalid provider.
