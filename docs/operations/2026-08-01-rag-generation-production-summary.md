# RAG Generation Production Diagnosis and Smoke-Test Summary

**Recorded:** 2026-08-01
**Production revision verified:** `36f6e1cb57c65130ba5c0efa8d18f0ff625e4643`

This is an operational post-mortem and runbook for the cited RAG generation-loop acceptance. It deliberately records configuration *state*, never `.env` values, credentials, prompts, answers, or source excerpts.

## Outcome

The real production acceptance passed for the complete approved-provider loop:

1. Bootstrap Administrator authentication.
2. Live embedding, Milvus indexing, retrieval of the newly published source.
3. Normal chat response with a cited published source.
4. SSE chat response with the same cited-source contract.
5. Public HTTPS health and static frontend delivery through Caddy.

The final production manifest reports `outcome=passed` for `retrieval-evidence generation-smoke`, with both `normal_contract` and `stream_contract` passed. Manifests intentionally contain only source revision, non-sensitive contract outcomes, counts, and normalized failure information.

## Incident: approved Ark generation timed out

### User-visible symptom

The original real-provider generation acceptance raised `TimeoutError` after 45 seconds, preventing proof of normal cited chat and SSE completion. Retrieval, authentication, live embedding, and indexing had already passed.

### Feedback loop used

The red-capable loop was the real-provider compose command:

```bash
python -m app.retrieval_evidence generation-smoke \
  --base-url http://backend:8000 \
  --output-dir /evidence \
  --source-revision <revision> \
  --run-id <unique-run-id>
```

It is invoked by [`deploy/production/acceptance.sh`](../../deploy/production/acceptance.sh) only after the service, HTTPS, Bootstrap Administrator, and retrieval checks pass. The command uploads and explicitly publishes a dedicated source, exercises the deployed retrieval and approved-provider paths, and exits nonzero with a named failed check when the exact acceptance contract fails.

To minimize the failure, direct short provider requests, the provider adapter, and a RAG-graph call were run independently. Those calls completed normally. Replaying the original composed prompt containing the synthetic `retrieval-evidence-<run-id>` sentinel reproduced the timeout; changing only the fixture wording removed it. A first generic natural-language replacement then exposed a separate retrieval miss in the existing corpus, so the final fixture was made both natural-language and retrieval-distinctive.

### Tested hypotheses

| Hypothesis | Prediction | Result |
| --- | --- | --- |
| Network, credentials, or provider-wide outage | Direct short provider calls also fail or time out. | Falsified. |
| Ark adapter or basic RAG graph fault | The adapter/graph call fails for ordinary input. | Falsified. |
| Synthetic machine-sentinel prompt triggers a provider long tail | The old composed sentinel request times out while a natural equivalent succeeds. | Confirmed. |
| Natural fixture is insufficiently distinctive for retrieval | A generic question does not retrieve the just-published source. | Confirmed and corrected. |

### Fix and regression coverage

The smoke fixture now publishes a natural, distinctive Chinese fact about the “蓝松石版本” and asks a natural question about that fact. The machine identifier remains in the uploaded fixture filename/content for run isolation, but is not sent as the model question.

[`backend/tests/unit/test_retrieval_evidence.py`](../../backend/tests/unit/test_retrieval_evidence.py) locks down that boundary and the SSE citation contract:

- the chat question is natural language and does not contain the synthetic sentinel;
- the fixture contains the distinctive fact used for retrieval;
- a stream that omits the cited source fails with `STREAM_CITATION_MISSING`;
- non-sensitive manifests do not include credentials, provider URLs, answers, or excerpts.

Validation completed after the fix:

```text
backend test suite: 191 passed, 1 skipped
production generation-smoke: passed
```

No temporary debug logging was added; a source scan found no `[DEBUG-...]` instrumentation.

## Smoke-test operating guide

### Retrieval Smoke

Run `./retrieval-evidence smoke` when validating the ingestion and retrieval half only. It proves health, Bootstrap Administrator login, document build, explicit publication, live embedding, Milvus indexing, and retrieval of the just-ingested published document. It does **not** call the chat provider.

### Generation Smoke

Run `./retrieval-evidence generation-smoke` when the active approved provider also needs verification. It includes every Retrieval Smoke check, then validates the normal chat and SSE evidence-summary contracts. It is the correct production signal for a cited answer; a provider HTTP ping alone is not equivalent.

### Production acceptance

Use the server-side wrapper rather than copying runtime values into a shell:

```bash
ssh -i "$HOME/.ssh/zhomind_ops_ed25519" \
  -o BatchMode=yes -o StrictHostKeyChecking=yes \
  ops@45.207.207.65 \
  'sudo -n env \
    DEPLOY_APP_DIR=/opt/zhomind-v2 \
    DEPLOY_CADDY_SITE_ADDRESS=zhomind.kedayoung.cc \
    SOURCE_REVISION=<deployed-revision> \
    bash /opt/zhomind-v2/source/deploy/production/acceptance.sh'
```

The wrapper reads the root-owned server runtime file itself. Do not use `cat`, `printenv`, shell tracing, or compose configuration rendering against that file. Evidence manifests are written under `/opt/zhomind-v2/evidence/`; inspect only their pass/fail fields and never copy source content or configuration values into issue comments.

Each smoke run creates and publishes an acceptance source in the running knowledge base. Treat it as operational test data: use an identifiable run ID, retain the manifest for traceability, and remove or withdraw test content through the normal administrator workflow when it is no longer needed.

## Server access and verified configuration state

### Connection

| Item | Verified value or practice |
| --- | --- |
| Host | `45.207.207.65` (`zhomind.kedayoung.cc`) |
| SSH account | `ops` |
| Local identity path | `$HOME/.ssh/zhomind_ops_ed25519` |
| Host-key verification | `StrictHostKeyChecking=yes`; expected ED25519 SHA-256 fingerprint: `SHA256:gEWi26JW7nC5PLm8012eRHnlouqHoFFu8NH4aY9BeYs` |
| Privilege boundary | Use `sudo -n` for `/opt/zhomind-v2` and Compose operations; do not weaken file permissions. |

For a routine non-sensitive status check:

```bash
ssh -i "$HOME/.ssh/zhomind_ops_ed25519" \
  -o BatchMode=yes -o StrictHostKeyChecking=yes \
  ops@45.207.207.65 \
  'sudo -n git -C /opt/zhomind-v2/source rev-parse HEAD && \
   sudo -n env DEPLOY_CADDY_SITE_ADDRESS=zhomind.kedayoung.cc \
     docker compose --env-file /opt/zhomind-v2/.env \
     -f /opt/zhomind-v2/source/deploy/production/compose.yml \
     ps --status running --services'
```

### Observed state on 2026-08-01

- Application directory: `/opt/zhomind-v2`.
- Runtime file: `/opt/zhomind-v2/.env`, owned by `root:root`, mode `0600`; its contents were not read into this report.
- Deployed source revision: `36f6e1cb57c65130ba5c0efa8d18f0ff625e4643`.
- Running services: `postgres`, `redis`, `etcd`, `minio`, `milvus`, `backend`, and `caddy`.
- Network shape: Caddy is the public HTTPS edge; Postgres, Redis, etcd, MinIO, Milvus, and backend communicate on internal Compose networks. Backend has the controlled egress path needed for approved external providers.
- Required runtime configuration presence was checked without exposing values: bootstrap administration, settings encryption, database/object-store credentials, Ark provider, embedding provider, embedding dimension, and primary-provider selection are complete.
- The active primary generation provider is the single approved Ark provider. The evidence gate remains production-enabled by the cited generation-loop contract; there is no provider fallback chain.
- System-settings draft/application controls and the document worker are configured. Tool use is configured but must remain governed by the application contract rather than inferred from this report.
- `https://zhomind.kedayoung.cc/api/health` returned a passing status.

## Operational boundaries and follow-up

- The first-release capacity ADR sets a 4-vCPU/3.8-GiB host boundary, at most 25 active members, five concurrent Q&A requests, 500 published versions, 25-MiB uploads, and one document-build worker. It targets P95 end-to-end chat latency at or below 12 seconds; a real provider call can still fail or exceed that objective and must fail closed.
- The provider-outage no-fallback behavior is covered by backend contract tests. Do not prove it by sending production source material to an invalid or alternate provider.
- The publish script's final browser acceptance needs local `node`; on the recorded release that local optional step was unavailable. Server-side Caddy frontend delivery, authentication, Retrieval Smoke, Generation Smoke, and public health all passed. Restore local Node/Playwright before relying on that final browser step for a future deployment.
