# Ticket Execution Protocol

Status: Required engineering workflow for implementation tickets.

This protocol turns a ticket into a reviewable sequence of owned work,
behavioral evidence, and a final diff. `AGENTS.md` contains the hard rules.
This document defines the detailed sequence. The ticket and its parent product
documents remain authoritative for scope and behavior.

## 1. Preflight And Ownership

Work from the repository root. Read `AGENTS.md`, the assigned ticket, every
parent PRD requirement named by the ticket, and the applicable contract and
ADR sections before changing code.

Confirm all numbered dependencies are resolved in the tracker and that the
current source history contains their accepted implementation commits when the
ticket names them. Do not substitute a status label for source-history proof.

Record:

```bash
BASE_SHA="$(git rev-parse HEAD)"
git branch --show-current
git status --short
```

Treat the initial status as the pre-existing worktree manifest. Preserve those
changes, do not stage unrelated paths, and do not use destructive Git commands.

The ticket must record the active owner or session, `BASE_SHA`, branch, and
start time under `## Comments`. If the ticket is already claimed and ownership
cannot be reconciled to the current task, stop without editing.

Run only the minimum environment and baseline smoke needed to establish that
the affected test runners can start. Record pre-existing failures separately;
do not repair or silently absorb them unless the ticket owns them.

## 2. Acceptance-To-Seam Mapping

Before writing a test, map every acceptance criterion:

| Acceptance criterion | Approved public seam | First failing test | Required negative case | Final evidence |
| --- | --- | --- | --- | --- |
| Ticket text | User- or caller-visible interface | Test path and name | Denial, failure, or boundary behavior | Command and result |

Tests observe behavior through public interfaces rather than private methods,
mutable diagnostics, or implementation-only database queries.

The repository owner has pre-approved these default seams:

- Pure domain or contract interface for deterministic state, identity,
  normalization, policy, and validation behavior.
- Authenticated HTTP interface for server-derived authority, ownership,
  persistence, and product outcomes.
- SSE terminal-event interface for streaming progress, interruption, and final
  result projection.
- Persisted reload and history interface for retained identity and redaction.
- Frontend presentation adapter or store interface for canonical projection
  without semantic reinterpretation.
- Disposable-API browser journey for user-visible workflow, authorization,
  failure, and responsive-layout behavior.
- Alembic upgrade interface for migration behavior, compatibility, and database
  enforcement when the ticket changes persistence.

Use the smallest applicable subset. The agent does not need to ask the user to
approve these seams again. Ask only when the ticket requires a new seam outside
this list or when two listed seams would assert contradictory product behavior.

Prefer the smallest set of seams that proves the complete behavior. For a
cross-surface contract, use shared contract vectors and verify every required
adapter rather than independently recreating expected semantics in each
language or transport.

## 3. Review-Risk Matrix

For protected, persisted, asynchronous, or cross-surface changes, answer each
applicable question before implementation and convert the answer into a test:

| Risk | Required question |
| --- | --- |
| Authority | Which current persisted fact authorizes this action or projection? |
| Identity | Which immutable identities, hashes, versions, attempts, and configurations must match? |
| Fail closed | What happens when a required fact is missing, stale, malformed, unauthorized, or contradictory? |
| State | Which transitions are legal, terminal, retryable, or forbidden? |
| Race and retry | What happens after concurrent change, cancellation, duplicate delivery, lease loss, or retry? |
| Compatibility | Can legacy or partially migrated data accidentally grant current authority? |
| Privacy | Can content, credentials, another member's data, or administrator-only diagnostics cross the seam? |
| Projection | Do normal, streaming, persisted, reloaded, historical, and frontend views preserve one semantic result? |
| Scope | Does the implementation avoid publication, provider, retention, recovery, or other later-ticket behavior? |

Happy-path coverage does not close this matrix. A risk is closed only by a
deterministic assertion at an approved seam or by an explicit, correctly
classified deployment-only obligation.

### Automatic Risk Profiles

Apply the following profiles automatically when the ticket requirements or
changed surfaces match them. The worker prompt does not need to repeat these
checks.

**Canonical authority or persistence**

- Derive authority from current persisted facts, never client flags, mutable
  diagnostics, titles, filenames, environment aliases, or `latest`.
- Bind exact aggregate identity, revision, version, hash, attempt, actor,
  configuration, and status event wherever they affect authorization.
- Reject stale, malformed, foreign, partially migrated, or contradictory facts.
- Cover duplicate submission, retry, cancellation, concurrent change, partial
  transaction failure, and finalization races.
- Use an additive Alembic revision and test the supported upgrade path.

**Retrieval, evidence, answer execution, chat, or feedback**

- Preserve the closed Answer Execution Outcome and keep execution states
  separate from completed outcomes.
- Do not re-retrieve, re-rank, reselect evidence, recut snapshots, or infer an
  outcome from text, source count, flags, or legacy diagnostics in an adapter.
- Preserve the same question conditions, outcome, evidence-set identity,
  snapshot identity, citation identity, and knowledge version across normal
  HTTP, SSE, persistence, reload, history, and feedback.
- Prevent a stale or interrupted stream from contaminating another conversation
  or a later execution.
- Insufficient Evidence invokes no provider and has no supporting citation or
  generated recommendation.
- Generation Unavailable may expose only the already frozen evidence preview
  and cannot present a Decision Summary or Knowledge Answer Citation.
- Candidate, draft, withdrawn excerpt, inaccessible source, administrator-only
  diagnostics, and another member's private content remain excluded.
- Use shared Python and JavaScript contract vectors for normalization, Unicode,
  punctuation, reason codes, retry limits, and presentation semantics.

**Frontend or browser workflow**

- Represent loading, empty, failure, stopped, throttled, rejected, and completed
  states explicitly without fabricating successful content.
- Wait on observable readiness rather than heading appearance or arbitrary
  sleeps.
- Verify desktop and narrow-screen layouts without overlap, clipped controls,
  inaccessible actions, or state-dependent layout shifts.
- Register browser, server, and build-process cleanup before every setup or
  launch failure window.
- Use browser journeys for the complete authenticated workflow rather than
  treating an isolated store, API call, or diagnostic page as product
  acceptance.

**Publication, recovery, queue, or background work**

- Separate admission, build completion, inspection, acceptance, publication,
  withdrawal, and recovery authority.
- Preserve prior accepted state after failed replacement, partial verification,
  worker interruption, lease loss, or derived-data cleanup failure.
- Give every failed or interrupted operation a structured terminal or retryable
  state and an allowed next action.
- Revalidate immutable inputs and current authority before irreversible
  finalization.

**Privacy, safety, or operations**

- Prove cross-member and administrator denial through authenticated interfaces.
- Keep questions, answers, excerpts, credentials, tokens, private descriptions,
  and full request bodies out of diagnostics, logs, audit, and public evidence.
- Verify deletion, expiry, redaction, and retained de-identified identity
  separately.
- Treat content-bearing telemetry or unauthorized projection as a blocking
  integrity defect.

## 4. Vertical TDD Slices

Implement one tracer slice at a time:

1. Add one failing behavioral test at an approved seam.
2. Confirm that it fails for the intended missing behavior.
3. Add the minimum implementation needed to pass.
4. Run the affected focused tests.
5. Refine the acceptance map and risk matrix with what the slice revealed.
6. Continue with the next acceptance criterion or negative case.

Do not write all tests first and then all implementation. Do not make a broad
refactor part of the red-green loop. Keep adapters thin and preserve the
canonical authority module as the single source of semantics.

Use checkpoints when the ticket crosses several layers:

- Contract checkpoint: identity, authority, state, failure, and compatibility.
- Product-path checkpoint: authenticated interface and required negative cases.
- Projection checkpoint: transports, persistence, history, and frontend render
  the same retained result.

Record checkpoint commands and results in the ticket while the work is fresh.

## 5. Verification Ladder

Verification proceeds from narrow to broad:

1. Environment and baseline smoke.
2. Focused unit or contract tests for the current slice.
3. Affected integration tests.
4. Authenticated product-path and browser journeys, including negative cases.
5. Static checks and production build for changed surfaces.
6. The applicable deterministic repository gate from `README.md`.
7. Separately commissioned deployment-only checks.

Do not repeatedly run the complete repository gate during early red-green
cycles. Run it after focused and product-path convergence, and again only when
a later fix can affect it.

Real embedding, Milvus, provider, PostgreSQL deployment, ingress, Compose,
persistent-stack, load, and production acceptance remain deployment-only unless
the ticket explicitly commissions them. A skipped or unavailable check records
a limit, not a pass.

## 6. Provisional Commit And Dual Review

After focused, integration, and product-path checks pass:

1. Stage only ticket-owned paths.
2. Create one provisional focused commit.
3. Verify that `git diff BASE_SHA...HEAD` contains the intended ticket diff.
4. Run the repository code-review workflow with both the standards reviewer and
   specification reviewer against `BASE_SHA...HEAD`.
5. Classify every finding as blocking, actionable non-blocking, or rejected
   with a documented reason.
6. For every behavioral finding, first add a regression that fails on the
   reviewed commit, then repair the behavior.
7. Rerun affected focused and product-path checks and amend the provisional
   commit.
8. Obtain fresh standards and specification reviews of the amended diff.

Do not replace a missing specialist reviewer with a generic reviewer. If a
required reviewer is unavailable, keep the ticket non-terminal or use the
tracker's blocked status with the exact required action.

After review convergence, run the final deterministic gate. If that gate
requires a code or behavioral-test change, amend and rerun both review axes.
The final `HEAD` must be the exact diff reviewed without blocking findings.

## 7. Ticket Closure

Before resolving the ticket:

- Check each acceptance criterion only when its mapped evidence exists.
- Record exact verification commands and results.
- Record review findings, regression tests, fixes, and fresh re-review results.
- Record the final commit and confirm the reviewed diff still matches `HEAD`.
- Separate local deterministic evidence from deployment-only limits.
- Record residual risks without presenting them as accepted behavior.
- Confirm no unrelated paths were staged or committed.

A ticket is not resolved when a criterion is only inferred from code, a
required reviewer is unavailable, a blocking finding remains, the final diff
has not been re-reviewed, or a remote-only check is being represented as a
local pass.

## 8. Worker Prompt Shape

A normal per-ticket worker prompt should be one short paragraph. The ticket's
dependency, requirement coverage, acceptance criteria, and exclusions are read
from the tracker; the default seams, risk profiles, TDD loop, verification
ladder, review sequence, language, and closure rules come from `AGENTS.md` and
this protocol.

Use this default shape:

```text
在 /home/lingdu/workspace/agent/Zhomind/sources/ZhoMind-v2 实施 Ticket <NN>。
读取并严格遵循仓库 AGENTS.md、docs/agents/ticket-execution.md、对应 ticket
及其引用的 PRD、contract 与 ADR。确认依赖和归属后，按纵向 TDD 完成全部验收项，
执行适用的产品路径与确定性验证，完成 standards/spec 双轨 review 和 re-review，
更新 ticket 证据，并提交仅包含本票的聚焦变更。全程用中文汇报，不扩大 ticket 范围。
```

Add extra prompt text only for information that does not already exist in the
ticket or repository policy, such as a user-supplied dependency commit, a
temporary external environment, a newly approved seam, or an explicit scope
override. Do not copy long-lived product contracts, risk checklists, or the
deterministic command list into each prompt.
