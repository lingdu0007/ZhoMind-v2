# ADR 0006: Version-Scoped Publication Withdrawal

Status: Accepted

Date: 2026-09-08

Supersedes: ADR 0004's completion-time tombstone redaction rule only.
Extends ADR 0002's narrow administrator publication authority to explicit
withdrawal of an exact current Published Knowledge Version.

## Context

Document deletion cannot prove a publication withdrawal's actor, reason,
version, or impact. Document-wide historical redaction also erases the
distinction between a superseded predecessor and the withdrawn current version.
A replacement under editorial review must not prevent emergency containment
of the version that is still published.

## Decision

- Only an active server-derived administrator may withdraw an exact current
  publication. Revalidate its immutable publication, runtime pointer, entry,
  generation and configuration before appending `publication_withdrawal/v1`.
  Retain the original event identity, actor, time, bounded reason and trigger,
  affected entry-version scope and predecessor identity. Repetition replays
  the original event; a changed reason or trigger is a conflict.
- Published and Needs Re-review versions can be withdrawn. Editorial Review
  and Candidate Build may also terminate in Withdrawn when the administrator
  proves an existing current publication: those states can describe an
  unpublished successor. These transitions do not authorize draft deletion
  or withdrawal of a never-published entry.
- Serialize containment with publication and answer finalization. Persist the
  withdrawal, retrieval tombstone, invalidated successor jobs and affected
  acceptance suspension before attempting history or vector cleanup. A failed
  cleanup cannot roll back an effective withdrawal.
  Candidate completion takes canonical authority locks before the job lock,
  matching withdrawal, and rechecks its lease and immutable input under those
  locks. It cannot retain a job lock while waiting for withdrawal's authority
  fence.
- Record cleanup as append-only `withdrawal_reconciliation/v1` events.
  Explicit retry revalidates each frozen generation and embedding fingerprint,
  cleans its derived data and clears only verified pending job obligations.
  An unavailable backend leaves a suspended, retryable obligation with bounded
  diagnostics. Neither retry nor recovery changes a publication pointer or
  makes withdrawn content eligible.
  This includes the exact runtime document generation and stored fingerprint
  produced by dense backfill, not only Candidate vector addresses. Backfill
  locks and revalidates the current document and withdrawal authority before
  writing, so restoring a compatibility tombstone cannot recreate these assets.
  A fingerprint switch must clean the previous address before replacing its
  readiness binding. Reconciliation reads validate their audit/scope binding
  and cannot accept a completed event while known derived obligations remain.
  Before any runtime backfill write, persist its exact publication, document,
  generation and fingerprint in append-only `runtime_dense_cleanup/v1`
  records. Reacquire the document fence and recheck withdrawal after that
  commit. Lost responses, readiness commit failures and cancellation retain
  these addresses. Reconciliation records each target's verified deletion in
  the same transaction as its completion, so retries cannot lose an address.
  Runtime rows cannot downgrade a known canonical publication to legacy.
  Cleanup also requires a persisted writer-exit proof: the Milvus adapter
  drains its synchronous upsert before propagating a verified cancellation.
  An unverified cancellation or process loss leaves the address pending and
  reconciliation suspended until writer termination can be established;
  the mere return of a delete request is not that proof.
  Successor Candidates obey the same rule. Every attempt that reached indexing
  must have its exact `candidate_writer_exit/v1` proof before its derived-data
  obligation can clear. Cancellation acknowledgement alone is insufficient.
  Workers record their exit even after withdrawal invalidates their lease, or
  when they exit before starting the external write.
  For successful Candidates built before this proof existed, the exact
  append-only completion event and matching immutable Candidate record can
  jointly prove that attempt's indexing returned. Validate their attempt,
  frozen input and complete address/configuration binding; do not synthesize
  a worker-exit event. Mutable terminal status, failed/canceled attempts,
  missing records and contradictory facts cannot substitute for this proof.
  Pre-0018 records may omit the frozen-input hash only when the verified input
  loader identifies that exact legacy format and recomputes the complete
  binding. Normalize the omitted hash on read; retain the original records.
  A wrong present hash or a missing hash on a modern record remains invalid.
  Runtime backfill also records known no-write exits throughout preparation
  after intent registration, including lock/chunk-read failures. A failed
  intent transaction with no write creates no orphan settled event.
  Startup recovery applies the same gate to withdrawn entries. Explicit
  reconciliation and its completed projection recheck all affected unpublished
  jobs, not just jobs whose mutable cleanup-pending flag is still set.
  Historical published jobs are also identified through immutable
  publication-to-Candidate-to-build-generation bindings when their mutable
  Candidate index is missing. New bundle intake shares the entry fence and
  rejects build admission for a withdrawn entry, even with a retained approved
  export; no-op/proposed-withdrawal records do not create new build jobs.
- An answer not yet completed fails if its frozen evidence was withdrawn.
  It cannot become a new supported answer with a withdrawal annotation.
  HTTP and SSE report the retained failure, not generated answer content.
- Already completed private answers retain their evidence-set, item,
  snapshot, citation and publication identities. Append version-scoped
  redaction events and join authoritative withdrawal facts on every final
  normal-response, history or pending-stream projection, including after a
  redaction-write failure or a withdrawal committed after answer completion.
  Show a withdrawal notice without an excerpt or openable current source.
  Predecessor history is not blanket-redacted.
  All projections validate the complete withdrawal shape, audit bindings and
  immutable publication lineage. Missing or contradictory facts fail closed;
  they are never interpreted as an unaffected source or acceptance scope.
- Existing acceptance records bind their original scope. Withdrawal appends
  suspension for active or at-risk affected records; their projection also
  excludes every withdrawn bound entry/version, including additional
  withdrawals after suspension. Unrelated records remain active. Cleanup is
  not reacceptance and cannot reactivate the withdrawn scope.
- Legacy single and batch document deletion reject published versions.
  Unpublished draft deletion remains a distinct operation.

## Consequences

Withdrawal is a durable containment action, not physical deletion of
publication lineage. Immutable publication and original completed results
remain retained while the user projection is redacted. No new database schema
is required: canonical append-only events and the existing execution-redaction
event stream carry the new records.

Real vector backend, PostgreSQL concurrency, persistent-stack recovery and
deployment acceptance remain separate obligations from deterministic local
tests. This decision does not activate a provider, retrieval profile, new
retention policy or general rollback interface.
