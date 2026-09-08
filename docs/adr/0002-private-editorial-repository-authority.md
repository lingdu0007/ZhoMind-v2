# ADR 0002: Private Editorial Repository Authority

Status: Accepted

Date: 2026-09-05

Updated: 2026-09-08

## Context

The product needs a retained editorial authority before Reviewed Release Bundle
intake, Candidate Build, and publication. Existing runtime document rows and
deployment copies cannot prove a reviewed revision, source definition,
role-separation decision, assurance support, or source freshness. Allowing
runtime material to write back into editorial records would reverse that
authority boundary and make later reconstruction unauditable.

## Decision

- Use the existing immutable `canonical_records` and append-only
  `canonical_events` tables as the Private Editorial Repository. Immutable
  `entry`, `editorial_revision`, and `source` records retain the authority
  snapshots; no schema migration is needed because the foundation already
  supports these canonical kinds and record classes.
- Store source definitions once. Record source availability as append-only
  `editorial_source_event/v1` facts and derive current availability from the
  event trail instead of mutating the source record. An Author's availability
  value is a proposal only: a new source begins
  `changed_or_unreachable_awaiting_review`, and an accepted Maintainer must
  append a qualified `verified_usable` fact before it can support review or
  export. That fact must use the source aggregate, bind the exact accepted
  Maintainer event, name a revision that contains the source, and use the
  accepted Maintainer as its recorder. Missing, foreign, malformed, or
  unqualified source trails resolve to `unknown`; T01 records this
  authenticated access decision rather than performing arbitrary runtime URL
  or locator fetches. A non-usable source blocks answer eligibility and export.
  Decisive loss is durable authority evidence; it can move a future published
  entry to Needs Re-review without creating a publication path in this ticket.
- Make the canonical coverage position, assurance level, source tier, source
  access scope, and revision change-kind vocabularies closed Python contract
  enums. Entry validation retains the complete decision schema, role facts,
  source links, and assurance-specific evidence requirements. Wording-only
  revisions normalize title and body whitespace only; punctuation, case,
  semantic tokens, comparison operators, and structured data require material
  review. High-impact requirements are conservatively inferred from authored
  text in supported languages as well as client-supplied claim metadata, so a
  client cannot lower a Claim-Evidence Link requirement by relabeling the
  content or by attaching an unrelated ordinary claim in the same section.
- Derive Author, Approving Reviewer, Maintainer, and Administrator authority
  from active server-side member rows. Authors, Reviewers, and Maintainers may
  work within their entry boundary; an Author or material reviser cannot
  approve that material revision. The assigned Maintainer must append a
  revision-specific responsibility acceptance before review, approval, or
  export. Administrators receive approved exports only and cannot inspect or
  directly modify private editorial records. Candidate publication is a narrow
  server-side exception, not an administrator editorial command: only after
  `CandidatePublicationService` re-verifies one exact frozen approved export
  under the finalization fence and atomically writes exact Candidate and
  Published Knowledge Version identities may `EditorialAuthorityService` append
  machine-derived `candidate_build` and `published` lifecycle events. That path
  exposes no private editorial content, accepts no administrator-selected
  editorial field, and cannot revise a private record.
- Use explicit entry `schema_version` 1 from the first retained editorial
  revision. There is no prior editorial schema to migrate; an incompatible
  future version must provide its own migration before acceptance.
- For Release-Assured entries, resolve every named reference to a retained
  immutable or authoritative canonical record, never a derived record or
  replaceable projection. The frozen acceptance must be an immutable
  `delivery_acceptance_record/v1` that validates under its canonical schema.
  Its current active fact must be an `at_risk`-to-`active` `status_changed`
  event with `checks_verified`, a canonical member recorder, and verification
  attachments matching every passing or carried-forward selected check; its
  retained scope must cover the exact entry, contract, calibration, and named
  gate.
- Reconstruct deterministic, SHA-256-addressed `editorial_export/v1` artifacts
  only from retained authority records and events. Approval and lightweight
  acceptance retain a snapshot bound to the exact entry/revision, trusted
  roles, approval fact, qualified source definitions and availability event
  hashes, Release-Assured record/status hashes where applicable, and the
  approval-time audit cutoff. Only structurally valid lifecycle, maintainer,
  and approval events establish those facts. A new export checks current
  authority facts; historical reconstruction uses its retained approval
  snapshot only. Editorial export, Reviewed Release Bundle manifest/item, and
  frozen Candidate-input integrity hashes use one shared canonical JSON
  serialization: sorted keys, compact separators, ASCII escaping of non-ASCII
  code points, and UTF-8 bytes. Reject credential-shaped material, non-empty secret-bearing
  fields at every structured depth, and automatic-publication instructions.
  The export is not a bundle, does not invoke intake, and cannot write a
  Candidate, published knowledge version, runtime document, or deployment
  copy.
- Let T02 consume an export only by read-only reconstruction from this
  repository. A `reviewed_release_bundle/v1` item must exactly match the
  retained approved artifact and its hash. The manifest source revision equals
  every artifact revision hash, and the item hash covers its stable identity,
  operation, artifact hash, and artifact. Intake accepts no type coercion:
  `schema_version` is a JSON integer rather than a boolean and identities are
  JSON strings. It then rechecks current source and Release-Assured authority
  without appending an export audit event. A failed whole-bundle integrity
  check or immutable identity collision records only a bounded content-free
  audit against a fresh `admission_attempt`, never a rejected bundle, item, or
  build-generation record. Its immutable bundle, item, and build-generation
  records may create a separate recoverable Candidate Build job, but import
  only plans it. An explicit System Administrator dispatch is the only action
  that records durable `dispatched_at`, appends a `dispatched` event with the
  current attempt and administrator `member:` identity, and makes queued work
  eligible for enqueue or startup recovery; an explicit retry appends
  `retry_dispatched` with that same current-attempt authority and establishes
  `cancel_or_await_candidate_build` for its new attempt. Runtime enqueue and
  startup recovery require that append-only evidence for the current queued
  attempt, never mutable `dispatched_at` alone. The proof must use
  `candidate_build_job_event/v1`, the exact queued transition and action
  shape, frozen editorial revision and both input hashes, and a `member` identity
  that still resolves to an active current System Administrator and
  authoritative identity record. Concurrent Candidate-generation
  allocation contention rereads the winner and retries the same immutable
  intake within a bounded attempt budget; exhaustion returns retry-required
  without admitted intake records instead of treating a distinct valid bundle
  as an immutable identity conflict. The bundle advances by append-only `received`, `validating`,
  `validated`, `processing`, and either `completed` or
  `completed_with_rejections` events. Its immutable snapshot retains whether
  any item was rejected, so mixed valid/rejected work is never represented as
  complete batch success. It waits in processing while a valid Candidate job is
  non-terminal. Every
  worker, retry, recovery, derived write, vector call,
  Candidate finalization, and cleanup path reconstructs and re-matches frozen
  inputs before use. Candidate finalization repeats approved-export, source,
  and Release-Assured authority verification after indexing and immediately
  before Candidate persistence while holding shared canonical authority locks
  for the entry, retained sources, and every Release-Assured reference. Source
  availability recorders and delivery-acceptance status writers take the same
  relevant locks, so no authority event can append between final verification
  and Candidate commit. The verifier must provide the finalization authority
  fence; no unlocked fallback is valid. On SQLite, the fence uses
  `BEGIN IMMEDIATE` before re-verification to serialize authority writers
  through Candidate commit. Candidate embeddings use a fingerprint over configuration
  schema, active flag, model, and dimension only, so their Candidate-specific
  collection is separate from normal runtime retrieval and contains no endpoint
  or secret. An inactive frozen configuration has no Candidate vector
  collection; cleanup without its frozen Candidate fingerprint does not query
  or delete any collection and never falls back to the normal retrieval
  fingerprint. Build stages are closed (`queued`, `parsing`, `chunking`,
  `indexing`); terminal status is separate and a worker must prove its exact
  attempt, owner, and unexpired lease before mutating a running job. Candidate
  and legacy document jobs acquire one process-wide bounded build-worker slot
  pool, so separate dispatchers cannot multiply configured concurrency. A newer
  generation supersedes unfinished work and a failed job whose only next action
  is `import_new_bundle`, without erasing historical evidence; unverified or
  failed cleanup remains a durable pending obligation and may
  delete only after a fresh frozen-input match. Neither intake nor the worker
  may write back to private editorial authority, legacy runtime document rows,
  or any published knowledge pointer. A Candidate remains an isolated derived
  result; Candidate inspection, publication, replacement, and withdrawal are
  not T02 responsibilities.
- Retain the approved-artifact `input_sha256` separately from
  `frozen_input_sha256`, which canonically hashes the complete immutable
  Candidate input binding: schema, bundle and item identities and hashes,
  entry/document identities, generation, editorial revision, artifact hash,
  chunk strategy, and embedding configuration. Both hashes appear on every new
  job and dispatch event. Pre-`0018` immutable input records and dispatch
  events remain unchanged: upgrade recomputes the full hash onto the job, and
  runtime accepts a legacy event without that field only when its matching
  immutable input record is likewise pre-hash and every remaining
  current-attempt administrator proof remains exact. This recognizes prior
  authorization without minting one. Expected authority validation failures
  remain item-local; verifier infrastructure or execution failure rolls back
  the whole intake transaction without persisting a false rejected item or
  partial supersession, so the immutable bundle is safely retryable.
- Treat Candidate Build recovery and concurrency as a durable authority
  boundary. Source usability at intake comes only from verifier-reconstructed
  retained authority facts, never an artifact Author's `availability` value.
  Bundle completion locks the bundle aggregate before reconstructing its state
  and child-job statuses. `parsing` validates and reads an approved export
  before `chunking` consumes only parsed data. While external indexing is
  awaited, the worker conditionally renews its exact owner/attempt lease and
  races those heartbeats against the indexing coroutine. A renewal failure or
  replacement cancels and awaits that coroutine before the worker stops without
  a terminal mutation or derived-data cleanup; a recovery owner locks and rechecks the
  expired job, persists the interrupted stale-worker fence with
  `derived_cleanup_pending`, commits it, and only then reconciles matching frozen
  inputs. Candidate job events snapshot frozen editorial source revision, input
  hash, structured failure reason, and allowed next action. Recovery locks and
  rechecks each selected queued, running, or cleanup-pending job immediately
  before mutation. These paths cannot create or change a Published Knowledge
  Version or publication pointer.
- Bind ordinary Pilot retrieval to a versioned authorization boundary. The
  active `retrieval-answer-policy/pilot-v1` uses genuine literal-preserving
  Sparse BM25 (`k1=1.5`, `b=0.75`, depth 20) only after it has constructed the
  authorized pool from current published compatibility rows and a fresh
  per-entry reconstruction of retained Private Editorial Repository authority.
  Compatibility metadata must exactly bind the row to the current entry,
  revision, section-source relationships, assurance, applicability, freshness,
  and supported access scope; it can never grant eligibility by itself.
  Candidate-derived chunks remain excluded. An explicit administrator-only
  Candidate preview verifies the immutable Candidate record, current-attempt
  job, frozen input, bundle/item artifact, and content-hashed chunk bindings
  before producing its diagnostic-only result. It carries Candidate rather than
  publication identity and cannot become answer evidence. Ordinary runtime
  traces retain normalized exclusion reasons without excluded Candidate or
  unpublished chunk identities. BM25 scores order the authorized
  pre-sufficiency pool only; they do not grant eligibility or establish
  sufficiency. The retained
  `retrieval-answer-policy/lexical-heuristic-migration-v1` path is explicitly
  migration/diagnostic and is never represented as Sparse BM25.
- If a later safe revision is not yet published, ordinary retrieval may keep
  using an existing pointed Published Knowledge Version only after it
  reconstructs the exact frozen Candidate and retained revision and
  revalidates current source records. Runtime metadata binds the projection but
  never supplies source, decision, assurance, or applicability authority;
  missing or unusable retained sources exclude the pointed version.

## Consequences

The Private Editorial Repository remains a durable, access-controlled
editorial authority and can prove a revision/export hash from retained data.
T02 consumes it through immutable Reviewed Release Bundle intake and
recoverable Candidate Build records, while T04 publication remains separate.
The bundle verifier rechecks current authority but never backfills or rewrites
private editorial records. Source availability is already authoritative for
fail-closed eligibility, while later publication and maintenance paths must
consume that retained evidence rather than infer it from a runtime copy.
The retrieval boundary now consumes those retained facts before ranking:
ordinary Pilot queries cannot see Candidate content, stale or withdrawn
generations, unavailable or unauthorized sources, expired review grace, known
contradictions, or ineligible assurance. A compatibility-row metadata mismatch
is an exclusion, not a fallback authority source. Candidate inspection is
deliberately not ordinary retrieval: its isolated administrative preview
remains diagnostic-only and cannot supply product answer evidence. This adds
no publication capability and does not decide answer sufficiency.
Candidate finalization serializes its final re-verification and persistence
with source-availability and Release-Assured authority writers through their
shared canonical authority records; this removes the interval in which a
freshly invalidated source or acceptance status could otherwise produce a new
Candidate. The finalization verifier fence is required, and on SQLite it starts
`BEGIN IMMEDIATE` before re-verification so concurrent authority writers remain
outside the interval through Candidate commit. Dispatch evidence also remains fail-closed: it must bind the exact
frozen input and queued transition to an identity that still resolves to an
active System Administrator.
Recovery requeues only queued jobs with durable administrator-dispatch
evidence for their current attempt, never a mutable timestamp alone, appending
`requeued_on_startup` on successful requeue; it treats
missing, expired, and prior-runtime-owned candidate leases as interrupted work
and must reconcile derived data before retry. This preserves the same authority
boundary across process restart without granting recovery any publication
capability. Recovery and runtime enqueue lock and refresh the current persisted
job before deciding eligibility, and recovery locks and rechecks every selected
queued, running, or cleanup-pending job immediately before mutation. It records
a successful requeue only while that current job remains queued or running.
Each append-only Candidate job event snapshots frozen editorial source revision,
input hash, structured failure reason, and allowed next action, so a retry
cannot erase the prior attempt's audit. Stage transitions occur before the work
they name: `parsing` validates and reads the approved export before `chunking`
uses parsed data. External indexing conditionally renews the exact worker
lease and races that heartbeat with the indexing coroutine; a worker that
cannot renew cancels and awaits the coroutine before stopping without terminal
mutation or cleanup, and fenced recovery later owns the interrupted/retryable
transition and any matching-input reconciliation after its fence is committed.
An administrator cancellation request succeeds only after worker control
confirms receipt. A false no-task result or an exception is a durable failed
Candidate with a structured reason and pending derived-data reconciliation,
never a successful cancellation.
Recovery must never let a stale worker or an input mismatch turn a Candidate
into a terminal success, erase unverified derived data, or move a Published
Knowledge Version. Superseded unfinished work carries an explicit cleanup
obligation until a matching-input reconciliation has succeeded.
Release-Assured references fail closed unless their canonical records are
appropriate authority records and the frozen delivery-acceptance record
actively covers the exact editorial authority.
