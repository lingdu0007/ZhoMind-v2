# Canonical Product Contracts

Status: normative product contract; additive foundation for tickets 13 through 19

## Purpose

The canonical contract layer gives later product slices stable identities,
closed state vocabularies, immutable records, and append-only events without
reinterpreting or deleting existing runtime rows. It is intentionally separate
from the legacy `documents`, `document_jobs`, `chat_messages`, and feedback
tables during migration.

This document is normative for the product rules it defines. Tickets must
preserve those rules or update the governing ADR and this English canonical
document with its Chinese mirror in the same change.

## Identity

`StableIdentity` is a `(kind, value)` pair. The following kinds are reserved
(events use the separate `event` kind):
`member`, `team_invitation`, `admission_attempt`, `entry`, `source`, `bundle`,
`bundle_item`, `build_generation`, `candidate`,
`published_knowledge_version`, `answer_execution`, `evidence_set`,
`evidence_snapshot`, `maintenance_item`, `delivery_acceptance_record`,
`collection`, `capability`, `configuration`, `concurrency`, `corpus`,
`data_boundary`, `deployment`, `editorial_revision`, `embedding_profile`,
`host`, `migration`, `objective`, `product_path`, `product_revision`,
`prompt_envelope`, `provider_route`, `public_claim`, `retrieval_profile`, and
`user_boundary`. Identity values are immutable. A title, filename, generation
number, score, environment label, or current `latest` pointer is not an
identity.

## State And Events

The Python contract module defines the PRD state vocabularies for entry
lifecycle, bundle intake, build stages and terminal statuses, answer execution
and outcomes, maintenance, and acceptance. `validate_transition` rejects
illegal transitions before persistence. `canonical_records` are immutable
snapshots classified as `immutable`, `append_only`, `authoritative`, `derived`,
or `replaceable_projection`. `canonical_events` are append-only and record
state changes without rewriting history.

No projection sets `answer_eligible`, editorial approval, source authorization,
publication eligibility, or acceptance to true when the legacy row does not
prove it. Missing legacy identity, assurance, source, bundle, condition, and
evidence fields remain explicit `unknown` values.

## Private Editorial Authority

The Private Editorial Repository is the sole authority for an Engineering
Decision Entry before later bundle intake. It uses immutable authoritative
canonical `entry`, `editorial_revision`, and `source` records plus append-only
canonical events; it does not write `documents`, Candidate records, Published
Knowledge Versions, runtime caches, or deployment copies. An immutable
`entry_id`, each `editorial_revision` identity, and each source identity retain
their original meaning forever. A source record retains its immutable
definition; current source availability is reconstructed from its append-only
`editorial_source_event/v1` trail and is never written back into that record.
A qualifying availability fact must use the `source` aggregate, name the
source's retained entry and revision, bind the exact Maintainer
responsibility-acceptance event, and be recorded by that accepted Maintainer.
The source must be part of the named revision. A foreign, malformed, or
unqualified event cannot replace a prior fact; it and a missing trail resolve
availability to `unknown`.

The canonical vocabulary contains exactly eight coverage positions:
`rag_source_admission_and_chunking`,
`sparse_dense_hybrid_and_reranking_choices`,
`evidence_sufficiency_refusal_and_acceptance`,
`tools_and_mcp_permissions_and_failure_behavior`,
`agent_context_state_and_memory`,
`orchestration_retry_human_intervention_and_side_effects`,
`provider_failure_and_observability`, and
`prompt_injection_isolation_and_security`. Assurance levels are
`source_grounded`, `claim_linked`, and `release_assured`. Source tiers are
`primary_evidence_source`, `reproducible_engineering_evidence`,
`secondary_discovery_source`, and `bounded_internal_case`; access scopes are
`public` and `controlled_internal`; revision change kinds are `material` and
`wording_only`.

An editorial revision retains title, coverage position, assurance, author,
assigned Approving Reviewer, accountable Domain Knowledge Maintainer, review
date, applicable versions and structured conditions, freshness triggers,
source metadata, chunk strategy, supported and Boundary acceptance material,
the required authored decision sections, section-source relationships, claims,
and optional replacement or supersession relationships. Public sources require
a sanitized canonical HTTPS URL; controlled sources require a sanitized
`controlled://` locator. Sources that are unavailable, inaccessible, secretly
redirected, or tier-ineligible cannot support review or export.

Server-derived active non-administrator member identities are the authority for
editorial roles. The Author, Approving Reviewer, and Maintainer are retained as
separate role facts. An Author or material reviser cannot approve the same
material revision; a Reviewer may edit, but a material reviewer-reviser must
assign a distinct reviewer. The assigned Maintainer must append an explicit
responsibility-acceptance event for each revision before review, approval, or
export. A System Administrator cannot inspect or modify the Private Editorial
Repository and may receive only an approved export.

The private lifecycle writes `draft`, `evidence_collected`, and
`editorial_review` events. Evidence collection and review require a complete
revision. Material changes create a new immutable revision and require
Editorial Review; wording-only changes require an approved base revision, a
distinct lightweight acceptance, and may normalize title and authored-body
whitespace only. Any punctuation, case, token, comparison/operator, or
structured-authority change is material. T01 records no Reviewed Release Bundle,
Candidate Build, publication, replacement, or withdrawal action: those are
owned by later tickets. An availability event for
`unavailable_for_new_evidence` records decisive source loss and, for an
already published entry created by a later product path, transitions that entry
to `needs_re_review`; all other source-state projections fail closed.

Author-provided source availability is only a proposal. A new retained source
has an initial `changed_or_unreachable_awaiting_review` record state, while its
derived availability remains `unknown` until the accepted Maintainer appends a
qualified `verified_usable` fact through the authenticated source-availability
command. T01 deliberately records that access decision rather than fetching
arbitrary public URLs or controlled locators at runtime. A missing, stale, or
unqualified source event reconstructs as `unknown`, so an immutable record's
state cannot make an entry eligible.

Source-Grounded revisions require reviewed section support. Claim-Linked adds
Claim-Evidence Links for material claims. Release-Assured additionally requires
frozen canonical contract, calibration, delivery-acceptance, and named-gate
identities that resolve to retained immutable or authoritative canonical
records, never derived records or replaceable projections. The immutable
delivery-acceptance record must validate as its canonical schema, have a
current `status_changed` `at_risk`-to-`active` event recorded by a canonical
member with the `checks_verified` reason, and retain a complete verification
attachment for every passing or carried-forward selected check. It must bind
this exact entry plus the named contract, calibration, and gate identities in
its retained scope. Material prescriptive, numeric, version, security,
privacy, and other high-impact claims always require Claim-Evidence Links.
High-impact text is conservatively inferred from its statement and authored
decision section in both supported languages as well as client-supplied
`claim_kind` or `material`; a client cannot lower the requirement by
mislabeling the content or by attaching an unrelated ordinary claim in the same
section.

Answer eligibility is a derived projection, never a publication command. It is
false outside Published or bounded Needs Re-review state, and it fails closed
for missing approval, source availability, explicit applicability, known
contradiction, integrity defect, or decisive source loss. Needs Re-review can
remain eligible for at most seven days and only when no blocking reason is
known. Approval and lightweight acceptance retain a deterministic authority
snapshot bound to the exact entry and revision: trusted role assignment,
approval identity and status, exact verified source definitions and availability
events, applicable Release-Assured record/status hashes, and the approval-time
editorial audit cutoff. Only structurally valid lifecycle, maintainer, and
approval events can establish those facts; malformed or foreign events fail
closed. A new administrator export verifies current source and assurance facts
before reconstructing the approved snapshot; historical reconstruction reads
only the retained snapshot, so later runtime or source-state changes cannot
alter an old artifact or its audit trail. The deterministic
`editorial_export/v1` includes stable revision and source identities, roles,
approval, and audit history, and is SHA-256 hashed. Export rejects
credential-shaped values, non-empty secret-bearing fields at every structured
depth, and automatic-publication instructions expressed by either field names
or imperative text. It is not a Reviewed Release Bundle and cannot modify or
automatically publish runtime material. The retained entry schema begins at
explicit `schema_version` 1; no prior editorial schema exists to migrate, and a
future incompatible version must add an explicit migration before it is
accepted.

The editorial export, Reviewed Release Bundle manifest and item, and frozen
Candidate input integrity hashes use one shared canonical JSON serialization:
sorted keys, compact separators, ASCII escaping for non-ASCII code points, and
UTF-8 bytes. This preserves the Private Editorial Repository's established
export byte contract across Unicode values.

## Reviewed Release Bundle Intake And Candidate Build

A System Administrator imports only an immutable
`reviewed_release_bundle/v1` manifest. The manifest retains its stable bundle
identity, schema version, editorial source revision, UTC export time,
bundle-level SHA-256, every bundle-item SHA-256, and the explicit item
operation: `create`, `replace`, `no_op`, or `proposed_withdrawal`. Bundle
integrity is fail-closed before any bundle item, job, or Candidate is
persisted: unsupported schemas, malformed identities or timestamps, hashes,
credentials, automatic-publication instructions, duplicate item identities,
and conflicting entry operations reject the entire bundle. Re-importing the
same bundle identity with the same bundle hash is idempotent; the same identity
with a different hash is a conflict. The manifest editorial source revision
must equal every item's approved-export `revision_sha256`; a bundle-item hash
is the canonical hash of exactly its stable item identity, operation, artifact
hash, and artifact. Schema version must be a JSON integer rather than a
boolean, and every manifest, bundle, and item identity must be a JSON string:
intake never coerces an identity or numeric value into an accepted input. A
whole-bundle integrity rejection or immutable bundle/item identity collision
records only a bounded, content-free canonical import audit against a fresh
immutable `admission_attempt` record and its append-only event; it never leaves
a partial item or job, an unsafe rejected manifest copy, or a rejected
`bundle`, `bundle_item`, or `build_generation` aggregate.
Concurrent Candidate-generation allocation contention rereads the winning
generation and makes a bounded retry of the same immutable intake; exhaustion
returns a retry-required result with no admitted intake records rather than
misclassifying a distinct valid bundle as a bundle-identity conflict.

The bundle and every item are immutable canonical records. A valid
`editorial_export/v1` is verified by reconstructing the exact retained
approved export from the Private Editorial Repository and comparing its
SHA-256 and full artifact. That read-only verification also checks current
source availability and Release-Assured authority facts. Source usability comes
only from those verifier-reconstructed retained authority facts, never an
author-declared `availability` value inside the artifact. Verification creates
no export audit event and never writes private editorial records. Metadata,
approval, role snapshot, source, assurance, access, chunking, or acceptance
failures are item-local `rejected` results with a structured blocking field and
allowed next action; independent valid items remain admitted. `no_op` and
`proposed_withdrawal` remain immutable, visible operation plans and create no
Candidate Build, publication, replacement, or withdrawal side effect.
Verifier infrastructure or execution failure is not an item defect: it aborts
the intake transaction without persisting a bundle, item, job, or false
rejection, so the same immutable bundle can be retried.

The immutable bundle record remains `received`; its intake state is reconstructed
from the append-only `received -> validating -> validated -> processing ->
completed` or `completed_with_rejections` event trail. The immutable bundle
snapshot retains whether intake rejected any individual item. A bundle with
admitted Candidate work stays `processing` until every Candidate Build is
`candidate_ready` or `superseded`; failed, canceled, or interrupted jobs keep
it `processing`. A bundle with no Candidate Build completes its explicit
operation plan immediately after processing only when no item was rejected.
Otherwise, and after mixed valid/rejected Candidate work finishes, it reaches
`completed_with_rejections`, never a complete batch success. An item-local
rejection therefore does not block independent valid siblings or collapse into
a whole-bundle integrity rejection. Bundle completion locks the bundle aggregate
before it reconstructs the current intake state and child-job statuses, which
serializes the terminal aggregate decision.

Every admitted `create` or `replace` item creates one recoverable Candidate
Build job and an immutable `build_generation` input record. They retain bundle,
bundle-item, entry, runtime-document, requested-generation, editorial-source
revision, approved-artifact `input_sha256`, complete
`frozen_input_sha256`, chunk-strategy, and non-secret effective embedding
configuration identities. The frozen-input hash covers the input schema,
bundle and item identities and hashes, entry and document identities,
generation, editorial revision, artifact hash, chunk strategy, and embedding
configuration. Import plans that work but does not enqueue or start it: its
initial allowed next action is `dispatch_candidate_build`. Only an
explicit System Administrator dispatch may set durable `dispatched_at`, append
a current-attempt `dispatched` event recorded by the administrator's
`member:` identity, and enqueue the queued job; an explicit administrator
retry appends `retry_dispatched` with the same current-attempt authority and
establishes `cancel_or_await_candidate_build` for its new attempt. Runtime
enqueue and startup recovery authorize a queued job only from that append-only
administrator event for its current attempt, never from mutable
`dispatched_at` alone. New input records and dispatch events must use
`candidate_build_job_event/v1`, the exact queued transition and action shape,
frozen editorial revision and both input hashes, and a `member` identity that
still resolves to an active current System Administrator and authoritative
identity record. Pre-`0018` immutable inputs and events are never rewritten:
upgrade recomputes and persists the full hash on the job, and runtime may honor
a historical dispatch event without `frozen_input_sha256` only when its
immutable input record also predates that field, the recomputed hash matches
the job, and every remaining exact current-attempt and administrator check
matches. This compatibility recognizes prior explicit authorization; it never
creates authorization from a mutable timestamp or migration state.
Before a worker, retry, recovery, cleanup, or
completion creates, indexes, deletes, or finalizes Candidate-derived data, it
reconstructs the immutable canonical records and checks every mutable job
binding against them; a mismatch fails closed without using or deleting the
Candidate-derived data. Completion repeats approved-export, source, and
Release-Assured authority verification after indexing and immediately before
Candidate persistence while it holds shared canonical authority locks for the
entry, retained sources, and every Release-Assured reference. Source
availability recorders and delivery-acceptance status writers acquire those
same relevant locks, so no authority change can append between final
verification and the Candidate commit. Finalization requires a
verifier-provided authority-fence context and never falls back to an unlocked
check. On SQLite, that context acquires `BEGIN IMMEDIATE` before
re-verification, serializing editorial-authority writers through the Candidate
commit just as record locks do on supported locking databases. Candidate Build
storage is separate from legacy
`documents` and `document_chunks`; its Candidate chunks and dense vectors are
derived data and are unavailable to ordinary retrieval. Candidate and legacy
document workers acquire the same process-wide bounded build-worker capacity,
so additional dispatchers cannot multiply the configured concurrency. The frozen Candidate
embedding configuration contains only configuration schema, active flag, model,
and dimension plus their non-secret fingerprint: it never retains an endpoint,
user info, query parameter, credential, or secret. Candidate vector operations
use that frozen Candidate fingerprint and therefore a Candidate-specific
collection, never the active normal-retrieval collection. The configuration
must still match the active non-secret configuration when a worker starts; a
mismatch fails the job rather than silently building under a different
configuration. Derived-vector cleanup uses the frozen fingerprint rather than
whichever embedding profile is active at cleanup time; a cleanup backend failure
requires reconciliation before retry. An inactive frozen configuration has no
Candidate vector collection: cleanup without a frozen Candidate fingerprint
does not query or delete any collection and never falls back to the active
normal-retrieval fingerprint.

The closed build-stage vocabulary is `queued -> parsing -> chunking ->
indexing`; terminal outcome is a separate job status and terminal state:
`candidate_ready`, `failed`, `canceled`, `interrupted_retryable`, or
`superseded`. A stage never becomes a terminal marker, and a
`candidate_ready` Candidate is still not published. A worker may mutate a
running job only while its exact owned attempt, exact lease owner, and
unexpired lease all match; a stale worker cannot finalize a Candidate, mark a
terminal result, or clean up derived data. A stage transition precedes the
work represented by that stage, so parsing and chunking timing remains
truthful: `parsing` validates and reads the approved export before `chunking`
uses only the parsed artifact. While external indexing is awaited, the worker
renews its exact owner/attempt lease through a conditional persisted update and
races the indexing coroutine against those heartbeats. If renewal fails or
another owner or attempt has replaced it, the worker cancels and awaits its
indexing coroutine before it stops without a terminal mutation or derived-data
cleanup. A recovery owner later locks and rechecks the expired job before it
alone persists its `interrupted_retryable` stale-worker fence with
`derived_cleanup_pending`, commits that fence, and only then reconciles
matching frozen inputs. Parser,
chunking, authority/source, indexing, cancellation, cleanup, and enqueue
failures record a bounded structured reason. Every append-only Candidate job event snapshots
the frozen editorial source revision, input SHA-256, `failure_reason` (when
any), and `allowed_next_action`, so retry's mutable reset cannot erase
prior-attempt diagnostics; a retry blocked by invalid immutable inputs or
unreconciled derived data also appends its blocking event. An administrator
cancellation request succeeds only when worker cancel dispatch confirms
receipt; a false no-task result, like an exception, is a durable
`CANDIDATE_CANCELLATION_REQUEST_FAILED` job with `derived_cleanup_pending` and
reconciliation before retry, never a successful cancellation. Retry either
uses the exact frozen accepted inputs with a new attempt or requires a new
bundle when those immutable inputs no longer verify. Retry, startup requeue,
and runtime enqueue lock and refresh current persisted job facts before
determining eligibility. Each recovery selection locks and rechecks the current
queued, running, or cleanup-pending job immediately before its mutation.
Startup requeues only valid queued work with durable administrator dispatch
evidence for its current attempt, never a mutable dispatch timestamp alone,
appending `requeued_on_startup` only while that refreshed job remains
queued or running; it marks a missing or expired running lease, or one owned by
another runtime instance, interrupted and retryable. Queue-dispatch failure is
itself a durable failed job, not a silent drop.

Admitting a newer generation supersedes an older unfinished or
`candidate_ready` Candidate for the same entry. It also supersedes an older
failed job whose allowed next action is `import_new_bundle`, allowing the
original bundle to reach its terminal aggregate state without retrying
corrupted immutable input. The older immutable Candidate record remains
historical evidence, while its job projection records that it is not
publishable. Superseding unfinished running or interrupted work retains a
durable `derived_cleanup_pending` obligation, as does any already-pending
cleanup on the failed generation. Startup scans terminal jobs with that
obligation and reconciles their Candidate chunks and vectors only after a fresh
frozen-input match; a mismatch preserves those assets and keeps the obligation
recoverable rather than deleting data under an unverified binding.
Intake, retry, recovery, cleanup, supersession, and Candidate completion never
create or change a Published Knowledge Version, a legacy published generation,
or a runtime publication pointer. Candidate inspection, publication,
replacement switching, and withdrawal remain later responsibilities.

## Retrieval Answer Policy And Authorized Candidate Pool

`retrieval-answer-policy/pilot-v1` is the active ordinary-user retrieval
profile. It is genuine `sparse_bm25`, with literal-preserving tokenization,
`k1=1.5`, `b=0.75`, candidate depth `20`, and the versioned
`retrieval-candidate-tie-breaker/v1`. The profile does not accept field boosts:
an unrecorded boost is rejected rather than silently changing the effective
profile. It keeps reranking, lexical-answer anchors, semantic near-duplicate
removal, query expansion, and online LLM sufficiency judging disabled. The
effective profile identity is carried with every retrieval result and trace.

Until a later ticket replaces the legacy runtime projection with canonical
Published Knowledge Versions, Pilot forms its ordinary pre-sufficiency
Candidate Pool only from the compatibility projection of a current legacy
published generation. A chunk is eligible only when its `Document` is not
withdrawn and its generation equals that document's current published
generation. Before ranking, the pool resolves the current Private Editorial
Repository authority for the entry: the current revision, lifecycle
eligibility, exact per-section verified source relationship, assurance,
applicability, freshness, and team-shared access scope. Compatibility metadata
is only a binding to those current facts; missing, malformed, or mismatched
metadata never grants eligibility. Its lifecycle is `published`, or
`needs_re_review` only within the seven-day grace interval; known
contradictions, integrity defects, expired grace, unavailable sources,
unsupported access scopes, and ineligible assurance all fail closed. The pool
preserves entry, revision, publication, section, source, assurance,
applicability, freshness, access, chunk identities, the reviewed
`decision_query`, and the source's untruncated content length. An
authority-qualified candidate may be marked eligible for later evidence
selection, but that pre-sufficiency boundary does not decide sufficiency. The
pool deduplicates exact content and repeated `(entry, section)` pairs
deterministically before it returns at most 20 candidates.

Candidate Build chunks and Candidate records are not members of this ordinary
pool. Candidate preview is an explicit, administrator-only, isolated path:
`GET /reviewed-release-bundles/candidates/{candidate_id}/preview`. It accepts
only an immutable `candidate_ready` Candidate and matching current-attempt
`candidate_ready` build chunks after it reconstructs and verifies the complete
immutable Candidate binding: Candidate record, build job, frozen input,
bundle/item records and artifact, then contiguous content-hashed chunks and
their exact section-source metadata. Preview results are diagnostic-only,
retain Candidate rather than publication identity, and set
`answer_evidence_eligible=false`; they cannot be returned to an ordinary user
or presented as product answer evidence. Ordinary runtime traces retain only
normalized exclusion reasons, never excluded Candidate or unpublished chunk
identities.

BM25 raw scores, including the copied `score` compatibility field, order the
already-authorized pre-sufficiency pool only. A raw score never establishes
eligibility, sufficiency, or a generated-answer decision. Evidence Set
selection and other later responsibilities remain outside this ticket.

`retrieval-answer-policy/lexical-heuristic-migration-v1` is a retained,
explicit migration/diagnostic profile only. Its strategy and candidate-pool
scope identify it as `lexical_heuristic_migration` and
`legacy_migration_diagnostic`; it is never labeled or treated as Sparse BM25.

## Evidence Sufficiency And Frozen Answer Evidence

Only the active Pilot profile may pass an Authorized Retrieval Candidate Pool
to evidence sufficiency. The decider requires that profile identity and the
`published_knowledge` pool scope; a missing or incompatible boundary is
`no_eligible_published_evidence`. Candidate count, non-empty context, raw
BM25 score, ranking position, an arbitrary score threshold, and a model
judgment never establish sufficiency.

The deterministic input is an immutable Query Condition Set (QCS): the
normalized question plus its ordered, explicit `field`, `operator`, and
`value` conditions. Every selected item must match every decisive
applicability condition in that QCS. The decider fails closed for missing
conditions, unresolved review, material conflict, unsupported assurance, or
missing Claim-Evidence Link support. It also requires a governing
`recommendation_or_reviewed_branches` item and every question-shaped
complement required for comparison, diagnosis, acceptance review, or
implementation guidance. The governing item must deterministically cover the
normalized question through its reviewed `decision_query`; a raw retrieval
score, rank, non-empty context, or model inference cannot substitute for that
coverage. Claim-Linked evidence requires every reviewed `(section_id,
source_id)` link for each claim that supports a selected item. The decider
searches canonical item identities, not retrieval order, and selects the
smallest viable set that meets those requirements from the authorized pool
alone.

The only insufficient-evidence reasons are
`no_eligible_published_evidence`, `decision_not_covered`,
`decisive_condition_missing`, `material_evidence_conflict`,
`assurance_support_missing`, `evidence_budget_exceeded`, and
`knowledge_needs_review`. An insufficient result is a structured reply with
one of those exact codes. It has no Answer Evidence Set, recommendation,
citation identity, or provider-visible evidence payload, and it does not
invoke a provider.

A sufficient result freezes one immutable Answer Evidence Set before answer
generation. It contains at most three selected items, each with at most 1200
characters and at most 3000 characters in total; the decider rejects rather
than silently dropping required evidence that would exceed a cap. A truncated
candidate preview cannot be frozen as complete evidence: the authoritative
source content length must also fit the per-item cap. Each item's identity
binds the exact entry, editorial revision, Published Knowledge Version,
section, content-hashed chunk, source content length, and Evidence Excerpt
Snapshot. The persisted item includes that canonical identity binding, and a
reader must recompute it before trusting the item, its snapshot, or its
citation. The set identity binds the QCS, ordered item identities, and
governing item. Each citation identity binds that set identity and exactly one
selected item. Scores and selection diagnostics are not citation identity
inputs and are not provider-visible or user-facing citation data.

The provider-visible prompt is derived only from that same frozen set. Its
structural regions keep the normalized question, QCS, selected evidence
sources, and response contract separate. The response contract names only the
selected citations and governing citation. Each provider-visible source carries
the frozen `snapshot_id`, `item_identity`, and `citation_identity` from that
same set, while scores, chunk locators, and selection diagnostics remain
excluded. Retrieved text cannot modify policy, permissions, provider routing,
QCS conditions, evidence identities, or citation identities. Generated output
may cite only the frozen selected items, must cite every material nonblank line
in each required response section, and cannot introduce an unknown or
contradictory QCS assignment, secret value, unsupported quantified assurance,
or universalization of a bounded internal case.

The historical first-three selector, non-empty-context gate, and
candidate-derived citation projection are superseded for active Pilot
production decisions. They may remain only behind the explicit
`lexical_heuristic_migration` / `legacy_migration_diagnostic` profile and
must not produce a product sufficiency decision, immutable Answer Evidence
Set, or product citation identity.

## Pilot Identity Authority And Audit

The current `users` row is the authorization fact: protected handlers require
an extant active member and read its current database role. JWT role claims,
browser storage, and client-submitted member or owner values are never
authority. Bootstrap administration comes only from server-held deployment
configuration. Public registration consumes one active invitation exactly once
and always creates a Knowledge User; administrator promotion is an existing
System Administrator action. If configured bootstrap creation or its audit
cannot persist, application startup fails rather than serving without that
identity invariant. A conditional claim race re-reads the invitation after
rollback so its durable denial remains `replayed`, `revoked`, or `expired`
according to its final state.

`member` and `team_invitation` immutable canonical records establish safe
identity references. A denied unknown invitation code uses the separate
hash-derived `admission_attempt` identity; a known invitation's denial remains
on that invitation's canonical aggregate. Their append-only `identity_audit/v1`
events contain only an action, outcome, reason, timestamp, and minimal
actor/target/reference identities. Invitation and session references are
SHA-256 references, not plaintext codes or bearer tokens. Audit events never
contain passwords, JWTs, request bodies, questions, answers, or private
conversation history. The admin-only identity-audit projection exposes that
minimal event shape without creating an administrative conversation-history
reader.

`POST /auth/logout` deletes exactly the `auth:session:{sub}:{jti}` key for its
authenticated bearer. Deactivation is stronger: it marks the member inactive,
revokes every matching session key, prevents future login and protected access,
and retains the account and private history. Each action first commits a
content-free `pending` audit event; a subsequent `revoked` or `deactivated`
event records completion, while a session-store failure appends `failed`
instead of erasing the durable trail. Retrying an already inactive member
appends a terminal completion or another failure, so the latest append-only
trail does not incorrectly end at a historical failure.

## Compatibility

`compatibility_read_projection` maps existing records to deterministic
replaceable projections. It is read-only and carries `legacy_id` plus an
`unknown_fields` list. New callers must write canonical records first and may
continue writing legacy projections during expand-contract migration.

The Ticket 14 tail migration treats every previously unrevoked, unexpired
legacy invitation as consumed. The historic reusable-invitation schema cannot
prove a code was unused, so this conservative conversion fails closed; an
administrator issues a new invitation for any pending admission.

## Migration Plan

| Later ticket | Legacy caller | Canonical replacement | Removal condition |
| --- | --- | --- | --- |
| 14 | User, invitation, and Redis session admission/authority paths | Member/invitation identities and content-free identity audit events | One-time invitation consumption, database-derived role checks, and append-only identity audit are used by every pilot identity path |
| 16 | Markdown/front-matter authoring and runtime document copies | Private Editorial Repository entry, revision, source, and deterministic `editorial_export/v1` authority | T02 consumes only reviewed immutable exports and no authoritative editorial write remains on legacy/runtime rows |
| 17 | Upload and batch build dispatch | Reviewed Release Bundle, bundle item, build generation, recoverable Candidate Build | Reviewed bundles are the only new authority-bearing intake; Candidate work has no publication side effect and legacy publication paths remain compatibility-only |
| 18 | Unqualified legacy retrieval and Candidate-derived chunks | Versioned Pilot Sparse BM25 and an authorized current-Published Candidate Pool | Ordinary retrieval returns only current, authorized compatibility-published chunks; Candidate preview remains administrator-only and diagnostic |
| 19 | First-three selection, non-empty context gate, and candidate-derived citations | Deterministic evidence sufficiency and immutable Answer Evidence Set | Active Pilot uses only an authorized pool, exact QCS and assurance rules, one frozen selected set, and its bound citation identities |
| 20 | `ChatMessage.rag_trace` and answer inference | Answer execution, conditions, evidence set, snapshot | Every answer persists the closed canonical outcome |
| 21 | HTTP, SSE and history adapters | Canonical execution projection | All surfaces read one canonical execution |
| 24 | Candidate inspection and publication | Candidate and Published Knowledge Version | Publication checks canonical generation, hash and acceptance identities |
| 25 | Tombstone and redaction | Withdrawal event and retained publication identity | All withdrawal reads and writes use canonical publication identity |
| 27 | Feedback and review work items | Maintenance item and validated finding | Raw feedback references can expire without losing the durable canonical decision |
| 15 | Acceptance scripts and evidence | Delivery Acceptance Record and status events | Acceptance binds exact canonical identities, never branch or `latest` |

Compatibility paths may be removed only after the named later ticket has
completed its authenticated product-path migration, replay/backfill check, and
no remaining legacy-only writes are observed for the declared retention window.
