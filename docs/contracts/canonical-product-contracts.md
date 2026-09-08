# Canonical Product Contracts

Status: normative product contract; additive foundation for tickets 13 through 20

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
export. A System Administrator cannot inspect or directly modify the Private
Editorial Repository and may receive only an approved export for editorial
handling. Candidate publication is a narrow server-side exception, not an
administrator editorial command: only after `CandidatePublicationService`
re-verifies one exact frozen approved export under the finalization fence and
atomically writes exact Candidate and Published Knowledge Version identities
may `EditorialAuthorityService` append machine-derived `candidate_build` and
`published` lifecycle events. The path exposes no private editorial content,
accepts no administrator-selected editorial field, and cannot revise a private
record.

The private lifecycle writes `draft`, `evidence_collected`, and
`editorial_review` events. Evidence collection and review require a complete
revision. Material changes create a new immutable revision and require
Editorial Review, including when they begin from `published`; the existing
Published Knowledge Version pointer remains live until a separately approved
successor passes Candidate publication. A wording-only revision may retain
`published` while it receives its own Maintainer responsibility acceptance and
lightweight approval. Wording-only changes require an approved base revision
and may normalize title and authored-body whitespace only. Any punctuation,
case, token, comparison/operator, or structured-authority change is material.
T01 records no Reviewed Release Bundle,
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
or a runtime publication pointer. Only the separate Candidate inspection and
explicit publication contract below may create the runtime projection.

## Candidate Inspection And Explicit Publication

Candidate inspection is an administrator-only read of one exact immutable
Candidate binding: Candidate record, matching `candidate_ready` job, frozen
input, bundle/item hashes, current generation, contiguous content-hashed
Candidate chunks, and effective embedding configuration. Recording inspection
creates immutable `candidate_inspection/v1` evidence plus an append-only
Candidate event. It re-verifies the retained approved editorial export and
creates or verifies the content-addressed immutable configuration record.
A Claim-Linked Candidate Build must copy its exact frozen Candidate
Claim-Evidence contract and canonical hash into each Candidate chunk. That
contract binds exactly its material claims and their reviewed section/source
links; it does not require a resolver, calibration, or a Release-Assured gate.
A missing, malformed, hash-mismatched, or link-mismatched contract fails the
build before Candidate persistence. The approved export freezes that canonical
JSON and hash at the artifact root. An editorial revision may additionally
retain a full Claim-Evidence contract as a compatibility projection, but its
resolver and calibration requirements remain Release-Assured authority.
A replacement inspection snapshots the current
`replaces_published_knowledge_version_identity` in immutable inspection
evidence, returns that exact Published Knowledge Version plus deterministic
per-index chunk `added`, `changed`, and `removed` hashes, and may be reloaded
for administrator audit even after the Candidate becomes stale. Candidate
acceptance copies that exact replacement identity, and a changed pointer fails
publication eligibility and confirmation. Inspection never makes material
retrievable or changes a publication pointer.

Candidate acceptance is an administrator-only adapter to the closed evidence
contract, not ordinary retrieval and not a provider invocation. For a
Claim-Linked section, the adapter creates one ephemeral Candidate-bound
evidence projection per reviewed source link so the closed executor must select
every required Claim-Evidence Link; those temporary source identities are
retained only in the immutable acceptance binding and never enter ordinary
retrieval. It records one supported query whose frozen Answer Evidence Set contains the exact
Candidate, build generation, governing entry and
`recommendation_or_reviewed_branches` section, content-hashed chunk, snapshot,
and citation marker. An acceptance query may retain an explicit QCS as the
same complete four-field condition records used by closed Answer Execution;
otherwise its QCS is derived from the frozen question. A supported query
requires the exact applicable conditions. It also records one Boundary Query
as a closed `decision_not_covered` or `decisive_condition_missing`
insufficiency reply with zero citations and zero provider calls. The immutable
`candidate_acceptance/v1` record binds the exact inspection event, Candidate,
bundle/item and input hashes, and configuration. Diagnostic preview, an older
or stale Candidate, a different configuration, or a mismatched frozen input
cannot satisfy this acceptance.
Both acceptance queries use the same closed Answer Evidence selector as Answer
Execution: it requires complete QCS parsing, exact Candidate-bound
source/assurance/section bindings, governing and required complement selection,
deterministic budget checks, and normal evidence, snapshot, and citation
construction. It never evaluates only a first chunk, truncates a Candidate to
establish support, tolerates malformed condition records, or hand-assembles
evidence.

Reload and publication eligibility verify the retained inspection's complete
entry, document, bundle/item, revision, generation, hash and configuration
binding. They validate the whole deterministic acceptance result against the
exact frozen Candidate, including QCS, evidence items, snapshots, citation
markers and JSON scalar types; a shape-compatible or corrupted record cannot
grant eligibility. This verification calls no provider, writes no new
acceptance record, and does not revise the historical result. Administrator
inspection reload exposes the retained acceptance and its exact bindings.
The administrator view keeps those records inspectable and distinguishes
partial publication from complete batch success.

Release-Assured acceptance recognizes both a stable `event:` reference and the
32-character lowercase UUID hex retained by `canonical_events.id` in an
authoritative editorial snapshot. It rejects other unqualified references and
does not normalize, rewrite, or rehash the frozen export to bridge these forms.

An administrator may reload a historical inspection after a Candidate becomes
stale or its job reaches `superseded`, but eligibility remains latest-generation
only. The administrator-only Candidate publication read projection reports that
Candidate's immutable Published Knowledge Version identity, its exact
inspection/acceptance and superseded-version identities, and whether it remains
the current entry pointer. An eligible Candidate must still prove current approved editorial authority,
the latest Candidate generation, exact inspection, exact Candidate acceptance,
and the same bundle and configuration identities. An administrator submits a
strict selected batch whose every item names the Candidate, `create` or
`replace` effect, the current Published Knowledge Version identity (or `null`
for create), and the exact immutable inspection and acceptance record
identities. The confirmation identifier is idempotent only for the same actor
and canonical selected payload. A changed selection, inspection/acceptance
identity, or replacement pointer fails closed.

Each selected item is a separate transaction. It first writes the runtime
`Document` and `DocumentChunk` projection tagged with the immutable
`published_knowledge_version` identity, then writes the immutable Published
Knowledge Version record with those exact inspection/acceptance identities and
its append-only publication event, and only then creates or switches the one
mutable entry pointer. Final verification holds the approved editorial
authority fence through this commit and advances the same verified entry from
`editorial_review`, through `candidate_build`, to `published` when needed, so
ordinary retrieval resolves the current publication pointer rather than a
Candidate preview. An otherwise-safe unpublished successor revision does not
replace the pointer's frozen Published Knowledge Version authority. A failure
rolls back that item before pointer switching and produces a retryable
`failed` result; other selected items continue independently.
Before allocating a new runtime source document, publication checks the
first-release maximum of 500 published source documents. A `replace` reuses
the existing published source allocation; a new entry at the limit fails only
that selected item with `PUBLISHED_SOURCE_LIMIT_REACHED`.
Batch output always separates `published`, `failed`, and `skipped`, and
`batch_complete` is true only when no item failed or was skipped. A duplicate
confirmation returns its retained outcome without writing a second version or
moving a pointer. A retained `processing` confirmation resumes from its
persisted item results; a version committed before its result was recorded is
recovered only when its exact selected inspection/acceptance identities match.

Publication reads verify each version projection against its immutable
canonical publication record and frozen Candidate, and verify the entry
pointer's document and generation against that version. A missing or
inconsistent pointed version cannot become a replacement inspection.
A terminal failed confirmation remains immutable in meaning: repeating its
identifier replays the same per-item results. Retrying failed items requires a
new explicit confirmation containing only the still-eligible selected items;
already-published siblings and their pointers remain unchanged.

`POST /documents/{id}/publish` rejects every identity because that legacy
compatibility route could bypass Candidate inspection, acceptance, and explicit
confirmation. Ticket 24 does not implement withdrawal, redaction, or historical
deletion: Ticket 25 will add those events while preserving the immutable
Published Knowledge Version identity and the later withdrawal seam.

## Retrieval Answer Policy And Authorized Candidate Pool

`retrieval-answer-policy/pilot-v1` is the active ordinary-user retrieval
profile. It is genuine `sparse_bm25`, with literal-preserving tokenization,
`k1=1.5`, `b=0.75`, candidate depth `20`, and the versioned
`retrieval-candidate-tie-breaker/v1`. The profile does not accept field boosts:
an unrecorded boost is rejected rather than silently changing the effective
profile. It keeps reranking, lexical-answer anchors, semantic near-duplicate
removal, query expansion, and online LLM sufficiency judging disabled. The
effective profile identity is carried with every retrieval result and trace.

Pilot forms its ordinary pre-sufficiency Candidate Pool only from the current
published runtime projection. A Ticket 24 projection carries an exact
immutable `published_knowledge_version` identity; older documents without that
field retain the explicitly labeled `published_knowledge_version:legacy:*`
compatibility mapping. A malformed claimed Published Knowledge Version
identity fails closed. A chunk is eligible only when its `Document` is not
withdrawn and its generation equals that document's current published
generation. Before ranking, the pool resolves the current Private Editorial
Repository authority for the entry: the current revision, lifecycle
eligibility, exact per-section verified source relationship, assurance,
applicability, freshness, and team-shared access scope. Projection metadata is
only a binding to those current facts; missing, malformed, or mismatched
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
When the current authority reports a later safe but unpublished revision, the
pool may preserve a genuinely pointed Published Knowledge Version only by
reconstructing its exact Candidate, frozen input, and retained revision, then
revalidating current source and release-assurance records. Runtime projection
metadata can bind that reconstruction only; it cannot supply a decision query,
source relationship, assurance, condition, freshness, or access authority.
Any missing, malformed, mismatched, or newly unusable retained fact excludes
the pointed version.

A successor's unverified or unavailable new source does not itself revoke the
pointed predecessor. The pool revalidates the pointed revision's own sources,
approval and integrity facts before retaining it. Decisive loss of a source
used by that revision remains blocking after source recovery; a later
successor approval cannot truncate that revision's loss history. A new revision
approved after the loss is evaluated from its own approval boundary. Both
current and historical authority use the same validated source/section
projection, while their eligibility decisions remain separately scoped.

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
sources, Evidence Set identity, ordered knowledge-version identities, and
response contract separate. The response contract names only the selected
citations and governing citation. Each provider-visible source carries the
frozen `snapshot_id`, `item_identity`, and `citation_identity` from that same
set, while scores, chunk locators, and selection diagnostics remain excluded.
Retrieved text cannot modify policy, permissions, provider routing, QCS
conditions, evidence identities, or citation identities. Generated output may
cite only the frozen selected items, must cite every material nonblank line in
each required response section, and cannot introduce an unknown or
contradictory QCS assignment, secret value, unsupported quantified assurance,
or universalization of a bounded internal case.

Before any provider call, the complete parsed provider-visible input must equal
the one record deterministically built from the frozen normalized question,
exact QCS identity and ordered condition records, Evidence Set, every selected
source field including its citation marker, ordered item and citation
identities, ordered snapshots, ordered knowledge-version identities, and the
complete response contract when one is present. JSON object parsing rejects a
duplicate key at every nesting depth; a later key must never silently overwrite
an earlier frozen field. The comparison does not trim, normalize, omit, or
reconstruct provider-visible fields: non-canonical QCS strings, an altered
citation marker, or an altered response contract are mismatches. Each
source-declared `snapshot_id` must equal the snapshot
recomputed from that source's provider-visible content. A valid observed
generation envelope must carry the same snapshot sequence. A malformed or
mismatched provider input or observed envelope is an application failure, not
`generation_unavailable`, insufficiency, or a supported answer. A provider
envelope that is present but cannot be validated is malformed rather than
absent. Under ADR 0005, normalized approved-route failure or exhaustion,
including missing approval, produces `generation_unavailable` even without a
completed provider call. Unclassified application errors remain application
failures. Neither case is insufficiency.

## Approved Generation Route

ADR 0005 binds immutable provider/model approvals, HTTPS endpoint class, data
scope, ordered primary/fallback selection, per-provider timeout, attempt bound,
and total budget to one versioned route identity. Administrator activation
requires exact current Delivery Acceptance evidence and connection validation
before an atomic pointer replacement; in-flight executions retain their
captured route. Available credentials and legacy settings grant no authority.

Only normalized connection, timeout, rate-limit, temporary/sanitized service,
and deterministic answer/citation failures may advance within remaining budget.
Insufficiency, cancellation, authorization, privacy/data-scope, safety/policy,
and undeclared-provider decisions never advance. Each attempt receives the
same frozen payload and snapshot hashes. Operational Events retain only
non-content identities, hashes, timings, and normalized reasons. Secrets are
encrypted at rest and never returned. Controlled live provider, failure,
privacy, and prompt/citation evidence is required before Pilot; deterministic
Local Development evidence is not a substitute.

The historical first-three selector, non-empty-context gate, and
candidate-derived citation projection are superseded for active Pilot
production decisions. They may remain only behind the explicit
`lexical_heuristic_migration` / `legacy_migration_diagnostic` profile and
must not produce a product sufficiency decision, immutable Answer Evidence
Set, or product citation identity.

## Closed Answer Execution And Private Conversation Persistence

An authenticated chat request is admitted as one private Answer Execution. It
retains one immutable `answer_execution_request/v1` request header and an
append-only `answer_execution_event/v1` trail while the private conversation is
retained. Event sequence is unique per execution, and every terminal,
redaction, or stream-delivery lifecycle writer locks that execution before
reading and appending its next event. Conversation deletion and retention purge
acquire those execution locks before deleting its event trail or header, so
retention cannot leave private events without their owning execution. On
SQLite, admission, event writers, deletion, and expiry share a transaction-wide
writer fence before their relevant reads: a fresh transaction begins it with
`BEGIN IMMEDIATE`; an already-open deferred read transaction upgrades it with a
no-op `chat_sessions` write. Cleanup locks matching execution headers, locks
the verified conversation session, then re-scans headers before delete;
admission locks that verified session behind the same fence. Thus an admission
or event write cannot race private deletion or expiry into an orphaned header,
event, or message. This is not a
`canonical_records` aggregate: it is private conversation data, so verified
conversation deletion or expiry may delete its header and events together with
its linked messages and session. While it is retained, neither the request
header nor an event may be rewritten. Admission durably binds its exact
user-message identity before execution can run.

Execution states are distinct from answer outcomes. The admitted path is
`admitted -> queued -> running`; its terminal states are `completed`,
`stopped`, `failed`, `throttled`, and `rejected`. A `completed` execution has
exactly one outcome: `evidence_gated_answer`,
`insufficient_evidence_reply`, `non_knowledge_base_reply`, or
`generation_unavailable`. A stopped, failed, throttled, or rejected execution
has no completed answer outcome, Answer Evidence Set, citation, or fabricated
answer text. `completed` is an execution state, not a fifth outcome.

Admission normalizes and records a visible Query Condition Set (QCS) with the
question. A QCS contains its identity, normalized question, and ordered,
explicit decisive conditions such as versions, environment, scale, and
targets. A caller may submit editable conditions explicitly, or inherit only
the latest completed QCS from the same private conversation and same owner.
Inheritance copies the conditions into a new QCS bound to the new normalized
question and records the source execution identity; it never reuses the prior
QCS identity, including when the user repeats the same normalized question. A
new conversation cannot inherit conditions. When no conditions
are submitted, admission derives them only from the admitted question text; a
hidden conversation memory, user profile, global profile, or retrieval result
cannot supply a decisive condition. Every completed turn freezes its QCS and
provenance, and the owner-visible Answer Execution projection exposes both.
A composer is only a pre-admission draft: opening a conversation, starting a
new conversation, or deleting the active conversation clears its draft
conditions and inheritance flag. A retry with any retained execution,
including a non-completed failure projected over SSE or history, submits that
turn's frozen QCS as explicit conditions. Missing client-side terminal data
does not authorize re-inheritance: a retry marked as inherited but lacking the
retained execution must fail locally until that execution is recovered. Only a
pre-admission transport failure may repeat the original requested conditions or
inheritance request. A client accepts a streamed execution only after binding it
to that submitted turn: an implicit QCS must exactly match conditions derived
from the submitted question, and an inherited QCS must name a completed source
execution already projected in the same private conversation and copy its exact
conditions. A completed execution becomes retry authority only after the full
terminal projection validates; a contradictory terminal projection clears local
retry authority until the persisted execution is recovered.

Before retrieval, the narrow non-knowledge-base allowlist may select
`non_knowledge_base_reply`. That result has no knowledge claim, Answer
Evidence Set, citation, or provider call. All other admitted knowledge
requests use the deterministic evidence-sufficiency decision from the
admitted QCS. A missing decisive condition remains reviewed conditional
branches or a structured insufficiency; it is never silently filled by memory
or inference.

A completed `answer_execution_result/v1` retains the exact normalized
question, frozen QCS and provenance, outcome, frozen text, Evidence Set
identity, ordered item identities, snapshot identities, and knowledge-version
identities. Its owner-visible completed execution projection also carries the
exact assistant-message binding, frozen answer text, and frozen
outcome-specific evidence summary so a transport projection can compare rather
than infer. Evidence-Gated Answer and Generation Unavailable also retain a
provider-input identity record containing that same question, QCS identity,
Evidence Set identity, item identities, snapshots, and knowledge versions.
An Insufficient Evidence Reply retains one structured
`insufficient_evidence_reply` record with its outcome, exact reason, and QCS
identity; no other completed outcome retains that record.
`generation_unavailable` retains those identities only to prove its closed
execution boundary. Its answer projection carries no knowledge claim,
citation, source, or evidence preview, and it cannot be presented as a
supported-answer projection.
The request header binds its exact user-message identity. Every completed
result, and every non-completed terminal represented by a persisted assistant
message, binds that exact assistant-message identity. If assistant-message
persistence fails after admission, completion is rolled back and the retained
record may instead append only a `failed` terminal with
`ANSWER_EXECUTION_PERSISTENCE_FAILED`, no assistant-message identity, no
answer text, and no outcome; it is owner-visible only through its frozen
user-message binding. `ChatMessage.answer_execution_id` is a mutable lookup
index, not an authority: readers discover and validate the immutable request
or terminal binding in the same private conversation, so clearing the index
cannot turn a bound message into a legacy trace projection, moving that
message to another private conversation fails closed, and a non-null
contradictory index is an application failure. Normal HTTP, SSE terminal events, the linked persisted assistant
message, reload, and private history are projections of this one result only
after validating those frozen message bindings. They may not retrieve again,
reselect evidence, re-slice snapshots, invoke a provider, derive an outcome
from text, a flag, a source count, a score, or a trace, or upgrade
insufficiency, a stop, or a failure to a supported answer.
SSE admission appends `stream_delivery_pending` before a closed result can be
delivered. A completed result with that pending record but without a delivery
completion or interruption record is not projectable through reload or history:
it is an application failure, not a completed-answer replay. Normal HTTP does
not create an SSE delivery lifecycle record and may project the same closed
result after ordinary persistence completes.
An SSE client accepts a completed terminal only when its assistant identity
equals the execution's assistant binding; its completed execution identity and
state, answer text, normalized question, complete non-duplicated QCS and
provenance, explicit outcome, and full outcome-specific evidence summary all
equal the frozen execution projection, and that question, QCS, and provenance
also equal the normalized submitted turn. For an Insufficient Evidence Reply,
the structured reply carried by both terminal stream and execution must exactly
match; no other outcome may carry one. A valid terminal is only a fully framed
SSE `event: done` whose data payload is the exact unquoted literal `[DONE]`.
EOF before that frame's blank-line separator, a quoted or otherwise altered
marker, or that marker on any other event is a missing or contradictory
terminal. A repeated `done`, or any semantic frame after `done`, is also an
application failure. Missing, duplicated, or contradictory terminal fields are
an application failure; a client cannot invent an insufficiency reply or other
replacement result. An error or
cancellation received after completed terminal fields is an application
failure, not a stop: the client clears the completed outcome, structured
insufficiency reply, evidence summary, and diagnostics rather than retaining a
completed projection.

`ChatMessage.rag_trace` is a compatibility diagnostic only. A linked Answer
Execution never reads outcome, evidence, citation, snapshot, or condition
semantics from that trace. For a linked execution, the persisted diagnostic
projection retains only bounded operational metadata such as gate state, step
names, candidate counts, provider identities, timing, and error
classifications; it does not retain the question, QCS, answer text or preview,
evidence content, provider-visible generation envelope, or private history. A
legacy message has no immutable request or terminal binding, not merely a null
mutable index; it may remain readable through its bounded compatibility
projection during retention, but it cannot become a new Answer Execution
result.

Authentication failure occurs before admission and creates no answer outcome.
An unrecovered retrieval or unclassified provider failure, and every execution, stream, or
persistence failure, is an application failure, never insufficiency. A
retrieval implementation may retain a diagnostic for a recovered fallback only
when it returns an actual candidate result; it may not synthesize an empty
result and label that provider failure as insufficient evidence. An interrupted
stream while the execution is running cancels it and waits until its private
stopped terminal event is durable. An ASGI send exception or disconnect uses
the same durable cleanup. A completed SSE delivery is not finished until an
outer ASGI transport observer has successfully written its terminal `done`
body and completed the response finalization beyond every response-buffering
middleware, then appended `stream_delivery_completed`. If delivery
interrupts after projection of a closed result has begun, an append-only
`stream_delivery_interrupted` record leaves the original result immutable but
makes it non-projectable; reload and history fail as an application failure
rather than reclassifying it as failed, insufficient, or supported. A missing
terminal event, a contradictory terminal state payload, or a persisted message
that contradicts the frozen binding is an application failure on projection,
not an invitation to infer a replacement answer. When a post-admission stream
fails, SSE projects the retained non-completed execution and its frozen QCS
(plus an assistant binding when persistence produced one) before its `error`
and `done`; it still exposes no completed outcome or evidence summary. A
persistence failure rolls back uncommitted completion rather than leaving a
partial completed result; if an assistant message still cannot be persisted,
the existing user binding retains only the explicit failed persistence terminal.

A separately authorized later document tombstone may append a private
evidence-redaction event. It redacts the historical excerpt in the execution
projection and marks the retained item as withdrawn while preserving its
Evidence Set, item, snapshot, and knowledge-version identities. It never
rewrites the original completed terminal result to make a new semantic answer.
Completion locks the frozen evidence documents before it appends its terminal
event, and tombstoning locks those same documents before it marks them
withdrawn. If a frozen document is already withdrawn when completion obtains
the lock, completion appends the redaction event in that same transaction and
uses the redacted projection immediately; it never reselects evidence or
removes its retained identities.

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
| 17 | Upload and batch build dispatch | Reviewed Release Bundle, bundle item, build generation, recoverable Candidate Build | Reviewed bundles are the only new authority-bearing intake; Candidate work has no publication side effect |
| 18 | Unqualified legacy retrieval and Candidate-derived chunks | Versioned Pilot Sparse BM25 and an authorized current-Published Candidate Pool | Ordinary retrieval returns only current, authorized Published projections; Candidate preview remains administrator-only and diagnostic |
| 19 | First-three selection, non-empty context gate, and candidate-derived citations | Deterministic evidence sufficiency and immutable Answer Evidence Set | Active Pilot uses only an authorized pool, exact QCS and assurance rules, one frozen selected set, and its bound citation identities |
| 20 | `ChatMessage.rag_trace`, transport-specific gates/generation, persistence, snapshot slicing, and outcome inference | Private append-only Answer Execution with frozen QCS and closed result | Normal HTTP, SSE, linked persistence, reload, and private history project one retained execution; legacy trace is diagnostic only |
| 21 | Chat UI view state and later interaction adapters | Private Answer Execution projection | Browser and later UI paths display the retained execution state and completed result without adding a second semantic owner |
| 24 | Candidate inspection and publication | Immutable inspection/acceptance events, Published Knowledge Version, and mutable entry pointer | Publication checks exact Candidate generation, bundle/input hashes, configuration and acceptance identities; `runtime-document:*` legacy publication rejects |
| 25 | Tombstone and redaction | Withdrawal event and retained publication identity | All withdrawal reads and writes use canonical publication identity |
| 27 | Feedback and review work items | Maintenance item and validated finding | Raw feedback references can expire without losing the durable canonical decision |
| 15 | Acceptance scripts and evidence | Delivery Acceptance Record and status events | Acceptance binds exact canonical identities, never branch or `latest` |

Compatibility paths may be removed only after the named later ticket has
completed its authenticated product-path migration, replay/backfill check, and
no remaining legacy-only writes are observed for the declared retention window.
