# Canonical Product Contracts

Status: normative product contract; additive foundation for tickets 13 through 17

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
| 17 | Upload and batch build dispatch | Reviewed Release Bundle, bundle item, build generation | Reviewed bundles are the only recurring intake and legacy jobs are terminally projected |
| 20 | `ChatMessage.rag_trace` and answer inference | Answer execution, conditions, evidence set, snapshot | Every answer persists the closed canonical outcome |
| 21 | HTTP, SSE and history adapters | Canonical execution projection | All surfaces read one canonical execution |
| 24 | Candidate inspection and publication | Candidate and Published Knowledge Version | Publication checks canonical generation, hash and acceptance identities |
| 25 | Tombstone and redaction | Withdrawal event and retained publication identity | All withdrawal reads and writes use canonical publication identity |
| 27 | Feedback and review work items | Maintenance item and validated finding | Raw feedback references can expire without losing the durable canonical decision |
| 15 | Acceptance scripts and evidence | Delivery Acceptance Record and status events | Acceptance binds exact canonical identities, never branch or `latest` |

Compatibility paths may be removed only after the named later ticket has
completed its authenticated product-path migration, replay/backfill check, and
no remaining legacy-only writes are observed for the declared retention window.
