# Canonical Product Contracts

Status: additive foundation for tickets 13 and 14

## Purpose

The canonical contract layer gives later product slices stable identities,
closed state vocabularies, immutable records, and append-only events without
reinterpreting or deleting existing runtime rows. It is intentionally separate
from the legacy `documents`, `document_jobs`, `chat_messages`, and feedback
tables during migration.

## Identity

`StableIdentity` is a `(kind, value)` pair. The following kinds are reserved
(events use the separate `event` kind):
`member`, `team_invitation`, `admission_attempt`, `entry`, `source`, `bundle`,
`bundle_item`, `build_generation`, `candidate`,
`published_knowledge_version`, `answer_execution`, `evidence_set`,
`evidence_snapshot`, `maintenance_item`, and `delivery_acceptance_record`.
Identity values are immutable. A title, filename, generation number, score, or
current `latest` pointer is not an identity.

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
