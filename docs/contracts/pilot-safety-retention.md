# Pilot Safety And Retention

Status: normative; Ticket 26, KB-SO-002/003 and the privacy portions of
KB-UX-007 and KB-FB-001/002. ADR 0007 records the decision.

## Content Admission

Each reviewed source retains `content_admission` in its immutable definition
and approved export. `material_class` is `public_material`,
`team_shared_internal`, or `sanitized_bounded_internal_case`.
`audience=all_admitted_members`, `sensitivity=restricted`, and
`sanitized=true` are required. Public material uses public access; internal
material uses controlled access; Bounded Internal Cases require the sanitized
case class. These declarations do not grant publication authority: accountable
Maintainer verification, distinct Reviewer approval, Candidate inspection,
acceptance and explicit publication remain required.

Secrets, credentials, personal/customer data, raw production dumps, unredacted
logs/screenshots, subset-authorized or mixed-sensitivity content, and dangerous
unpublished exploit detail are not admitted classes. Recognizable credential
and PII patterns are screened before editorial or legacy-upload persistence.
Patterns cannot prove absence of every sensitive fact; human review remains
accountable. Legacy uploads cannot publish. Missing admission is unverified,
not a reason to fabricate approval or rewrite old hashes.
Upload, locator and editorial checks share the same named credential rules,
with PII checks composed separately. Admission checks raw strings and up to
three percent-decoding rounds; inputs still requiring decoding are rejected
at this bounded boundary rather than assumed safe.

`sanitize_public_source_url` prepares a public HTTPS locator without credentials,
unsafe query data or fragments and rejects private/local addresses and control
characters. Editorial and parser authority accept only an already-canonical
locator; unsafe input is rejected rather than rewritten inside hashed artifacts.
Only bounded `version`, `v`, `revision`, `rev`, `page` and `lang` query selectors
are preserved. Other parameters, recognizable credentials and private address
values are removed by preparation; authority rejects the unsanitized input.
Recognizable private material in a raw or encoded URL path is rejected,
not retained as a canonical path or silently rewritten into another source.

## Private Records

Session and feedback access remains owner-scoped. Foreign members and
administrators receive the same empty/no-op result as a nonexistent private
record; there is no administrative transcript bypass. Conversation deletion
removes messages and private execution headers/events under Ticket 20's writer
fences. While retained, withdrawn history follows Ticket 25's exact redacted
identity. Deletion removes the private record completely.
Deletion and orphan repair remove messages identified by immutable execution
bindings before deleting those bindings, even if mutable session/owner/index
fields were corrupted. Surviving bound messages abort deletion, preserving the
facts needed to deny cross-member disclosure and retry verified removal.
This includes terminal events whose header is already missing. A mutable
message execution index never authorizes cross-member deletion.

Explicit feedback is independently retained after conversation deletion, without
automatically copying a transcript. Optional descriptions live only in raw
signals; review projections join retained signals rather than copied metadata.
Signal deletion/expiry removes unclassified triggers. Classified decisions
retain only non-personal subject, classification and disposition: raw references,
deduplication links, descriptions, metadata and exact submission timestamps are
severed; gap subjects become `detached-feedback`. Ticket 27 owns the later
complete maintenance/finding workflow.
Owner feedback deletion verifies both raw removal and reference severing before
committing success; surviving data returns `PRIVACY_DELETE_UNVERIFIED`.

## Policy And Cleanup

The frozen version-1 defaults are `conversations=30`, `operational_events=30`,
`feedback_signals=180` days, with a canonical SHA-256 `configuration:` identity.
Administrator-only `GET /retention` exposes policy and non-content cleanup
status. `PUT /retention/policy` requires `expected_identity` and strict integer
durations from 1 through 3650. Zero, disabled, malformed and content-access
settings are rejected. Registry-locked changes append immutable configuration
and policy events. Same values are no-ops; stale updates return
`RETENTION_POLICY_STALE`.

Conversation/event expiry uses creation time, not activity. New feedback uses
the effective duration; cleanup honors the earlier of its original deadline
and the current creation-time deadline. A policy increase cannot extend an
already promised feedback expiry. Classes never share content storage or grant
administrator transcript access.

Incoming request identifiers are preserved only as canonical UUIDv4 values.
Other values are replaced before response, logging context and Operational
Event correlation; a free-text header cannot become a content telemetry field.

Policy changes suspend affected acceptance records. Pilot, Daily-Use and Public
Evidence acceptance must bind current policy and verified cleanup. Old policy,
missing or stale cleanup blocks the deployment projection. Suspended records
require new acceptance; cleanup does not restore them automatically.

Startup and the 60-second loop clean all three classes without user traffic,
each in a separate transaction. The bounded `retention_cleanup_states` table
retains only class, attempt, policy identity, pending/verified/failed status,
time, deletion count, remaining expired count and normalized reason. Pending
commits before deletion; verified commits with the verified deletion transaction.
Conversation verification includes orphan messages, execution headers and events.
Feedback verification includes orphan review references, and retries discover
them even after the original signal has disappeared. Failure recovery compares
attempts so an older worker cannot overwrite a newer verified result.

Exceptions produce `RETENTION_CLEANUP_FAILED` without exception text; survivors
produce `PRIVACY_EXPIRY_SURVIVED`. Failure in one class does not stop siblings.
Administrator-only `POST /retention/cleanup` retries. Missing, policy-stale or
more-than-180-second-old, future-dated or contradictory observations remain
unverified and block acceptance. Noncanonical state/error text is not projected.
Cleanup failures also append a non-content canonical failure event, referenced
by a durable shared-privacy deployment blocker on affected live acceptance
records. A later successful cleanup cannot remove that retained suspension.
Failure evidence commits independently of cleanup-state recovery and binds the
exact affected acceptance identities, including records already suspended for
a local defect. Its deployment blocker coexists with prior local blockers.
Status reads validate retained failure evidence and cannot reuse an older
verified observation when a newer failure exists. A retry response also retains
the current attempt's failure if status persistence is unavailable.
Failure evidence records the locked observation attempt it invalidates;
worker start timestamps do not decide which verification supersedes a failure.
New attempts advance beyond retained invalidations even if mutable state was
lost. A fresh pending attempt may be normally superseded by concurrent cleanup
without a permanent failure. A pending observation stale for 180 seconds (or
contradictorily future-dated) is treated as interrupted.

## Exposure Evidence

`deploy/production/compose.yml` is the supported Pilot shape: only Caddy
publishes 80/443, infrastructure/backend have no host ports, and the application
network is internal. HTTPS certificates, plaintext redirect/rejection,
authenticated user paths and external exposure still require deployment proof.
The same shape may omit Caddy's UDP 443 mapping when HTTP/3 is not enabled;
TCP 80/443 remain the only permitted published TCP ports.

```bash
cd backend
uv run python -m app.operations.exposure \
  --compose ../deploy/production/compose.yml \
  --source-revision <exact-source-sha> \
  --output <private-evidence-directory>/exposure.json \
  --authorized-host <authorized-deployment-dns> --all-tcp-ports --tcp-workers 1
```

Probe only explicitly authorized hosts. Reports contain hashes, revision, time,
port scope and booleans, not hosts, credentials or bodies. Full TCP mode permits
only SSH and HTTP/HTTPS; default bounded mode checks declared infrastructure
ports. UDP and firewall inspection are separate obligations. Timeouts are
incomplete probes, not closed-port proof.
The TCP probe defaults to four concurrent connections; `--tcp-workers` accepts
1 through 64. Use one worker on constrained paths. Lower concurrency does not
reduce the port scope or turn a timeout into proof of closure.
Configuration-only runs without `--authorized-host` deliberately exit nonzero and retain
`live_exposure_verified=false`, never a public Pilot pass.
