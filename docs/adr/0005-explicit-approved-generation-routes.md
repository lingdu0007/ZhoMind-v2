# Explicit Approved Generation Routes

Status: accepted for Ticket 22 implementation; Pilot activation still requires
controlled live evidence.

We replace permanent single-provider selection with one explicitly approved,
versioned generation route: one primary and at most three ordered, separately
approved fallbacks. This permits bounded recovery without treating available
credentials, environment aliases, or a provider registry as permission to send
team data.

## Authority And Replacement

An immutable `provider_route` record binds provider/model identity, HTTPS
endpoint and declared endpoint class, `team_shared_pilot` data scope,
provider-specific timeouts, total budget, maximum attempts, and each provider's
`configuration` approval identity. Credentials live only as Fernet ciphertext
behind opaque references; neither credentials nor references are returned.
Every saved replacement requires explicit credential submission.

An administrator saves an inactive draft, supplies a currently active Delivery
Acceptance Record binding the route and every approval identity to verified
provider, failure, privacy, and prompt/citation checks, and requests activation.
A bounded connection validation uses only a fixed non-content prompt. The
final transaction rechecks the current administrator, evidence, latest draft,
and expected active pointer, then appends an activation event and switches the
pointer atomically. Failed or stale activation preserves the previous route.
New executions capture this persisted authority; captured executions retain
their prior immutable route. Revoked or changed acceptance evidence stops new
captures and is not projected as active.

The server's `GENERATION_DEPLOYMENT_IDENTITY`, exact
`GENERATION_PRODUCT_REVISION` (`product_revision:<40-hex-commit>`), and
`GENERATION_VALIDATION_MODE` must match the acceptance record. Default
`controlled_live` mode rejects Local Development records and requires an
Editorial Preview or later record explicitly declaring that mode. Empty
deployment/revision configuration denies activation. `local_development`
accepts only Local Development records and disables real provider construction
in the production factory; isolated tests must explicitly install provider I/O
doubles. A browser cannot change this server boundary.

## Attempts And Outcomes

Only connection failure, timeout, rate limit, temporary service error,
sanitized service error, deterministic answer-structure failure, and citation
failure may advance. Each provider is attempted at most once within both the
declared attempt count and remaining time budget. SDK retries and HTTP
redirects are disabled; attempt-local HTTP clients close after completion or
cancellation.

Insufficiency calls no provider. Cancellation, authorization, data-scope,
safety, policy, and undeclared-provider decisions never advance. Every attempt
receives identical frozen question/QCS/evidence payload and ordered snapshot
hashes; output validation does not retrieve or choose new evidence.

Normalized route failure or exhaustion returns the executor's completed
Generation Unavailable outcome, with no Decision Summary or Knowledge Answer
Citation and at most the existing frozen Generation Evidence Preview.
Unclassified application failures, malformed provider input, and malformed or
mismatched observed envelopes remain application failures, never insufficiency
or a successful fallback. Non-content route/approval identities, attempt
numbers, normalized reasons, timings, payload hashes, and snapshot hashes are
retained in the existing 30-day Operational Event boundary. Raw exceptions,
questions, answers, and evidence text are excluded.
Request-scoped attempt observations are emitted independently of successful
answer projection, including cancellation and application failure. Privacy
or policy output rejection is not an answer-structure retry reason.

## Compatibility And Evidence

This supersedes the permanent-one-provider restriction in workspace historical
ADRs `0010-use-one-approved-generation-provider` and
`0010-fail-closed-on-generation-provider-outage`, and the source ADR 0004 rule
that every unrecovered provider exception or missing completed provider call is
an application failure. ADRs 0003/0004's frozen evidence, privacy, transport,
and malformed-input rules are unchanged.

Legacy settings remain a compatibility configuration surface, not generation
approval; generic generation discovery is removed. Existing credentials do not
create an approved route during migration.

Deterministic fixtures activate only Local Development evidence and replace
provider I/O explicitly. They cannot establish real-provider behavior or
authorize Pilot use. Before Pilot, the owner must commission a controlled live
smoke for the exact declared provider/model/endpoint/data boundary and retain
failure, privacy, prompt/citation, and activation evidence. An unavailable
environment or missing approval remains an unmet acceptance obligation.
