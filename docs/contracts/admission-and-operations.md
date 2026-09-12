# Admission And Operations

Status: normative; Ticket 28, KB-SO-004/005, KB-SL-001 and operations portions
of KB-RL-003, KB-UX-002 and KB-RA-007. ADR 0009 records the decision.

## Admission

`pilot_admission/v1`, version 1, admits at most two executing and two queued
questions across the single API process. Each authenticated member has at most
one executing and one queued request, including maintenance diagnosis.
The configuration exposes these bounds, a 30-second queue timeout, and a
canonical SHA-256 `configuration:` identity. The public operations snapshot
contains aggregate counts, not member or conversation identifiers.

Reservations are released once on success, failure, timeout or cancellation.
The timeout deadline is fixed when the request queues; expiry is decided under
the admission lock before promotion, including when the waiter wakes late.
The oldest eligible queued member starts when capacity becomes available;
a member already executing cannot occupy a second executing position.
`CHAT_MEMBER_LIMIT` or `CHAT_QUEUE_FULL` is a retryable throttled rejection
before private execution admission. It creates no fictitious completed answer.
Browser-created conversations use random UUIDs, not wall-clock identity.

An admitted request commits its queued private execution before awaiting a slot.
SSE exposes waiting/position, reserved answer identity, start and the retained
terminal execution. Normal HTTP waits for the same retained result.
Queued requests recheck active persisted membership before starting.
`CHAT_QUEUE_TIMEOUT` is a retryable failed execution with no completed outcome;
its identity and failure are visible in owner history. Queuing and overload
never mean Insufficient Evidence. Existing owner, conditions, frozen evidence,
generation-route capture and delivery fences remain authoritative.

## Build Priority And Recovery

Candidate work waits before claiming a new job when interactive work exists.
An already claimed build yields at cooperative stage boundaries, commits its
transaction while waiting, renews its exact attempt lease, and revalidates
authority before proceeding. In-flight external calls are not preempted.
The existing one-worker bound, cancellation, derived cleanup and publication
separation remain unchanged.

Before serving requests, startup reconciles retained queued/running Answer
Executions to failed `ANSWER_EXECUTION_INTERRUPTED` terminals. Reconciliation
is idempotent and does not replay providers. Existing Candidate startup recovery
retains dispatched queue work and reconciles interrupted attempts through its
lease and cleanup protocol. Retry remains an explicit operation with immutable
input checks; restart does not grant publication authority.

## Non-Content Events

The additive `20260912_0028` migration adds nullable operational dimensions
without rewriting existing events. Producers and administrator reads sanitize
both new and legacy values:

- Request identity is canonical UUIDv4 or `unknown-request`; route class is a
  registered method/path template plus response-status class, never a raw URL.
- Execution states and outcomes use the closed Answer Execution vocabulary.
  Counts and stage durations are nonnegative integers, never booleans or text.
  Stages are queue, application, retrieval, provider, persistence and stream.
- Gate outcomes are passed/rejected/unavailable. Provider routes retain only
  canonical route/configuration/hash identities, bounded attempt counts,
  durations and closed fallback results, not provider labels or exception text.
- Effective retention policy and admission configuration use hash identities.
  Unknown error strings become `APPLICATION_FAILED`. Provider, retrieval,
  persistence, stream, queue and application failures have distinct categories.
  A generation-route `application_failure` remains an application failure.
  Invalid legacy total durations project as null rather than content or a
  fabricated measurement.
- Questions, answers, excerpts, bodies, credentials, tokens, arbitrary labels
  and unknown fields are excluded even when injected into an allowed field.

Events remain in the independent 30-day default retention class from
[Pilot Safety And Retention](pilot-safety-retention.md), not conversation
storage. Deleting a conversation cannot delete independently retained events.
The existing scheduled cleanup and versioned policy-change authority apply.

## Administrator Surface And Evidence

Administrator-only `/operations` shows aggregate admission, configuration,
bounded recent timings, normalized failures, route results and recovery links.
Candidate and legacy build counts are included. Supported retry actions point
to existing authorization-checked operations; private conversation access is
never needed. Knowledge Users cannot open this surface.

Local authenticated HTTP/SSE/history, browser and migration tests establish
bounded deterministic behavior. They do not establish sustained load latency,
multi-process scheduling, live providers, Milvus, PostgreSQL deployment,
network exposure or production acceptance. Ticket 29 owns workload-conditioned
latency measurement; this contract makes no unmeasured latency guarantee.
