# ADR 0009: Bounded Interactive Admission And Non-Content Operations

Status: Accepted

Date: 2026-09-12

The daily-use Pilot replaces the historical five-concurrent-question limit
with two executing and two queued requests, at most one of each per member.
A process-owned admission gate and durable private Answer Execution events
separate bounded scheduling from retained user outcomes. Overflow is retryable;
restart closes interrupted executions without replaying provider work.
This supersedes only the concurrency portion of the earlier single-server
release decision, not its member, publication, upload or build-worker limits.

The supported Pilot has one API process and its existing Candidate worker.
Candidate work cooperatively yields at stage boundaries while interactive work
is admitted; no preemption of an already running external call is promised.
Multiple API processes require a shared admission authority before acceptance.
Stage timing is evidence, not proof of the later Ticket 29 latency objective.

Use the normative [Admission And Operations contract](../contracts/admission-and-operations.md).
Operations retains typed, closed non-content dimensions, never transcript
access. The independent retention policy from ADR 0007 remains authoritative.
Queue recovery does not publish, re-retrieve, or infer Insufficient Evidence.
