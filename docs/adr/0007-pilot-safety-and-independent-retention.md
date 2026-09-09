# ADR 0007: Pilot Safety And Independent Retention

Status: Accepted

Date: 2026-09-09

Extends ADRs 0001, 0002, 0004 and 0006 without changing identity,
publication, closed-outcome or withdrawal semantics.

## Decision

Use the normative [Pilot Safety And Retention contract](../contracts/pilot-safety-retention.md).
Source admission is an immutable part of accountable editorial review.
Missing all-member restricted-sensitivity authority is not inferred from old
public/internal labels. Existing artifact identities remain unchanged.

Retain separate 30/30/180-day defaults and immutable versioned policy changes.
Protect updates through exact-identity comparison and affected acceptance
suspension. Configuration cannot enable transcript browsing, disable expiry or
combine data classes. Independently scheduled cleanup uses the existing private
writer fences and a new bounded non-content observation table. Failed or
unverified cleanup blocks deployment acceptance, not merely a dashboard counter.

Descriptions live only in raw feedback. An additive migration removes legacy
review-item copies. Reads join retained signals; deletion/expiry severs references
while preserving already-classified non-personal decisions. Ticket 27 still owns
the complete maintenance workflow.

## Consequences

Deployments must apply `20260909_0026`. Sources lacking reviewed admission
require explicit re-review, not invented authority backfill. Local checks cannot
establish public HTTPS/network isolation; the exposure tool labels configuration-
only runs as live non-passes. No universal PII, secret-detection, network security
or Prompt Injection guarantee follows.
