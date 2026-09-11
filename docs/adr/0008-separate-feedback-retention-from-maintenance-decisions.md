# ADR 0008: Separate Feedback Retention From Maintenance Decisions

Status: Accepted

Date: 2026-09-11

## Decision

Keep explicit feedback in its deletable, bounded-retention store and retain
maintenance decisions as canonical immutable snapshots with append-only
events. Separate deletable links connect retained signals to decisions.
The [Knowledge Feedback And Maintenance contract](../contracts/knowledge-maintenance.md)
governs qualification, independent verification, containment and cadence.
This extends ADR 0007 and replaces automatic Helpful work creation with
adoption-signal capture; legacy review rows do not acquire canonical authority.

Do not promote raw or merely anonymized conversations into durable Findings.
Work Owners deliberately author and verify non-personal scenarios, and
Maintainers approve their normalized results. A replay must establish its
actual conditions, publications and retrieval profile; historical ownership
alone cannot establish current execution provenance.

Retain only the independently verified scenario's request digest and full
Query Condition Set identity in its immutable fixture, not a replay request
body or a private answer identity. Each replay takes an explicitly selected,
currently owned private execution and verifies both bindings before running it.
The deletable signal's structured-condition digest relates an independently
authored scenario to the report without requiring identical question text.
Reject retaining a question in an immutable fixture merely because a Work
Owner checked a synthetic-material confirmation: that creates an undeletable
conversation copy. The trade-off is that deleted private scenario input must
be independently reauthored before replay, rather than reconstructed from
maintenance evidence.

Make unresolved P0/P1 decisions effective at ordinary retrieval and answer
completion, rather than relying on a suspended content acceptance to change
editorial eligibility indirectly. A late block and answer completion serialize
through the same write fence. Explicit Work Owner verification may ignore only
its own item in a request-local, currently qualified context; other blocks,
editorial/source authority and Provider admission remain intact. Retain that
context's closed provenance in the fixture/replay. This permits verification
before re-acceptance without reopening ordinary traffic during repair.

For P0/P1 Provider diagnosis, extend ADR 0005 with explicitly authorized
request-local verification while ordinary route capture remains suspended.
The administrator signs exact work/scenario/publication bounds and a separate
fully approved route admission; the ordinary-member Work Owner performs the
non-personal reproduction or approved repair replay. Both admission and
finalization revalidate authority. Retain authorization provenance instead of
fabricating a route activation. This does not weaken Provider/data approval,
publish a route, or replace post-repair acceptance. The rejected alternative,
activating a route merely to test whether its repair works, exposes ordinary
traffic before the bounded repair scenario has been verified.

## Consequences

Investigating a report may require independent reproduction instead of opening
the reporter's conversation. This costs work, but allows signal deletion and
expiry without erasing legitimate non-personal decisions or retaining a hidden
transcript. The rejected alternative, copying feedback into an immutable event
or Finding and attempting to redact it later, contradicts both append-only
authority and verifiable deletion.

Maintenance never changes published knowledge automatically. A repair uses
the existing editorial, source, acceptance and publication authorities.
Roadmap qualification records a separate bounded decision, not permission to
implement new scope. The additive `20260911_0027` migration creates only
deletable maintenance-signal links and invents no legacy ownership or findings.
