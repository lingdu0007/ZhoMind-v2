# ADR 0003: Deterministic Evidence Sufficiency And Frozen Answer Evidence

Status: Accepted

Date: 2026-09-06

Supersedes: the active-Pilot first-three selector, non-empty-context gate, and
candidate-derived citation projection.

## Context

Ticket 18 established an authorized, ranked Retrieval Candidate Pool but
deliberately stopped before answer sufficiency. The historic runtime path could
select the first three candidates, treat non-empty context as a gate, and
derive citations from the broader candidate list. Those shortcuts cannot prove
that a recommendation covers the user's decisive conditions, required
branches, source assurance, or known conflicts. They also permit a frozen
answer, a citation, and a provider payload to diverge.

ADR 0002 remains the authority for the editorial facts used to build the
authorized pool. This decision starts only after that authority-qualified
boundary and does not create a new publication, Candidate preview, transport,
SSE, history, UI, or provider-activation path.

## Decision

- Active Pilot answer planning accepts only a Candidate Pool identified as
  `retrieval-answer-policy/pilot-v1` with the `published_knowledge` scope.
  Any other input fails closed as `no_eligible_published_evidence`.
- Sufficiency is a deterministic, assurance-aware product rule. It evaluates
  an immutable Query Condition Set, candidate applicability, required
  question-shaped branches, reviewed `decision_query` coverage, complete
  Claim-Evidence Link support, release assurance, material conflicts, and
  review state. A Claim-Linked selected item requires every link of every
  supporting reviewed claim. Counts, raw ranking scores, non-empty context,
  retrieval order, and model judgment are not sufficiency inputs.
- The decision has exactly seven closed insufficient-evidence codes:
  `no_eligible_published_evidence`, `decision_not_covered`,
  `decisive_condition_missing`, `material_evidence_conflict`,
  `assurance_support_missing`, `evidence_budget_exceeded`, and
  `knowledge_needs_review`. An insufficient decision creates one structured
  reply and no Answer Evidence Set, recommendation, provider request, or
  citation identity.
- A sufficient decision searches canonical item identities and selects the
  smallest viable set that includes the governing recommendation or reviewed
  branches and every required complement. The governing item must cover the
  normalized question through its reviewed `decision_query`. The hard limits
  are three items, 1200 characters per Evidence Excerpt Snapshot, and 3000
  characters total. The authoritative source content length must fit the
  per-item cap; a truncated candidate preview cannot masquerade as complete
  evidence. Required evidence that cannot fit is rejected with
  `evidence_budget_exceeded`; it is never silently trimmed out of a
  sufficient answer.
- The selected evidence freezes before generation. Each selected item binds
  the entry, editorial revision, Published Knowledge Version, section,
  content-hashed chunk, source content length, and Evidence Excerpt Snapshot.
  The persisted item also contains the canonical identity binding used to
  compute its item identity; consumers must recompute it before trusting a
  snapshot or citation. The Answer Evidence Set identity binds the QCS,
  ordered selected item identities, and governing item. Each citation identity
  binds that set identity and one selected item.
- The provider-visible prompt is constructed only from the frozen Answer
  Evidence Set. Its normalized-question, QCS, evidence-source, and response
  contract regions remain structurally isolated. The same selected item
  identities, snapshots, citation identities, and governing citation appear
  in the frozen set and provider payload. Provider-visible source records
  carry `snapshot_id`, `item_identity`, and `citation_identity`; scores,
  chunk locators, and internal selection diagnostics are excluded from
  provider-visible and user citation data. Required response sections cite
  every material nonblank line and cannot introduce unknown or contradictory
  QCS assignments, secret values, unsupported quantified assurance, or a
  universalization of a bounded internal case.
- The historical selector, non-empty gate, and candidate citation path may
  remain only in the explicit lexical heuristic migration/diagnostic profile.
  They cannot influence active Pilot production sufficiency, frozen Answer
  Evidence Sets, provider evidence payloads, or product citation identities.

## Consequences

An answer either has a single immutable evidence basis or a structured,
closed-reason insufficiency reply. A provider cannot make the sufficiency
choice, add an unselected source, invent a citation, or turn retrieved text
into a change of conditions or policy. The outcome has a stable form for later
answer execution persistence and surface projections without making those
later responsibilities part of this ticket.

The decision makes the Pilot more conservative: incomplete branches,
unresolved conflicts, missing assurance, and over-budget evidence decline to
answer rather than produce a partial recommendation. This is intentional
because the evidence boundary is authoritative only when it is complete for
the declared QCS and answer shape.

## Scope

This ADR does not activate a provider, define HTTP or SSE transport, project
UI state, or define a new publication or withdrawal policy. The existing
tombstone operation redacts frozen evidence copies so this record shape cannot
leak a withdrawn excerpt; the broader history and withdrawal contract remains
with its named later ticket.
