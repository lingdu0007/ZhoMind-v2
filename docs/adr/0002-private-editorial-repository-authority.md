# ADR 0002: Private Editorial Repository Authority

Status: Accepted

Date: 2026-09-05

## Context

The product needs a retained editorial authority before Reviewed Release Bundle
intake, Candidate Build, and publication. Existing runtime document rows and
deployment copies cannot prove a reviewed revision, source definition,
role-separation decision, assurance support, or source freshness. Allowing
runtime material to write back into editorial records would reverse that
authority boundary and make later reconstruction unauditable.

## Decision

- Use the existing immutable `canonical_records` and append-only
  `canonical_events` tables as the Private Editorial Repository. Immutable
  `entry`, `editorial_revision`, and `source` records retain the authority
  snapshots; no schema migration is needed because the foundation already
  supports these canonical kinds and record classes.
- Store source definitions once. Record source availability as append-only
  `editorial_source_event/v1` facts and derive current availability from the
  event trail instead of mutating the source record. An Author's availability
  value is a proposal only: a new source begins
  `changed_or_unreachable_awaiting_review`, and an accepted Maintainer must
  append a qualified `verified_usable` fact before it can support review or
  export. That fact must use the source aggregate, bind the exact accepted
  Maintainer event, name a revision that contains the source, and use the
  accepted Maintainer as its recorder. Missing, foreign, malformed, or
  unqualified source trails resolve to `unknown`; T01 records this
  authenticated access decision rather than performing arbitrary runtime URL
  or locator fetches. A non-usable source blocks answer eligibility and export.
  Decisive loss is durable authority evidence; it can move a future published
  entry to Needs Re-review without creating a publication path in this ticket.
- Make the canonical coverage position, assurance level, source tier, source
  access scope, and revision change-kind vocabularies closed Python contract
  enums. Entry validation retains the complete decision schema, role facts,
  source links, and assurance-specific evidence requirements. Wording-only
  revisions normalize title and body whitespace only; punctuation, case,
  semantic tokens, comparison operators, and structured data require material
  review. High-impact requirements are conservatively inferred from authored
  text in supported languages as well as client-supplied claim metadata, so a
  client cannot lower a Claim-Evidence Link requirement by relabeling the
  content or by attaching an unrelated ordinary claim in the same section.
- Derive Author, Approving Reviewer, Maintainer, and Administrator authority
  from active server-side member rows. Authors, Reviewers, and Maintainers may
  work within their entry boundary; an Author or material reviser cannot
  approve that material revision. The assigned Maintainer must append a
  revision-specific responsibility acceptance before review, approval, or
  export. Administrators receive approved exports only and cannot inspect or
  modify private editorial records.
- Use explicit entry `schema_version` 1 from the first retained editorial
  revision. There is no prior editorial schema to migrate; an incompatible
  future version must provide its own migration before acceptance.
- For Release-Assured entries, resolve every named reference to a retained
  immutable or authoritative canonical record, never a derived record or
  replaceable projection. The frozen acceptance must be an immutable
  `delivery_acceptance_record/v1` that validates under its canonical schema.
  Its current active fact must be an `at_risk`-to-`active` `status_changed`
  event with `checks_verified`, a canonical member recorder, and verification
  attachments matching every passing or carried-forward selected check; its
  retained scope must cover the exact entry, contract, calibration, and named
  gate.
- Reconstruct deterministic, SHA-256-addressed `editorial_export/v1` artifacts
  only from retained authority records and events. Approval and lightweight
  acceptance retain a snapshot bound to the exact entry/revision, trusted
  roles, approval fact, qualified source definitions and availability event
  hashes, Release-Assured record/status hashes where applicable, and the
  approval-time audit cutoff. Only structurally valid lifecycle, maintainer,
  and approval events establish those facts. A new export checks current
  authority facts; historical reconstruction uses its retained approval
  snapshot only. Reject credential-shaped material, non-empty secret-bearing
  fields at every structured depth, and automatic-publication instructions.
  The export is not a bundle, does not invoke intake, and cannot write a
  Candidate, published knowledge version, runtime document, or deployment
  copy.

## Consequences

T01 now has a durable, access-controlled editorial authority and can prove a
revision/export hash from retained data. T02 must consume the export through
its own immutable Reviewed Release Bundle intake contract; T03 Candidate Build
and T04 publication remain separate responsibilities. Source availability is
already authoritative for fail-closed eligibility, while later publication and
maintenance paths must consume that retained evidence rather than infer it
from a runtime copy. Release-Assured references fail closed unless their
canonical records are appropriate authority records and the frozen
delivery-acceptance record actively covers the exact editorial authority.
