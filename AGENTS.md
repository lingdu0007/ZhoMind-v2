# ZhoMind-v2 Agent Instructions

## Start Every Ticket

Work from the repository root. Before changing code, read the ticket, the
parent PRD requirements it names, this file, and the normative contract for
the domain being changed. Follow `docs/agents/ticket-execution.md` for the
required claim, TDD, verification, review, and closure sequence. Inspect the
current implementation and focused tests before choosing an approach.

The ticket defines the scoped deliverable and acceptance evidence. A ticket
must not weaken a retained product invariant. When it intentionally replaces a
documented invariant, add or update the governing ADR and its Chinese mirror
in the same change.

## Normative Documentation

- `docs/contracts/` defines stable product contracts: identities, states,
  authority, data boundaries, audit shapes, compatibility, and product-path
  requirements.
- `docs/contracts/canonical-product-contracts.md` is authoritative for
  canonical records/events and Pilot identity authority. Read its Pilot
  Identity Authority And Audit section for admission, roles, sessions,
  identity audit, and the Ticket 14 compatibility path.
- `docs/adr/` records accepted, hard-to-reverse decisions and their
  consequences. ADR 0001 explains the server-derived Pilot identity and audit
  decision; it does not replace the detailed contract.
- `README.md` indexes the normative documents and records the deterministic
  local gate. Its Chinese mirror is a reader-facing translation, while English
  contract and ADR files are canonical for agent work.

## Working Language

- Use Chinese for user-facing progress updates, review summaries, questions,
  blocker reports, and final responses.
- Preserve the canonical language of repository artifacts. English PRDs,
  contracts, ADRs, and existing English ticket prose remain English; update
  their Chinese mirrors when required.
- Keep commands, code identifiers, schema values, protocol names, error text,
  and quoted source text in their original form when translation would reduce
  precision.

## Durable Engineering Rules

- Treat the server's current persisted facts as the authority for identity,
  role, resource ownership, state, and eligibility. Browser state, client
  inputs, and stale compatibility data are not authority.
- Preserve fail-closed behavior whenever legacy data cannot prove a required
  fact. Temporary adapters may preserve final invariants only and need a
  removal condition.
- Add database changes as a new Alembic revision after the current head; never
  rewrite an applied migration. Test the supported migration path where the
  ticket requires persistence behavior.
- Do not retain credentials, bearer tokens, invitation plaintext, private
  conversations, full request bodies, or secrets in audit events, diagnostics,
  logs, test artifacts, or public evidence.
- Keep a member's private data private. Administrator or operational surfaces
  may expose only the projection explicitly allowed by the relevant contract.

## Delivery Discipline

- Preserve unrelated worktree changes. Do not reset, revert, or stage files
  outside the ticket's scope.
- Before editing, record the ticket owner or session, starting `BASE_SHA`,
  current branch, and pre-existing worktree changes. Stop when ticket ownership
  is uncertain.
- Before writing tests, map every acceptance criterion to an approved public
  test seam, an observable assertion, and any required negative case. The
  repository owner has pre-approved the default seams and automatic risk
  profiles in `docs/agents/ticket-execution.md`; ask only when a ticket truly
  requires a new seam outside that policy.
- Drive behavior changes as vertical TDD slices: one failing test at an
  approved seam, the minimum implementation, and the affected focused tests
  before the next slice.
- For protected or cross-surface behavior, complete the review-risk matrix in
  `docs/agents/ticket-execution.md`, including authority, immutable identity,
  fail-closed behavior, state transitions, retries and races, legacy
  compatibility, privacy, and projection equality.
- Verify in layers: environment and baseline smoke, focused tests,
  authenticated product-path tests, the applicable deterministic checks from
  `README.md`, and separately declared deployment-only checks.
- Never represent a skipped or unavailable Milvus, provider, PostgreSQL,
  persistent-stack, ingress, or production check as completed acceptance.
- When an English PRD, ADR, evidence report, or contract has a `.zh-CN.md`
  mirror, update the mirror in the same change and run
  `python3 scripts/check-docs-parity.py`.
- After focused and product-path verification passes, create one provisional
  focused commit and review `BASE_SHA...HEAD` with both standards and
  specification reviewers. Add a regression test for every behavioral
  finding, amend the commit, and obtain fresh reviews of the final diff.
- Do not mark a ticket resolved when a required reviewer is unavailable,
  blocking findings remain, the final diff differs from the reviewed diff, or
  an acceptance criterion lacks recorded evidence.
- Keep the local ticket current throughout the work. Record the claim,
  acceptance mapping, verification checkpoints, review findings and re-review,
  final commit, and remaining deployment-only risks under `## Comments`.
