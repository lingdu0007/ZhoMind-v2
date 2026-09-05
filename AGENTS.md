# ZhoMind-v2 Agent Instructions

## Start Every Ticket

Work from the repository root. Before changing code, read the ticket, the
parent PRD requirements it names, this file, and the normative contract for
the domain being changed. Inspect the current implementation and focused tests
before choosing an approach.

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
- Drive behavior changes with focused tests first, then run the deterministic
  checks from `README.md` that cover the changed surfaces.
- When an English PRD, ADR, evidence report, or contract has a `.zh-CN.md`
  mirror, update the mirror in the same change and run
  `python3 scripts/check-docs-parity.py`.
- Update the local ticket only after its acceptance criteria are demonstrably
  met. Record the commands, results, and any remaining deployment-only risk
  under its `## Comments` heading.
- Run code review on the completed diff, address actionable findings, and
  create a focused commit.
