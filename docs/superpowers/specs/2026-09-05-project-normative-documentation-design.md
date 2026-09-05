# Project Normative Documentation Design

Status: approved for implementation

Date: 2026-09-05

## Goal

Make durable project rules easy for an implementation agent to find without
copying a long restatement of those rules into every ticket prompt.

## Decision

ZhoMind-v2 will use a layered documentation model:

1. `AGENTS.md` is the short project-level execution entry point. It directs an
   agent to the ticket, the relevant PRD requirements, and the applicable
   normative contract. It also holds durable engineering rules for migrations,
   privacy-safe audit data, verification, documentation parity, and completion.
2. Documents under `docs/contracts/` are the normative product contracts.
   `canonical-product-contracts.md` is already the authority for Pilot identity,
   administrative authority, identity audit, one-time invitation consumption,
   logout, deactivation, and compatibility behavior.
3. ADRs under `docs/adr/` record irreversible implementation decisions and
   their consequences. ADR 0001 explains the rationale for the Pilot identity
   rules but does not duplicate the complete contract.
4. The README contains a brief index identifying the normative documents and
   their responsibilities.

The repository will not add a second `pilot-identity-authority` contract. That
would duplicate the authoritative Pilot Identity Authority And Audit section
already in `canonical-product-contracts.md` and risk later divergence.

## Responsibilities

`AGENTS.md` must stay short and procedural. It must not enumerate individual
routes, event values, or state transitions. It tells an agent how to discover
the governing contract and how to complete work safely.

`docs/contracts/canonical-product-contracts.md` owns stable identities, allowed
state transitions, authority boundaries, privacy-safe event shapes, migration
compatibility, and product-path requirements. Its Chinese mirror changes in
the same commit whenever the English contract changes.

`docs/adr/0001-server-derived-pilot-identity-and-audit.md` owns the historical
decision record for server-derived Pilot identity authority. Its Chinese mirror
remains paired with it.

The README index helps a person or agent select the right document without
turning the README into a restatement of the contracts.

## Ticket Prompt Pattern

A future ticket prompt should name only the worktree, ticket, relevant PRD
requirements, and relevant contract. It should then require focused TDD,
deterministic verification, ticket evidence, review, and a focused commit.

For Ticket 14, the prompt will reference `AGENTS.md`,
`docs/contracts/canonical-product-contracts.md`, ADR 0001, and the Ticket 14
issue. It will not repeat identity lifecycle requirements that the contract
already defines.

## Scope

The implementation adds:

- a project-level `AGENTS.md`;
- a concise normative-documentation index to `README.md` and `README.zh-CN.md`;
- a small status clarification in the English and Chinese canonical contract
  that designates it as normative for the existing Pilot identity rules.

It does not change application code, runtime behavior, migrations, tests,
existing ADR decisions, or the content of Ticket 14.

## Review Checklist

- The source of truth for each rule is unambiguous.
- No new document duplicates the existing identity contract.
- The agent entry point is concise enough to be read on every ticket.
- README links resolve and English/Chinese README content stays equivalent.
- The short Ticket 14 prompt names the governing documents rather than
  restating them.
