# ADR 0001: Server-Derived Pilot Identity And Audit

Status: Accepted

Date: 2026-09-05

## Context

Pilot admission previously had reusable invitation semantics and browser/JWT
metadata that could be mistaken for authority. Identity lifecycle evidence also
needed a durable form without exposing credentials or private conversations.

## Decision

- Bootstrap Administrator creation uses server-held configuration and a
  database-enforced single-bootstrap invariant. Replays preserve the existing
  identity and password. A configured bootstrap persistence failure aborts
  startup rather than allowing a partially initialized pilot.
- Public registration consumes an active invitation through an atomic,
  conditional one-time claim and always creates a Knowledge User.
- Every protected handler derives active status and role from the current
  database member. Only System Administrators issue or revoke invitations,
  promote, deactivate, and use administrator capabilities.
- Authenticated logout deletes only its current `(sub, jti)` Redis session.
  Deactivation revokes every member session, blocks future login, and retains
  the account and conversation history.
- Identity lifecycle uses canonical immutable member/invitation records and
  append-only, content-free audit events. Unknown codes use a SHA-256-derived
  admission-attempt identity; known invitation denials keep the invitation
  aggregate. Audit stores minimum identities and SHA-256 references, never
  passwords, bearer tokens, invitation plaintext, request bodies, or
  conversation content. Pending audit commits precede Redis mutation so a
  session-store failure remains visible; a retry appends completion or another
  failure. Conditional claim races re-read the invitation after rollback before
  recording the final denial outcome.

## Consequences

The tail migration adds invitation-consumption fields and the portable partial
unique bootstrap index. It marks legacy unrevoked, unexpired invitations
consumed because the former reusable schema cannot prove they were unused.
Deployments must run it before relying on the new admission and bootstrap
guarantees, and administrators reissue invitations for any pending admission.
