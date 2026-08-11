# ADR 0003: Adopt The Evidence-Aware Research Workbench Direction

Status: Accepted

Date: 2026-07-30

## Context

The existing frontend mixes chat, document operations, browser-local
configuration, authentication, and raw RAG details in a small number of views.
The redesign prototype tested whether a shared workbench could separate normal
knowledge work from administrator operations without creating a second product
shell.

## Decision

Adopt the confirmed prototype direction as the UI and interaction reference:

- Both roles land in Conversation Workspace inside one shared application
  shell with a narrow persistent primary rail.
- Knowledge Users see Conversation Workspace only. System Administrators gain
  Document Library, Indexing Jobs, and System Settings in that same rail.
- Evidence Summary is the default answer companion. Source excerpts open in a
  side drawer. Retrieval Diagnostics is an administrator-only disclosure.
- Authentication becomes a standalone entry surface. Administrator registration
  shows an invitation-code field only for that registration role.
- Document Library and Indexing Jobs stay distinct views. Initial uploads use
  the general strategy automatically; only rebuilds expose advanced chunk
  strategy selection.
- System Settings is a server-managed administrator surface with explicit
  draft, save, and apply states. Browser-local configuration is not presented
  as active system configuration.
- The visual system uses warm paper, ink, copper, and moss in a restrained
  research-workbench style, with Chinese serif headings and sans-serif UI data.

## Consequences

Future implementation tickets must use the accepted prototype as their visual
and interaction reference while rewriting it in production Vue components.
They must add server-derived role guards and a protected System Settings API
before enabling administrator settings. This decision does not authorize
copying prototype code into production or weakening existing backend document
authorization.
