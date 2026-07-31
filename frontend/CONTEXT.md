# ZhoMind UI Context

The frontend presents the authenticated RAG experience and the administrator
workspaces that operate its knowledge base and system settings.

## Language

**Knowledge User**:
An authenticated `user` account that asks questions and manages its own
conversation sessions.
_Avoid_: End user, consumer

**Authentication Entry**:
The standalone entry surface for signing in or registering before a user enters
an authorized workspace.
_Avoid_: Chat dialog, inline login

**System Administrator**:
An authenticated `admin` account that operates the knowledge base and System
Settings.
_Avoid_: Power user, operator

**Conversation Workspace**:
The `/chat` surface where a Knowledge User asks a question and reviews its
streaming answer, Evidence Summary, and sessions.
_Avoid_: Chat page, assistant screen

**Conversation Session**:
A Knowledge User-owned chronological conversation, identified by a session ID
and represented by its update time and message count.
_Avoid_: Thread title, saved search

**Evidence Summary**:
The concise, user-facing account of the sources and retrieval coverage that
supports an answer in the Conversation Workspace.
_Avoid_: RAG trace, debug output

**Retrieval Diagnostics**:
The administrator-only, expandable technical record of retrieval steps and raw
trace data for investigating an answer.
_Avoid_: Evidence summary, source citation

**Knowledge-base Operations**:
The administrator-only workspaces for ingesting and operating documents,
inspecting chunks, and monitoring asynchronous indexing work.
_Avoid_: Document page, upload page

**Document Library**:
The `/documents` workspace that inventories uploaded documents and offers
document-level actions.
_Avoid_: File browser, upload queue

**Indexing Job**:
An asynchronous build operation that prepares a document for retrieval and
reports its stage, progress, terminal status, and operational message in the
`/jobs` workspace.
_Avoid_: Upload, document status

**System Settings**:
The administrator-only configuration of model, retrieval, storage, and API
parameters that governs the running system.
_Avoid_: Local preferences, browser configuration

**Settings Application**:
The explicit administrator action that turns saved System Settings into the
active configuration, with a visible lifecycle state and audit metadata.
_Avoid_: Auto-save, instantaneous edit
