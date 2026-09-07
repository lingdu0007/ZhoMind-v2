# ADR 0004: Transport-Independent Closed Answer Execution Persistence

Status: Accepted

Date: 2026-09-06

Supersedes: transport-specific answer gates, generation and persistence
semantics, snapshot slicing, outcome inference, and source-count heuristics
that treated `ChatMessage.rag_trace` or a mutable chat message as the answer
authority.

## Context

ADR 0003 makes evidence sufficiency deterministic and freezes an Answer
Evidence Set before a provider may generate. That alone does not prevent a
normal HTTP handler, SSE handler, persistence path, reload path, or history
projection from making independent decisions. A mutable message body, a legacy
trace flag, a source count, or a later retrieval can otherwise turn the same
user request into inconsistent outcomes across surfaces, fabricate support
after an interruption, or re-slice a historical snapshot.

The product needs one private retained result per admitted question. It must
keep the user's decisive conditions explicit and editable before admission,
but frozen per completed turn. It must distinguish execution lifecycle from
the four completed answer outcomes, and it must fail as an application failure
when it cannot prove the retained result is complete and internally
consistent.

ADR 0003 remains the authority for deterministic sufficiency, the frozen
Answer Evidence Set, provider-visible evidence, and citation identities. This
ADR owns only execution admission, terminal persistence, and cross-surface
projection of that already-closed evidence decision.

## Decision

- An authenticated chat request creates one private Answer Execution with an
  immutable `answer_execution_request/v1` header and append-only
  `answer_execution_event/v1` records. It is retained with the user's
  private conversation and may be deleted only by the same verified
  conversation deletion or expiry path. It is not a global canonical-record
  aggregate. Event sequence is unique per execution; a terminal, redaction,
  or stream-delivery lifecycle writer locks that execution before it reads and
  appends its next event. Conversation deletion and retention purge acquire
  those execution locks before deleting an event trail or header, so private
  retention cannot leave orphaned execution events. On SQLite, admission,
  event writers, deletion, and expiry share a transaction-wide writer fence
  before their relevant reads: a fresh transaction begins it with `BEGIN
  IMMEDIATE`; an already-open deferred read transaction upgrades it with a
  no-op `chat_sessions` write. Cleanup locks matching execution headers, locks
  the verified conversation session, then re-scans headers before delete;
  admission locks that verified session behind the same fence. Thus an
  admission or event write cannot race private deletion or expiry into an
  orphaned header, event, or message. Admission durably binds the request
  header to its exact user-message identity before execution begins.
- The execution state path is `admitted -> queued -> running`, followed by
  exactly one terminal state: `completed`, `stopped`, `failed`, `throttled`,
  or `rejected`. `completed` is not an answer outcome. It has exactly one
  outcome: Evidence-Gated Answer, Insufficient Evidence Reply,
  Non-Knowledge-Base Reply, or Generation Unavailable. The other terminal
  states have no completed outcome and cannot expose a fabricated answer,
  evidence, citation, or provider result.
- Admission stores an explicit QCS with its normalized question. Conditions
  may be supplied explicitly, edited for each new turn, or inherited only
  from the latest completed execution in the same private conversation and
  for the same owner. Inherited conditions are copied into a new QCS bound to
  the new normalized question and retain source-execution provenance. The
  inherited QCS has a new per-turn identity even when the normalized question
  is identical. A new conversation starts without inherited conditions and
  clears the composer draft. A retry with any retained execution, including a
  non-completed SSE or history projection, submits that turn's frozen QCS
  explicitly. Missing client-side terminal data never authorizes
  re-inheritance: a retry marked as inherited but without its retained
  execution fails locally until that execution is recovered. Only a
  pre-admission transport failure can repeat the original requested QCS or
  inheritance request. A streamed execution is accepted only after it binds to
  the submitted turn: an implicit QCS must exactly equal conditions derived
  from the submitted question, and an inherited QCS must name a completed
  source execution already projected in the same private conversation and copy
  its exact conditions. A completed execution gains retry authority only after
  its full terminal projection validates; a contradictory terminal projection
  clears that local authority until the persisted execution is recovered.
  Neither hidden memory nor a global profile may supply a decisive condition.
- The narrow non-knowledge-base allowlist is chosen before retrieval and
  produces no knowledge claim, evidence, citation, or provider call. Every
  other admitted request uses the QCS that admission recorded. A missing
  decisive condition remains a reviewed conditional branch or a structured
  insufficiency, never an inferred condition.
- A completed result persists exactly one normalized question, QCS and
  provenance, completed outcome, answer text, Evidence Set identity, ordered
  item identities, snapshot identities, and knowledge-version identities.
  Evidence-Gated Answer and Generation Unavailable additionally persist the
  provider-input identity record containing the same values. The owner-visible
  completed projection carries the exact assistant binding, frozen answer text,
  and frozen evidence summary, so transports compare the retained projection
  instead of inferring one. An Insufficient Evidence Reply retains one
  structured reply with its outcome, exact reason, and QCS identity; no other
  completed outcome retains it. No later consumer may substitute any of those
  values. The request header binds its exact user message, and every completed
  result or assistant-backed terminal binds its exact persisted assistant
  message. If assistant persistence itself fails after admission, completion
  rolls back and the existing user binding may retain only a `failed`
  `ANSWER_EXECUTION_PERSISTENCE_FAILED` terminal with no assistant, answer
  text, or outcome. `ChatMessage.answer_execution_id`
  is a mutable lookup index, so reload and history discover and validate the
  immutable binding in the same private conversation; clearing that index
  cannot demote a bound message to a legacy trace projection, moving the
  message to another private conversation fails closed, and a contradictory
  non-null index fails closed. Generation Unavailable retains frozen
  identities only to prove its closed boundary: its answer projection contains
  no knowledge claim, citation, source, or evidence preview and is never a
  supported-answer projection.
- Normal HTTP, terminal SSE events, linked private persistence, reload, and
  history project the retained result only. They must not perform retrieval,
  reselect evidence, re-slice snapshots, invoke generation, or infer outcome
  from message text, gate flags, source count, score, or a legacy trace.
  `ChatMessage.rag_trace` may remain a bounded diagnostic compatibility
  artifact, but it is not semantic input for any linked Answer Execution. For
  a linked execution, its persisted form contains only bounded operational
  metadata such as gate state, steps, counts, provider identities, timing, and
  error classifications. It excludes the question, QCS, answer text or
  preview, evidence content, provider-visible generation envelope, and private
  history. A client may accept a completed SSE terminal only when the assistant identity
  equals the execution binding; completed execution identity and state, answer
  text, normalized question, complete non-duplicated QCS and provenance,
  explicit outcome, and full outcome-specific evidence summary all equal the
  frozen execution projection and the question, QCS, and provenance equal the
  normalized submitted turn. An Insufficient Evidence Reply carries one
  structured reply whose outcome, reason, and QCS identity match exactly in
  both terminal stream and execution; no other outcome carries it. A valid
  terminal is only a fully framed SSE `event: done` whose data payload is the
  exact unquoted literal `[DONE]`. EOF before that frame's blank-line
  separator, a quoted or otherwise altered marker, or that marker on any other
  event is a missing or contradictory terminal. A repeated `done`, or any
  semantic frame after `done`, is an application failure. Missing, repeated,
  or contradictory terminal fields are application failures, never a cue to
  fabricate an insufficiency or another replacement outcome. An error or
  cancellation after completed terminal fields is an
  application failure, not a stop, and clears the completed outcome,
  structured reply, evidence summary, and diagnostics from the client
  projection.
- Authentication failure occurs before admission. An unrecovered retrieval or
  provider failure, and every execution, stream, or persistence failure, is an
  application failure, never insufficiency. A retriever may retain a
  diagnostic for a recovered fallback only when it returns an actual candidate
  result; it may not synthesize an empty result from a provider exception and
  label that as insufficient evidence. A stream interruption while the
  execution is running cancels and awaits the durable stopped event. SSE
  admission appends `stream_delivery_pending` before a closed result can be
  delivered. A completed result with that pending record but without a delivery
  completion or interruption record is not projectable through reload or
  history: it is an application failure, not a completed-answer replay. Normal
  HTTP does not create an SSE delivery lifecycle record and may project the
  same closed result after ordinary persistence completes. If delivery
  interrupts after projection of a closed result has begun, an
  append-only `stream_delivery_interrupted` event keeps the closed result
  immutable but makes it non-projectable; reload and history fail as an
  application failure instead of changing `completed` into another terminal
  state. The response layer performs the same cleanup for an ASGI send
  exception or disconnect, and considers `done` delivered only after an outer
  ASGI transport observer writes the terminal body and successfully completes
  response finalization beyond every response-buffering middleware, then
  appends `stream_delivery_completed`. A missing
  terminal event, contradictory terminal state
  payload, or contradictory persisted message binding is an application
  failure on projection. Persistence failure rolls back uncommitted completion
  rather than leaving a partial completed execution; if an assistant still
  cannot persist, the owner sees only the explicit failed persistence terminal
  on the frozen user turn. A post-admission SSE failure projects the retained
  non-completed execution and frozen QCS, plus an assistant binding when one
  exists, before its `error` and `done`; it exposes no completed outcome or
  evidence summary. Before a provider call, the complete provider-visible input
  must equal the one record built from the frozen question, exact QCS identity
  and ordered condition records, Evidence Set, every source field including
  citation marker, ordered item and citation identities, source-recomputed
  ordered snapshots, knowledge-version identities, and the complete response
  contract when present. JSON object parsing rejects a duplicate key at every
  nesting depth; a later key must never silently overwrite an earlier frozen
  field. The comparison does not trim, normalize, omit, or reconstruct
  provider-visible fields. A mismatched or malformed input, an observed
  generation envelope that is present but malformed, or a mismatched valid
  observed generation envelope, is an application failure rather than
  Generation Unavailable. An unrecovered provider exception, or a route
  configuration that produces no completed provider call, is also an
  application failure; Generation Unavailable is reserved for a completed
  provider call with no usable frozen-boundary answer.
- A separately authorized document tombstone may append an evidence-redaction
  event to a completed private execution. The projection removes the
  historical excerpt and marks the item withdrawn while retaining the
  Evidence Set, item, snapshot, and knowledge-version identities. The
  original terminal event is never rewritten. Completion locks the frozen
  evidence documents before appending its terminal event, and tombstoning
  locks the same documents before marking them withdrawn. If completion sees a
  tombstoned document after acquiring that lock, it appends the redaction event
  in the same transaction and immediately returns the redacted projection.

## Consequences

Every transport exposes one result rather than independently reproducing
answer semantics. A user can inspect the per-turn QCS and condition provenance
in their own private history, revise conditions in a later turn, and know that
history does not silently acquire conditions from memory or a different
conversation. A stopped or failed request is visibly distinct from an
insufficiency reply and cannot be upgraded by reload. A closed result whose
SSE delivery was interrupted is deliberately not replayed as a completed
answer: its retained interruption event causes later projections to fail
closed. An assistant persistence failure is likewise not silently erased or
converted into a reply: its retained failed execution remains visible on the
admitted user turn without inventing an assistant message.

Private execution records are intentionally not global canonical records:
their content follows private-conversation retention and deletion. Their
immutability is therefore while retained, with append-only events providing
the audit boundary needed for reload and cross-surface equality. Historical
withdrawal redaction preserves identity while preventing a retained excerpt
from continuing to expose withdrawn content.

The legacy lexical heuristic migration profile may continue to produce
diagnostics, but it cannot make a product Evidence Set, completed supported
answer, or Generation Unavailable result. It therefore freezes an explicit
insufficiency rather than reactivating a legacy gate or provider path.

## Scope

This ADR does not activate or approve a provider route, define provider
fallback policy, create publication, replacement, or withdrawal authority,
change feedback retention, or add later UI workflows. It defines only the
transport-independent execution boundary and the projection of an already
authorized evidence decision. Document tombstone authority remains outside
this ADR; this ADR specifies only the private historical-redaction effect once
such a tombstone is authorized.
