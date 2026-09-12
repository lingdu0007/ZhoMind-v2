# Pilot Measurement Contract

Status: normative measurement boundary for Ticket 29; not a live acceptance report.

## Authority And Identity

The authenticated product path remains the measurement seam. Administrator
`GET /api/v1/operations/measurement-snapshot` returns only fingerprints and
counts from the same Authorized Retrieval Candidate Pool used by chat, before
ranking. It also binds current approved route, admission configuration,
retrieval policy, evidence envelope, product revision, deployment, observed
host CPU/memory and one-worker configuration. Host measurements describe the
OS-visible host, not an independently verified container quota.

`MeasurementBinding` includes those identities, corpus and eligible chunk count,
the hash of the exact admitted `(entry, editorial revision)` set,
concurrency, host class, exact timezone-aware start/end and explicit
`local_deterministic` or `controlled_live` mode. Missing active route or product
revision prevents construction of a complete workload binding. CLI declarations
must match the server snapshot before and after a request suite or bundle run.
Every required snapshot field is validated; an incomplete response fails closed.
Changing a bound value creates a different fingerprint; a sample from another
binding or outside its half-open observation window is rejected. A snapshot
does not activate a route, publish a Candidate, or change answer eligibility.

Administrator `GET /api/v1/operations/requests/{request_id}` retrieves one exact,
unexpired chat Operational Event. Missing, duplicate, expired or non-chat
observations are unavailable. It never reads a conversation. The existing
event sanitizer and independent retention policy apply to both new and legacy
values. No migration is required: existing event dimensions hold timings.

## Measurements And Reports

The versioned profile declares eight distinct admitted Knowledge Users, two
executions, two queued positions, one of each per member, one worker, 50 active
entries and at most 10,000 eligible chunks. The request runner issues 40 c1,
40 c2 and ten four-request burst rounds through authenticated SSE. It checks
eight distinct current member identities rather than counting bearer strings.
Each burst requires an observed simultaneous two-executing/two-queued snapshot;
having four responses or separate historical queue states is insufficient.

Acknowledgement is the first admission identity or queued/running frame.
Meaningful processing is the first running/retrieval/generating/persistence
frame. Actual content time is the first nonempty content frame, never a header,
progress frame or Generation Evidence Preview. Closed-outcome time requires
one consistent completed execution/outcome and the fully framed terminal
`done: [DONE]`; interrupted, duplicate or contradictory terminals cannot pass.
These are transport-observed times, not model-token emission times. The current
product emits content after answer validation and persistence.
The collector also reloads this new conversation using the requesting member's
credential. Its streamed execution, answer identity, outcome, complete content,
evidence summary and insufficiency projection must equal that canonical retained
message. Only hashes and timings survive collection; administrators never
perform this private reload.

Queue, retrieval, provider and persistence durations come from the exact
administrator Operational Event. Provider attempt durations are retained
individually; the provider-stage duration must cover every attempt.
Application-controlled elapsed time is client total minus the provider stage,
so it includes queue, application work and transport overhead. It is not an
isolated CPU measurement. Missing critical dimensions make that objective
unavailable, not zero. Known unexecuted chat stages are explicitly recorded as
zero by the producer.

All valid authenticated admitted requests remain in the measurement population,
including completed insufficiency, non-knowledge replies and Generation
Unavailable. Invalid, unauthorized, canceled, throttled and rejected populations
are separate. An admitted failed execution has no closed outcome and makes
closed-outcome evidence incomplete; it cannot be dropped to improve a percentile.
Content, question text, credentials and arbitrary errors are not report fields.

Reports retain exact values, count and maximum. At 20 or more eligible samples
they additionally report nearest-rank P50, P95 and observational P99. P99 is
observational, not a separately accepted service objective. Missing samples,
partial rounds and undersized windows cannot claim percentile compliance.
Acknowledgement P95 is at most 1,000 ms; processing P95 at most 2,000 ms;
c2 closed-outcome P95 at most 30,000 ms. Every admitted burst request must close
within 60,000 ms, including its queue and complete provider route.

`measure_bundle` imports an explicitly supplied reviewed manifest and explicitly
dispatches each fresh job. It never publishes. Ten-item and fifty-entry runs
retain per-item hashes, attempts, frozen inputs, embedding configuration hashes,
actual execution times and Candidate chunk counts. An old completed job or
no-op is not a rebuild. Partial, canceled, failed, superseded or retried work is
not a failure-free batch pass. Ten items have a 30-minute budget; fifty entries
have 60 minutes; individual Candidate execution has a five-minute P95 objective,
subject to the same minimum-sample rule.
Build observations retain actual start/end and validate both the run and job
timestamps against the bound window. Fifty-entry rebuilds must use `replace`
operations for exactly the snapshot's immutable entry/revision set, not fifty
new entries or different revisions.

## Availability And Decisions

`collect_probe` checks health and, when requested, a declared sanitized
knowledge question through the authenticated product path. An ordinary social
reply cannot satisfy the knowledge-path probe. Valid Evidence-Gated Answer or
Generation Unavailable is product-available; the latter remains a route failure.
No private production conversation is sampled.

Availability aggregation requires an explicitly declared, non-overlapping
minute-aligned core schedule spanning all four weeks of an exact 28-day binding.
One health observation is required each minute and a product observation no more
than five minutes old. Missing minutes remain in the denominator and missing
coverage makes acceptance unavailable. Probe inputs outside the binding, after
the observation cutoff or duplicated in a minute are rejected.

The evaluator's `TriggerObservation` binds an independently retained evidence
hash and exact measurement conditions. Its `values` carry durations in ms,
except: queue saturation uses distinct offsets 0..6 in a rolling week;
provider-unavailable observations use occurrence offsets in that week;
route success uses `(successful, 20)`; availability uses `(available, expected)`.
Consecutive percentile values require matching eligible counts of at least 20
and complete consecutive evidence windows. `complete_window` is an evaluator
attestation, not a server authorization flag or proof of elapsed calendar time.

The decision evaluator covers every KB-SL-004 trigger: 80-percent headroom or
earlier retrieval failure; consecutive 30-second normal/provider P95 misses;
two capacity-related 60-second misses; three queue-saturated days in seven;
two provider-caused unavailability outcomes in seven days; sub-95-percent route
success; a 30-minute real-work provider outage; two slow ten-item bundles;
one slow fifty-entry rebuild; sub-99-percent four-week availability; one
four-hour reconstruction miss; and two fifteen-minute restart misses.
It emits content-free At Risk records with evidence, hashed owner, required
action and review deadline within seven days. Allowed resolutions are
remediation, narrower commitment, re-acceptance or suspension.

These records are evaluator artifacts, not automatic Delivery Acceptance status
mutations. The responsible administrator/evaluator must retain the weekly
disposition and append the affected existing Delivery Acceptance status event.
No artifact grants publication, enables fallback, revokes valid content, or
automatically restores acceptance.
Request and build runners feed their actual reports directly to `assess_reports`.
It revalidates raw samples/build observations and recomputes metrics instead of
trusting stored verdict flags. Explicit historical reports enable consecutive
window and repeated bundle triggers. Different conditioning identities or
concurrency remain separate groups; duplicate evidence is rejected.
Derived decisions retain a hashable `evidence` object containing the exact
validated, content-free source observations and aggregate inputs. Its canonical
SHA-256 equals `evidence_sha256`; it never includes the decision itself.
Automatic capacity attribution requires a measured queued request whose total
exceeds sixty seconds but would fit the budget without its measured queue wait.
Other slow requests remain objective misses without an inferred capacity cause.

## Execution And Remaining Obligations

From `backend`, run `uv run python -m app.operations.pilot_runner --help`.
Supply `--base-url`, `--binding`, `--output` and explicit `--commissioned`.
Tasks are `requests`, `ten_item_bundle`, `fifty_entry_rebuild`, and `probe`.
Repeat `--history-report` to include exact prior request/build artifacts.
`--trigger-file` accepts evidence-bound evaluator observations for separately
measured availability, recovery and real-work outage facts; this does not
execute recovery or invent absent calendar evidence.
Requests/probes use a private `--question-file`; bundle tasks use an approved
`--bundle-file`. Tokens come only from `PILOT_ADMIN_TOKEN` and
`PILOT_MEMBER_1_TOKEN` through `PILOT_MEMBER_8_TOKEN`, never CLI arguments or
reports. The CLI disables redirects and proxy inheritance, requires HTTPS for
controlled-live targets, and creates a new owner-only output file without
overwriting historical evidence.

The `probe` command is one observation, not a four-week scheduler. A commissioned
operator must invoke it every minute, retain the de-identified observations and
declared core schedule, and aggregate after the complete window. Actual
controlled-load, approved real corpus/build/index, provider success, continuous
monitoring and weekly status-disposition evidence must be retained separately.
Restart and clean reconstruction execution remain Ticket 30 obligations.

All local reports retain `live_acceptance=false`; no command independently
activates a Pilot baseline. Deterministic HTTP tests, mocked build transport,
synthetic four-week vectors and historical provider smoke are not real Pilot
capacity, indexing, availability or deployment acceptance. Ticket 29 remains
non-terminal while these required observations or dispositions are missing.
