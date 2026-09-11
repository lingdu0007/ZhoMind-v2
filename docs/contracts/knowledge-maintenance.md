# Knowledge Feedback And Maintenance

Status: Normative

Ticket 27 implements KB-FB-001 through KB-FB-006 and the feedback portions of
KB-UX-007 and KB-SO-003. This contract extends
[canonical authority](canonical-product-contracts.md) and
[Pilot retention](pilot-safety-retention.md); it grants no publication,
provider activation, deployment or roadmap implementation authority.

## Explicit Feedback

Supported and insufficient completed outcomes expose the same four choices:
`helpful`, `insufficient_evidence`, `outdated`, and `out_of_scope`.
An optional description is bounded to 500 characters and requires edit,
preview, cancel and explicit confirmation. Selection, navigation and other
passive behavior create no signal. Helpful confirmation normally remains an
adoption signal, not an automatically created work item.

The retained raw envelope binds the submitting member, exact submission time,
owned persisted answer and Answer Execution, completed outcome, exact published
versions, affected entries or gap context, selected label and confirmed note.
Default capture copies no question, answer, source excerpt or conversation.
An identical repeat for the same owner, answer and scope returns the retained
signal; conflicting submission does not silently overwrite it.

Owner deletion and the default 180-day expiry sever every raw-signal reference.
Deletion and cleanup cannot report success while a signal or reference survives.
An independently verified non-personal decision may remain. No administrator
permission grants access to an underlying Private Conversation Record.
Feedback deletion, expiry cleanup and maintenance-reference writes acquire
the shared retention registry before locking signal rows. A failed write fence
cannot report a successful mutation. Creation rechecks retained-signal authority
after containment verification; consolidation rechecks before committing its
decision. Both use expiry and the current policy's age limit. Expiry during
verification cannot authorize a new item, revision or signal reference.

## Responsibility And Decisions

The System Administrator assigns maintenance responsibility; the named active
member explicitly accepts it. The accountable Maintainer creates, consolidates,
triages, diagnoses and approves findings; a named active Work Owner executes
the assigned work. Current persisted authority governs commands, not client
role flags or a legacy review classification.
Assignment reads and acceptance retries validate the canonical record's
identity, class, state, closed fields and administrator provenance. Accepted
responsibility requires exactly one matching member acceptance event with
the closed event shape. Duplicate or malformed events cannot grant authority
or produce a successful acceptance receipt.

One durable item has exactly one classification, P0-P3 severity, disposition,
affected scope, accountable Maintainer and Work Owner, plus a de-identified
verified pattern and result links. The seven classifications are `confirmation`,
`content-integrity`, `source-freshness`, `coverage-gap`,
`retrieval-answer-behavior`, `product-privacy-operations` and `scope-roadmap`.
`needs-reproduction` is a disposition, never an eighth classification.

`coverage_position` is the curated subject category, not a substitute for
concrete affected scope. At creation the server retains `affected_scope`:
entry reports bind their actual canonical publication and editorial revision;
gap reports retain only the closed insufficiency reason and Query Condition
Set identity, never question text, condition values or a private execution ID.
Consolidation appends deduplicated targets without removing earlier scope.
Raw deletion or expiry cannot erase these non-personal technical targets,
including before reproduction. A qualified roadmap candidate snapshots the
same scope and validates it against the originating maintenance decision.
Both work and roadmap details display this scope after raw removal.

Canonical snapshots and append-only events govern `open`, `triaged`,
`in_progress`, `resolved`, `deferred` and `closed_confirmation`. Commands bind
the expected revision. Unknown fields, malformed or missing authority,
contradictory histories and stale commands fail closed. A terminal state or
result cannot be established by changing a browser label.
Item reads validate the authoritative record class, maintenance identity
kind/value and closed owner fields before owner authorization. A missing or
malformed owner cannot become a server exception, and an unrelated record
class cannot become maintenance authority merely by carrying its schema.
List discovery does not bypass the same strict item projection.

Raw envelopes and optional descriptions remain in the deletable signal store.
They are not copied into immutable item events, findings, roadmap candidates
or audit. An artifact must have both a valid shape and its qualifying parent
decision; a schema name alone is not authority.

## Containment And Remedy

Every P0/P1 item immediately involves the System Administrator and suspends
the smallest independently verified Acceptance Blocking Scope. Evidence must
bind the affected scope and exact acceptance status event, not an unrelated
historical suspension. P0/P1 cannot await cadence or enter roadmap deferral.
Entry-version containment checks every selected retained signal against its
affected entry and the canonical publications bound to its actual execution.
An unbound gap, another entry or another entry's published version cannot
borrow that suspension. A shared-privacy failure cannot establish containment
for an unrelated maintenance classification.

An unresolved P0/P1 item's canonical history is an effective ordinary-answer
block, not merely a label on its acceptance record. Retrieval excludes the
exact affected entry/version; a collection block applies only to this knowledge
base's canonical collection, and a deployment block must match the configured
deployment identity. Foreign collections, unrelated deployment identities and
public-claim-only suspensions cannot establish an answer-scope containment.
Malformed retained blocking authority fails closed. Ordinary completion checks
the same block under the shared maintenance/retention write fence, so a block
committed during generation prevents a successful answer from being persisted
or returned. HTTP and SSE retain the same failed execution; the SSE end marker
does not imply a completed Answer Execution Outcome.

The authorized Work Owner's explicit reproduction or approved-fixture replay
may use a request-local verification context for its own item only. It binds
the current item revision, owner, exact blocking scope and qualified suspended
acceptance event, and rechecks the active Work Owner and administrator. It does
not remove another item's block, weaken source/editorial eligibility, or grant
Provider access. The existing separate Provider invocation authorization
remains required where that route is suspended. The fixture/replay retains a
closed containment-verification receipt bound to its qualifying parent event.
Ordinary chat cannot supply this context. Repairing a source alone does not
reopen ordinary answers; the block remains until the qualified resolution.
Consolidation applies the same containment check to incoming signals before
writing references. The current blocking scope and suspension event must
still match the maintenance decision. Unbound gaps, mismatched publications
and superseded containment cannot be added by reusing an existing item;
rejection leaves its revision and retained signal references unchanged.
Every retrieval-answer or product-privacy-operations item also requires
administrator participation, even at P2/P3.

P2 source freshness may use the existing seven-day grace only while there is
no known contradiction and applicability remains explicit. P3 presentation
work may remain in an owned queue. Neither weakens high-severity containment.
The accepted entry Maintainer explicitly requests freshness review against
the current published version, exact current editorial revision and one of
its declared freshness triggers. Maintenance responsibility alone grants no
editorial authority. Only an eligible published revision can start this
review; the append-only event records its original review time. An identical
retry does not restart the seven-day clock, and changed revision/publication
or undeclared triggers are rejected. Existing source-loss and integrity
exclusions remain authoritative throughout the grace period.
The P2 Work Owner workspace offers explicit review registration for retained
entry targets. Publication, trigger choices and the original review time come
from the member-authorized editorial projection; users do not supply invented
publication identities. Loading or selecting a target creates no review.
Reload and repeated confirmation retain the original review time.
Review reads, retries and current/historical revision eligibility share the
same retained-event qualification. It verifies closed fields, the exact
published revision and declared trigger, accepted Maintainer event and actor,
roles, lifecycle transition and a timezone-aware start equal to the event
timestamp. Duplicate or malformed review events cannot establish a grace
period or expose injected fields. Rewriting only the timestamp payload cannot
extend the review clock.

Remedies link to real authoritative entry revisions, source changes, Boundary
Queries, retrieval or product experiments, product repairs, incidents or
accountable deferrals. Maintenance cannot bypass editorial review or let a
material entry reviser approve that revision. Restoring a high-severity scope
requires its applicable accepted repair/re-acceptance evidence.
High-severity source repair supplies a separate active Delivery Acceptance
Record at the same acceptance stage and deployment, covering the blocked
identity. Its record evidence and the passed check that originally failed
both reference the actual successful repair replay. Activation must follow
both containment and replay; an older acceptance or a new unrelated approval
cannot authorize closure. The resolution retains the exact acceptance status
event as well as the record identity. The Maintainer supplies the administrator's
acceptance identity without gaining administrator-only acceptance access.
Source-change closure requires the independently approved stale-source
fixture, a current successful supported replay and the exact repaired source
identities. It records the qualified source status events, not just mutable
source labels. Missing, unrelated or changed-after-replay source evidence
cannot authorize closure.
Content-revision closure requires an independently approved wrong-content
fixture with its retained integrity-review identity, a current successful
supported replay, and the exact repaired editorial revision and publication
as artifacts. The repair must be a separately approved material revision of
the same entry and remain answer-eligible. Its publication must actually
support the repair replay and follow the defective publication through a
verified same-entry publication lineage; multiple reviewed replacements
before closure are permitted. The publication service's immutable authority
checks remain in force. Missing repair artifacts, an old revision/version
or a new integrity defect after replay cannot establish resolution.
Maintenance records the resulting revision/publication links and does not
itself revise, approve, build or publish the repair.
Coverage-work may close with an entry-revision result when its independently
approved coverage-gap scenario originally expected a supported answer and now
passes against genuinely published eligible knowledge. At least one supporting
publication must be new to the fixture's publication snapshot. Closure binds
the exact supporting publications and editorial revisions and verifies their
current publication authority and answer eligibility. Unrelated publication
artifacts or a still-insufficient answer cannot close a coverage repair. This
path does not relax the separate wrong-content repair lineage requirements.

Every closure must cover the item's complete retained affected scope, including
targets added by consolidation. For repair resolution, submitted approved fixtures and successful
replay targets must match. The current replay is selected independently for
each fixture by maintenance event revision; a later failed result disqualifies
an earlier pass for that fixture. Selection qualifies each preceding replay's
event, actor, record, digest and parent authority before inspecting its fixture;
a missing or damaged latest result cannot silently revive an older pass.
Multiple fixtures can jointly close complete
scope, but a partial repair cannot close the entire item. The workspace displays
the approved scenarios and latest results, allows explicit selection of closing
evidence, and submits all selected fixture/replay and remedy artifacts together.
Direct P2/P3 confirmation instead requires independently approved confirmation
Findings whose fixtures jointly cover the entire retained scope. Each fixture
must have expected and actually observed evidence-gated answers; every entry
target must occur in its actual evidence publications, which must remain
currently published and eligible at closure. No repair replay is invented for
this path. The explicit `confirmation_closed` event retains the exact Finding
result links. An ordinary transition cannot directly close unresolved work.
Historical reads requalify the retained Findings, fixtures, scope and links,
without retroactively invalidating a legitimate closure for later source loss.
The workspace exposes direct closure only for a complete approved confirmation
scope; loading it does not close the item.
This qualified confirmation may close from either `triaged` or `in_progress`;
starting its assigned verification work does not remove the valid closing path.
For content work, the workspace offers explicit repair
confirmation using the selected approved scenario and its successful replay.
It submits the replay's exact revision/publication identities without asking
the user to invent artifact identifiers. Historical reproduction versions are
labelled as such, separately from the repair replay's verified publication.
Loading the workspace does not resolve work; closed confirmation survives
reload with the recorded result links.
For P0/P1 content, retrieval and product repairs, the successful approved
replay exposes a separate re-acceptance input to the accountable Maintainer.
The closure command includes that identity alongside the exact repair
artifacts; it never substitutes the containment record or Provider invocation
admission. Empty input cannot enable closure. An old, suspended or unrelated
record is rejected by the existing scope, timing and replay-evidence checks,
without changing work state. Loading or filling the field submits nothing.
Successful resolution retains the qualified acceptance event and survives
closed-confirmation reload. This does not publish, deploy or activate a route.
Provider recovery evidence binds the route actually used by the authenticated
execution, its activation event and its verified acceptance digest. Resolution
requires the approved failure fixture, successful current replay and matching
route/acceptance artifacts. For ordinary-severity work, reactivation invalidates earlier repair evidence;
saving an inactive draft does not. No provider credential or endpoint is copied
into the maintenance verification context.
Suspending the acceptance bound to the active route also closes ordinary
maintenance reproduction's access to that Provider. A generation-unavailable
result caused by this closed gate is not an independently reproduced Provider
defect and cannot create a verified fixture or Finding. A connection-only
validation is not a substitute for a frozen-evidence repair scenario.
The current System Administrator may sign a bounded verification authorization
for the item's active ordinary-member Work Owner. Signing binds the exact
P0/P1 product/privacy item revision, retained Query Condition Set, current
publications, current containment event, and a separate active route admission
record validated through the existing complete Provider approval checks.
Identical authorization content returns the same qualified immutable record;
malformed retained records cannot be replaced by retrying. Reads retain the
item's current member authorization boundary. Signing alone makes no Provider
call, activates no route, and supplies neither a successful repair replay nor
post-repair re-acceptance evidence.
An explicit Work Owner reproduction or approved-fixture replay may consume
that authorization only for its bound non-personal Provider-failure scenario.
Admission and finalization recheck current ownership, item revision, retained
conditions, publications, containment and the complete route admission digest.
The request-local router uses the ordinary retrieval, frozen-evidence,
generation, citation and answer-persistence pipeline without replacing the
ordinary active pointer. Verification context records the authorization
identity and digest instead of inventing an activation event. Retained fixture
and replay reads qualify that authorization and bind its exact work revision,
actor and scenario to their recording event. A changed work revision requires
fresh authorization. Successful isolated replay is repair evidence, not route
activation or high-severity re-acceptance; it cannot by itself authorize closure.
Initial Provider reproduction authority requires a retained signal. Once the
independent Provider-failure fixture has an approved Finding, a repair grant
instead records that exact `approved_fixture_identity`. Signing and execution
qualify the Finding, fixture scope and exact scenario, while still requiring
the Work Owner's current owned input. The grant can replay only that fixture,
not authorize a new reproduction. Raw deletion or expiry therefore neither
revokes legitimate repair evidence nor recreates raw references. An unapproved
fixture cannot replace missing feedback, and current containment, ownership,
publication and route-admission checks remain mandatory.
P0/P1 Provider closure requires one separate active repair acceptance at the
original stage and deployment, covering the contained scope. Both its record
evidence and the passed check that originally failed must reference the
successful isolated replay, and acceptance must follow containment and replay.
The current active route must be the route actually verified, explicitly
activated after replay by the item's participating administrator and bound to
that repair acceptance, not the isolated-invocation admission record. The
original invocation admission must also remain qualified. Closure retains the
exact repair acceptance status event and route activation event. An inactive
route, unrelated activation, or a replay reference at only one acceptance
level cannot establish resolution. Normal route activation remains the
existing explicit administrator operation; maintenance never performs it.
The high-severity Provider workspace separates administrator signing from
Work Owner execution. Signing requires explicit confirmation of the displayed
route, invocation admission and Query Condition Set; loading or editing these
fields signs nothing. The Work Owner explicitly loads the retained grant and
uses it only when its item, revision, owner and query conditions match the
selected reproduction or approved replay. Reload clears the local selection;
an older work-revision grant cannot enable replay. Each successful signing
clears its confirmation checkbox. Maintainer closure submits the separately
entered repair acceptance rather than substituting the invocation admission.
Isolated replay labels its retained admission as invocation admission and
shows the verification authorization separately, not as ordinary route acceptance.
Server rejection remains visible and does not change the displayed work to
resolved; successful closed confirmation reloads with the retained result links.
Product-workflow repair requires an independently approved failed-execution
scenario with a current eligible supported reference. Its closure artifacts
are that exact fixture and its current successful supported repair replay,
carrying exactly the reference publications. A completed Provider failure
cannot stand in for the failed product execution; an unrepaired replay or
another maintenance classification cannot authorize product-repair closure.
This bounded verification does not publish code, deploy changes or replace
the applicable high-severity re-acceptance requirement.

## Independent Diagnosis

The Work Owner independently reproduces or verifies a non-personal scenario.
A synthetic fixture is deliberately shared material, not an automatically
copied private conversation. Explicit confirmation must be a boolean `true`,
not a coerced numeric or textual substitute.

The retained signal carries a server-derived digest of its structured condition
records, separately from the original full Query Condition Set identity.
Independent reproduction matches that condition digest, not the original
question-bearing identity. Changing the question wording therefore does not
require copying the original question, while changing declared conditions
cannot borrow the report. Missing condition provenance fails closed.
The independent scenario's full Query Condition Set remains exact across its
authenticated reproduction and repair replay.

Immutable fixtures retain request digests and verified categorical facts, never
`replay_request`, question text, condition values, or private answer identities.
Each fixture also retains the selected signal's closed non-personal affected
scope, not the raw signal identity or envelope. Registration rechecks that
scope after execution. Source/content targets must match the selected entry;
registration and every fixture read qualify the retained scope against the
maintenance item and its publication review. Another entry's verified defect
cannot stand in for the reported entry, including after raw deletion.
The Work Owner explicitly selects an owned, retained execution for each replay;
its request digest and full Query Condition Set must match the approved fixture
before execution. A missing, foreign, deleted or different input is rejected.
The input remains under ordinary private-conversation retention. Reauthoring
the same independent scenario can supply a new matching input without reviving
deleted feedback or copying the original report. The workspace clears replay
input selection on refresh and scenario change; selecting an input runs nothing.
Initial Provider signing verifies the requested full Query Condition Set
against the Work Owner's own retained execution and matches only its structured
condition digest to the report. Repair signing binds the approved independent
fixture instead. Neither path exposes the question or private execution
identity to the administrator or durable authorization record.

Diagnosis checks the fixture's exact Query Condition Set and current Published
Knowledge Versions. A reproducible authenticated fixture binds its actual
retrieval profile, expected outcome, observed outcome or separate execution
state, and verification result. Assigning today's profile to a historical
execution is not proof of its execution profile.
Current publication identity comes from authoritative publication pointers,
not Knowledge Map visibility or the newest editorial draft. Source diagnosis
reads the sources of that exact published revision and binds their current
qualified status-event identities. A later draft does not silently replace
the publication under diagnosis. Source changes invalidate earlier diagnostic
or repair evidence even when the displayed availability value changes back.
The source snapshot also binds the complete retained event-identity set.
Ambiguous same-time source status events cannot establish current maintenance
authority through a random-identity tie-breaker.
Historical supported references must also retain current evidence eligibility
for their exact publication and source events. An unavailable source cannot
be relabelled as a retrieval miss merely because its version remains published.

Missing eligible knowledge routes to coverage work. Wrong or stale knowledge
routes to content or source work. Retrieval, condition-loss, citation-drift,
provider and product diagnoses require the corresponding independently
verified evidence; insufficiency alone never proves a retrieval defect.
An existing publication alone does not prove wrong content. The accepted
entry Maintainer explicitly confirms an independently reviewed contradiction
or integrity defect against the exact current published revision and one of
its declared sources. The command retains a closed categorical event, not a
question, answer, source excerpt or feedback description. Maintenance
responsibility alone cannot register that editorial review.
The event excludes that revision from new answers immediately, including
during a freshness grace period, without changing the publication pointer.
Matching retries retain the original event. Current and historical eligibility
validate its publication, source, actor, accepted responsibility and event
shape. Content diagnosis binds that qualified event identity in its publication
review; the Work Owner still independently registers the authenticated scenario
and the Maintainer separately approves its Finding. This review neither
publishes a repair nor replaces material-revision approval.
The content-maintenance workspace reads permitted revision/source choices
and the retained review identity from the editorial projection. Loading,
choosing a defect and reloading submit nothing. A separate unchecked
independent-review confirmation is required for every explicit submission,
including a retry. A retained review fixes its original source and defect.
Citation-drift diagnosis requires a current eligible supported reference,
frozen evidence and an authenticated failed generation whose attempts were
rejected by the existing citation validator. A provider timeout is not proof
of citation drift. The retained fixture stores the verified categorical fact,
not the rejected generated text. Citation repair can close a bounded retrieval
experiment only after the same declared scenario passes with the exact
reference publication identities; an unrepaired replay cannot close it.
Retrieval-miss diagnosis requires a current eligible supported reference,
an actual insufficient-evidence result with no frozen evidence and the
`no_eligible_published_evidence` reason. A generation failure after evidence
selection does not prove a retrieval miss. The approved retrieval-miss
scenario uses the same bounded experiment closure: a current successful
supported replay must carry exactly the reference publication identities.
Condition-loss verification recognizes the execution contract's actual
pre-provider comparison: a structurally valid generation input differs only
in its frozen Query Condition Set. That execution remains `failed`, with no
completed outcome and no claim that the Provider received the changed input.
A timeout or another malformed/changed input is not condition-loss proof.
Maintenance retains only the categorical verified difference and requires a
current eligible supported reference. Repair uses the same declared scenario
and exact-reference-publication experiment closure.
The same declared fixture can be replayed after a remedy. Expectations are not
changed after observing a result to manufacture a successful repair.

A repair replay uses an approved independent fixture and records a separate
immutable result. Fixture registration and replay events bind the complete
artifact digest; a schema label or internally consistent rewritten result
does not establish verification. Resolution requires a current successful
replay as well as its qualified closure artifact. A failed, stale or
unqualified replay cannot authorize closure.
Later reproduction does not revoke a previously approved fixture or make
its historical replay unreadable. The approved Finding, not the latest-fixture
pointer, authorizes replay of the exact registered scenario. Duplicate evidence
cannot replace the original approved expectation or authorize a second fixture.
The workspace separates current diagnosis from approved repair scenarios;
resolution controls bind the selected scenario to its own successful replay.

The Maintainer approves an independently verified Finding, not their own
unverified assertion. Findings exclude the original question, answer,
conversation, submitting identity, exact submission time and optional note.
Their normalized pattern, verification method/result, affected scope and
bounded action links survive removal of the originating raw signal.
Approval binds the complete Finding digest in the maintenance history.
Reading, replay authorization and roadmap qualification validate the same
closed Finding shape, approved digest and independently registered fixture;
rewritten fields cannot become retained evidence or leak through projection.
Finding identity binds the concrete fixture scope and publication/reference
reviews as well as its execution verification fingerprint. Verifying another
target cannot return the first target's Finding, while a duplicate of the same
target retains its original approval. The execution fingerprint used to count
roadmap independence is unchanged: distinct targets alone do not manufacture
independent verification samples.

## Cadence And Roadmap

Ordinary feedback is triaged within seven days. Weekly review covers open
work and roadmap qualification; monthly review covers knowledge health and
deferrals; quarterly review records sampled acceptance. Review evidence binds
current work revisions and responsible actors. One Maintainer's review does
not silently satisfy another's responsibility. Overdue work remains visible.

Review submission binds the displayed context digest. The retained snapshot
includes the accountable open-work revisions, current deferral revisions and
knowledge health. Published revision identity remains distinct from the current
editorial revision and its eligibility. Quarterly samples bind each active
acceptance record's exact `checks_verified` event, accepted scope, verification
time and actor, and complete per-check results and verification evidence.
The verification must fall in the current UTC calendar quarter and cannot be
future-dated. A historical Active record without current-quarter verification
is not a new quarterly sample. Existing Delivery Acceptance verification is
the authority; maintenance does not manufacture a passing run from a timestamp
or duplicate its execution workflow. Carried-forward checks retain their
distinct result and evidence instead of being relabeled as newly passed.
A changed context
requires a fresh review; duplicate or malformed recording events cannot
establish cadence completion. A signal triaged by another accountable
Maintainer is not counted as still untriaged.

Roadmap qualification requires three distinct Answer Executions with outcomes
within a rolling 30-day window, or two independent Validated Findings.
Different executions may have the same completed outcome kind. A member is
never an aggregation key; duplicate signals or a changed expected result do
not create independent evidence. Explicit `scope-roadmap` may instead create
an accountable direct deferral. P0/P1 cannot use any qualification path.

Candidates retain only the non-personal pattern, affected scope, desired
outcome, bounded-work limitation, current scope boundary, owner, review date
and maintenance-decision links. Monthly owner review either starts a separate
Wayfinder map, renews with owner/rationale/date, or closes the candidate.
Creating a map does not commission its implementation. A successful ownership
transfer returns its command receipt; subsequent reads use the new authority.

Candidate discovery follows current ownership or the originating maintenance
responsibility. Qualification binds the initial candidate digest, and monthly
history validates the actor, consecutive revision and legal state transition.
A map must match the exact identity and digest created by the qualifying
monthly decision; a merely schema-labelled or replaced artifact is rejected.
