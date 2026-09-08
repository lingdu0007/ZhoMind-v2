# Ticket 24 Candidate Publication Acceptance

Status: all fifteen Ticket 24 criteria verified; scoped implementation complete
Date: 2026-09-08
Implementation revision: `59064b77ff8dd6adef09823604c15799c4e1bf94`
Fixed baseline: `4c8a5437ec1bded09df89435dd13b28cc7925e0d`
Branch: `ticket24-candidate-publication`

## Scope

This record covers all fifteen criteria of the workspace ticket
`.scratch/usable-team-knowledge-base/issues/24-inspect-accept-publish-candidates.md`:
KB-RL-004, KB-RL-005 and the Candidate portions of KB-AC-004.
It includes the prior implementation at `4bcad79` and its continuation at
`b189f85` and `59064b7`. It does not claim a Pilot Entry Baseline, live-provider
activation, production deployment, withdrawal implementation, or Portfolio
Release acceptance.

Editorial approval, Candidate inspection, authenticated Candidate acceptance
and explicit publication remain separate actions. Candidate acceptance
executes the closed deterministic evidence contract without calling a
generation provider. The disposable browser API separately verifies a
published answer through the authenticated chat path with a deterministic
test provider.

## Acceptance Mapping

The principal HTTP regression file is
`backend/tests/integration/test_ticket24_candidate_publication_api.py`.
The browser journey is `frontend/tests/reviewed-bundles-workspace.test.mjs`.
The table gives representative tests; the full suites contain additional
negative cases.

| Criterion | Retained verification |
| --- | --- |
| Authorized metadata, chunks, bundle/revision/hash/generation/configuration visibility | `test_inspection_is_durable_and_exactly_bound`; role-denial matrix; real browser metadata inspection |
| Focused replacement comparison | `test_replacement_failure_is_isolated_and_preserves_the_existing_pointer`; `test_multi_source_runtime_projection_does_not_invent_replacement_diff_changes` |
| Durable exact inspection | `test_publication_rejects_corrupted_retained_inspection_or_acceptance`, including foreign identities and JSON boolean/numeric substitutions |
| Supported and Boundary queries against the exact Candidate | `test_candidate_acceptance_requires_exact_supported_and_boundary_records`; real Release-Assured chain described below |
| Governing entry/section, Evidence Set, citations and snapshots retained | Supported-result assertions, full deterministic result verification, multi-source Claim-Linked tests and reloadable browser acceptance |
| Structured Boundary insufficiency with no provider or citations | Boundary-result assertions and corruption cases for its QCS identity and provider count |
| Diagnostics, old inputs and changed configuration cannot substitute | Frozen input/chunk metadata checks; configuration-change test; stale-generation and mismatched selection tests |
| Selection requires approval, inspection, acceptance and exact current identities | `test_publication_requires_exact_eligible_confirmation_and_creates_a_version`; retained-record corruption matrix |
| Confirmation names every Candidate and create/replacement effect | HTTP exact-selection assertions and browser confirmation journey |
| Separate published/failed/skipped, with no false complete-batch success | Mixed-batch HTTP assertions and browser partial-publication message |
| Atomic publication per selected entry | Injected pre-pointer persistence failure, independent successful sibling and persisted publication reload |
| Prior eligible version stays live until successful replacement | Failed-replacement test and unpublished-successor retrieval tests, including a real successor with an unverified new source |
| Failure and duplicate confirmation preserve one pointer and permit retry | New-confirmation retry succeeds; duplicate confirmation replays retained results; leased execution and interrupted-result recovery tests |
| Candidate and unselected content excluded from ordinary retrieval | Ordinary-pool publication tests, pointer corruption tests and browser role boundary |
| Exact identities retained for acceptance and reconstruction | `test_delivery_acceptance_persists_an_exact_candidate_publication_binding`; immutable publication, inspection and acceptance reloads |

`backend/tests/unit/test_editorial_authority.py` also exercises a real
Release-Assured editorial approval/export, bundle intake, Candidate build,
inspection, acceptance and publication. It proves that snapshot bytes and
export hashes are unchanged and that an unpublished successor's unverified
new source does not revoke the still-valid pointed version.

## Verification

Commands are run from `backend/`, `frontend/`, or the repository root as
appropriate.

| Command or check | Result |
| --- | --- |
| `uv run ruff check .` | Passed |
| `uv run pyright` | 0 errors, 0 warnings |
| `uv run pytest -q -ra --tb=short --disable-warnings` at the implementation revision | 833 passed, 1 skipped |
| `npm test` final implementation rerun | Passed: 45 unit tests and 123 browser/gate tests (108 base, 9 reviewed-bundles, 6 settings); production builds passed |
| `uv run alembic heads` | Single head: `20260908_t24_candidate_pub`; migration regressions included in the backend suite |
| `python3 scripts/check-docs-parity.py` | Passed, including this bilingual evidence record |
| `python3 scripts/validate-evidence-bundle.py --all` | All seven retained bundles passed |
| `python3 scripts/scan-secrets.py` | Passed |
| Playwright at 1440, 1024 and 390 pixels | Nonblank screenshots; document width equals viewport width; mobile preserves the declared desktop-only administrator boundary |

The first backend full run, concurrent with browser work, reported 822 passed,
one failure in `test_long_running_indexing_renews_the_candidate_worker_lease`,
and one skip. That test uses a 50 ms lease. Its isolated rerun and the final
complete backend run passed. No lease, heartbeat or production timing behavior
was weakened to obtain the final result. Existing dependency deprecation
warnings and the frontend bundle-size warning remain visible.

## Standards

The prescribed `standards_reviewer` reviewed the complete baseline diff and
then the fixes through `59064b7`. Three documented violations and one possible
Duplicated Code smell were resolved:

- Historical decisive source loss is no longer cut off by a successor's approval.
- Ordinary retrieval validates the pointer's entry, document and generation.
- Retained inspection/configuration checks distinguish JSON booleans from numbers.
- Current and historical authority share source validation and section projection.

Final Standards re-review: no remaining actionable findings.

## Spec

The prescribed `spec_reviewer` reviewed the same complete diff and the fixes.
Both identified P1 findings were resolved:

- Valid Release-Assured exports accept their retained canonical event UUID
  references without rewriting frozen snapshots.
- An unpublished successor's own source problem does not remove the eligible
  predecessor; the predecessor's own source loss still fails closed.

Final Spec re-review: no remaining actionable findings or scope regression.

## Limits And Known Failure

- The live Milvus test is explicitly skipped without `RUN_MILVUS_E2E=1`.
  No live embedding, generation provider or persistent production stack was
  exercised by this task.
- The historical command
  `python3 scripts/verify-portfolio-release.py --expected-source-revision 91753f1c1ff6fc07bc262dfa50fb719a63210e0b --expected-release-revision HEAD`
  remains **failed**: the Public Evidence Bundle source is not an ancestor of
  the release artifact revision. The ancestry check also fails at the fixed
  baseline, before Ticket 24. No evidence was rewritten and no Portfolio
  Release pass or waiver is claimed here.
- The local preview uses a separate temporary SQLite database and synthetic
  editorial material, including one intentional item failure. It is not
  production acceptance or a permanent service.
