# Workload-Bound Pilot Service Objectives

Status: accepted measurement contract; live Pilot acceptance remains evidence-gated.

Date: 2026-09-12

The Pilot replaces the historical workspace ADR 0014
`bound-the-first-single-server-release` fixed-host service commitment and
12-second P95 with the measured Pilot Workload Profile and KB-SL-001 through
KB-SL-004. Eight admitted test members, two executions, two queue positions,
one execution plus one queue position per member, and one build worker define
the measured load. Fifty active entries and at most 10,000 answer-eligible
chunks are measured headroom, not permanent publication ceilings.

We keep the historical portfolio c1/c5 performance artifacts unchanged.
Their first-body-byte measurement is not actual answer-content TTFT, their
12-second target is not the current Pilot objective, and their corpus, host,
provider, configuration and window cannot grant current acceptance.
The existing 25-member/500-source safety guards and 25 MiB upload guard remain
compatibility resource controls until separately reconfigured and verified;
they are not a supported Pilot capacity claim. This ADR does not raise them,
reduce publication authority, change provider approval, or replace ADR 0009's
single-process scheduling rule.

The accepted measurements distinguish acknowledgement (1-second P95), meaningful
processing (2-second P95), actual answer content, and closed outcome (30-second
P95 at concurrency two; every admitted burst closes within 60 seconds including
queue and every route attempt). Fewer than 20 samples cannot establish a
percentile pass or failure. Provider reliability and four-week core-window
availability are separate: Generation Unavailable is product-available, not
route-successful. Missing observations cannot become measured uptime.

Performance misses require evidence-linked, owned, time-bounded decisions,
not withdrawal of otherwise valid knowledge. The
[Pilot Measurement Contract](../contracts/pilot-measurement.md) defines the
current interfaces, report boundary and remaining deployment obligations.
