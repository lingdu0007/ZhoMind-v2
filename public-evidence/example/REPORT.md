# Example Public Evidence Bundle

## Provenance

The example bundle records source_revision, one run identity, the project-derived-corpus, and the evaluation-query-set.

## Retrieval Metrics

Answerable-query aggregates report evidence_recall@3, context_precision@5, first_gold_rank, and retrieval_duration_ms for sparse_bm25, dense, hybrid_rrf, and migration modes.

## Answer Outcomes

Deterministic answer cases record the evidence_gated_answer, insufficient_evidence_reply, non_knowledge_base_reply, and generation_unavailable outcomes.

## Prompt Injection

Adversarial cases classify instruction_override, secret_extraction, forged_source, and unsupported_answer_pressure.

## Performance

TTFT and total P50, P95, and P99 percentiles are reported with error_rate and stage durations.

## Limits

The evidence does not generalize beyond the recorded run conditions; the twelve-second P95 remains a target.
