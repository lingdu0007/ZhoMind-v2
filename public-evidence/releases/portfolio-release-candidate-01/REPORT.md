# Portfolio Release Candidate Evidence

## Provenance

This accepted bundle records source_revision `91753f1c1ff6fc07bc262dfa50fb719a63210e0b`. All referenced runs use that candidate revision and publish only bounded, non-sensitive results.

## Retrieval and Answer

The controlled Evaluation Retriever reran `sparse_bm25`, `dense`, and `hybrid_rrf` over the frozen Project-Derived Corpus and Evaluation Query Set. The bundle includes corpus/query integrity metadata, normalized annotations, sanitized candidate traces, and retrieval run conditions. Authenticated answer checks passed for `evidence_gated_answer`, `insufficient_evidence_reply`, and `generation_unavailable`; normal, SSE, and history citation contracts passed where applicable.

## Security

The live Prompt Injection run passed all four fixed case kinds. Public records contain only normalized case identity, outcome, citation counts, and failure classification.

## Performance

| load | TTFT P50 | TTFT P95 | TTFT P99 | total P50 | total P95 | total P99 | error_rate | retrieval_ms | provider_ms | embedding_provider_ms | persistence_ms | application_controlled_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c1 | 7651.026 | 15928.331 | 23625.013 | 7654.209 | 15929.913 | 23627.261 | 0.000000 | 4152.850 | 4727.400 | 4009.840 | 6.850 | 249.737 |
| c5 | 12463.722 | 74460.959 | 77341.271 | 12467.550 | 74464.770 | 77343.061 | 0.000000 | 18279.400 | 4905.550 | 17457.305 | 8.650 | 1056.362 |

The observed `application_controlled_ms` envelope uses 40 successful samples: center 653.049 ms, spread 1156.406 ms, tolerance 3470.000 ms, and threshold 6128.299 ms. It is derived from observed variance, not the twelve-second target.

## Production Acceptance

Authenticated normal answer, SSE, history, insufficient evidence, generation unavailable with zero generation fallback hops, withdrawal, deactivation, service health, two independent clean-start smokes, and persistent-stack Administrator and Knowledge User role boundaries passed. The controlled provider route and all isolated resources were restored or removed after observation.

## Limits

The twelve-second end-to-end P95 target was not met: c1 total P95 was 15.930 seconds and c5 total P95 was 74.465 seconds. External Provider latency is reported separately from project-controlled time. Results apply only to the recorded corpus, query set, model identity, load, host envelope, and observation window; they do not establish universal performance or retrieval superiority.
