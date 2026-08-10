# Portfolio Release Candidate Evidence

## Provenance

本次已验收 bundle 记录 source_revision `91753f1c1ff6fc07bc262dfa50fb719a63210e0b`。所有引用的 run 均使用该 candidate revision，且仅发布有界、非敏感结果。

## Retrieval and Answer

受控 Evaluation Retriever 在冻结的 Project-Derived Corpus 和 Evaluation Query Set 上重新运行 `sparse_bm25`、`dense` 与 `hybrid_rrf`。Bundle 包含 corpus/query 完整性元数据、归一化 annotations、已净化的 candidate traces 与 retrieval run conditions。`evidence_gated_answer`、`insufficient_evidence_reply` 与 `generation_unavailable` 的 authenticated answer checks 均通过；适用场景下 normal、SSE 与 history citation contracts 均通过。

## Security

live Prompt Injection run 的四类固定 case 全部通过。公开记录仅包含归一化 case identity、outcome、citation counts 与 failure classification。

## Performance

| load | TTFT P50 | TTFT P95 | TTFT P99 | total P50 | total P95 | total P99 | error_rate | retrieval_ms | provider_ms | embedding_provider_ms | persistence_ms | application_controlled_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c1 | 7651.026 | 15928.331 | 23625.013 | 7654.209 | 15929.913 | 23627.261 | 0.000000 | 4152.850 | 4727.400 | 4009.840 | 6.850 | 249.737 |
| c5 | 12463.722 | 74460.959 | 77341.271 | 12467.550 | 74464.770 | 77343.061 | 0.000000 | 18279.400 | 4905.550 | 17457.305 | 8.650 | 1056.362 |

观测到的 `application_controlled_ms` envelope 使用 40 个成功样本：center 653.049 ms、spread 1156.406 ms、tolerance 3470.000 ms、threshold 6128.299 ms。该 envelope 来自实际方差，而不是由 12 秒目标反推。

## Production Acceptance

authenticated normal answer、SSE、history、insufficient evidence、generation unavailable with zero generation fallback hops、withdrawal、deactivation、service health、两次独立 clean-start smoke，以及 persistent stack 上的 Administrator 与 Knowledge User role boundaries 均通过。观测结束后，受控 Provider route 已恢复，所有隔离资源均已删除。

## Limits

12 秒端到端 P95 目标未达成：c1 total P95 为 15.930 秒，c5 total P95 为 74.465 秒。External Provider latency 与 project-controlled time 分开报告。结果仅适用于已记录的 corpus、query set、model identity、load、host envelope 与 observation window；不构成通用性能或检索优越性结论。
