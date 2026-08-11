# 示例 Public Evidence Bundle

## Provenance

示例 bundle 记录 source_revision、一个 run identity、project-derived-corpus 和 evaluation-query-set。

## Retrieval Metrics

可回答查询聚合报告 sparse_bm25、dense、hybrid_rrf 和 migration 模式的 evidence_recall@3、context_precision@5、first_gold_rank 与 retrieval_duration_ms。

## Answer Outcomes

确定性回答 cases 记录 evidence_gated_answer、insufficient_evidence_reply、non_knowledge_base_reply 和 generation_unavailable 四种结果。

## Prompt Injection

对抗 cases 对 instruction_override、secret_extraction、forged_source 和 unsupported_answer_pressure 进行分类。

## Performance

报告 TTFT 和 total P50、P95、P99 百分位，以及 error_rate 和各阶段时长。

## Limits

证据不推广到记录的 run conditions 之外；十二秒 P95 仍然是 target。
