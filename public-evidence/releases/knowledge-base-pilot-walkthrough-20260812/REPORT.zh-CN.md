# Production Interviewer Walkthrough 证据

## 溯源

此 bundle 记录 source_revision `42bc93989ed39ea25d811a9e51f45f0950860d57` 上的 production-compose run `knowledge-walkthrough-20260812T051218600Z`。该 run 使用 generation model `deepseek-ai/DeepSeek-V4-Flash` 与 embedding model `Qwen/Qwen3-Embedding-8B`。

## 方法

已认证产品路径执行真实 Candidate Build 与显式发布，通过 Team Invitation 接纳 Knowledge User，打开 Knowledge Map，并运行 direct、paraphrased 与 Boundary query。normal、SSE 与 history projection 通过持久化 answer identity 和相同 Evidence Summary snapshot 验证。该 run 还打开 Public Source Citation、提交结构化 feedback，并验证 deactivation。

## 结果

`walkthrough.json` 中 12 个归一化 check 全部通过。direct 与 paraphrased 路径返回 `evidence_gated_answer`；不支持的 Boundary query 返回 `insufficient_evidence_reply`。production migration、dense indexing、generation contract、feedback persistence 与 access revocation path 在同一 observation window 内启用。

## 限制

- 这是对一条已复核 Pilot entry 的一次有界 production-compose run，不是完整 corpus 的 live acceptance。
- 独立 model generation 按 Evidence Summary 比较；每个 normal 或 SSE answer 只与其自身持久化 history projection 做逐字节比较。
- Question、answer、excerpt、user identity、credential、host 与 private corpus text 均被排除。
