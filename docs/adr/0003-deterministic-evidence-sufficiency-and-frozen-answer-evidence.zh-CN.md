# ADR 0003：确定性证据充分性与冻结回答证据

状态：已接受

日期：2026-09-06

替代：活跃 Pilot 的 first-three selector、non-empty-context gate 与
candidate-derived citation projection。

## 背景

Ticket 18 建立了已授权、已排序的 Retrieval Candidate Pool，但刻意止步于 answer
sufficiency 之前。历史 runtime path 可能选择前三个 candidate、将 non-empty context
视为 gate，并从更宽的 candidate list 派生 citation。这些 shortcut 无法证明一个
recommendation 覆盖用户的 decisive condition、required branch、source assurance 或
已知 conflict。它们还会允许 frozen answer、citation 与 provider payload 相互偏离。

ADR 0002 仍是构建 authorized pool 所使用 editorial fact 的 authority。本决策只在该
authority-qualified boundary 之后开始，不会创建新的 publication、Candidate preview、
transport、SSE、history、UI 或 provider-activation path。

## 决策

- 活跃 Pilot 的 answer planning 只接受 identity 为
  `retrieval-answer-policy/pilot-v1` 且 scope 为 `published_knowledge` 的 Candidate
  Pool。其他输入都会以 `no_eligible_published_evidence` fail closed。
- sufficiency 是确定性的、assurance-aware 的 product rule。它评估不可变的 Query
  Condition Set、candidate applicability、由问题形状要求的 branch、经过 review 的
  `decision_query` coverage、完整 Claim-Evidence Link support、release assurance、
  material conflict 和 review state。每个 Claim-Linked selected item 都要求其每个
  supporting reviewed claim 的全部 link。count、raw ranking score、non-empty context、
  retrieval order 与 model judgment 都不是 sufficiency input。
- decision 恰好有七个封闭的 insufficient-evidence code：
  `no_eligible_published_evidence`、`decision_not_covered`、
  `decisive_condition_missing`、`material_evidence_conflict`、
  `assurance_support_missing`、`evidence_budget_exceeded` 和
  `knowledge_needs_review`。insufficient decision 只创建一个结构化 reply，不创建
  Answer Evidence Set、recommendation、provider request 或 citation identity。
- sufficient decision 按 canonical item identity 搜索，并选择包含 governing
  recommendation 或 reviewed branches 及每一个 required complement 的最小可行集合。
  governing item 必须通过经过 review 的 `decision_query` 覆盖 normalized question。硬上限
  是三个 item、每个 Evidence Excerpt Snapshot 1200 个字符、总计 3000 个字符。
  authoritative source content length 也必须满足 per-item cap；被截断的 candidate preview
  不能伪装为完整 evidence。无法装入的 required evidence 会以
  `evidence_budget_exceeded` 被拒绝，绝不从看似 sufficient 的 answer 中静默裁掉。
- selected evidence 会在 generation 之前冻结。每个 selected item 绑定 entry、
  editorial revision、Published Knowledge Version、section、content-hashed chunk、
  source content length 与 Evidence Excerpt Snapshot。持久化 item 还包含用于计算
  item identity 的 canonical identity binding；consumer 在信任 snapshot 或 citation 前
  必须重新计算它。Answer Evidence Set identity 绑定 QCS、有序的 selected item identity
  和 governing item。每个 citation identity 绑定该 set identity 及一个 selected item。
- provider-visible prompt 只能从冻结的 Answer Evidence Set 构造。它的
  normalized-question、QCS、evidence-source 和 response-contract region 保持结构化
  隔离。相同的 selected item identity、snapshot、citation identity 与 governing
  citation 同时出现在 frozen set 和 provider payload。provider-visible source record
  携带 `snapshot_id`、`item_identity` 与 `citation_identity`；score、chunk locator 与内部
  selection diagnostic 不会进入 provider-visible 或 user citation data。每个 required
  response section 的 material nonblank line 都必须引用，且不得引入 unknown 或矛盾的 QCS
  assignment、secret value、unsupported quantified assurance，或将 bounded internal case
  universalize。
- 历史 selector、non-empty gate 与 candidate citation path 只能保留在显式 lexical
  heuristic migration/diagnostic profile 中。它们不能影响活跃 Pilot production
  sufficiency、冻结的 Answer Evidence Set、provider evidence payload 或 product
  citation identity。

## 后果

一个 answer 要么有唯一的不可变 evidence basis，要么有一个结构化的、closed-reason 的
insufficiency reply。provider 无法作出 sufficiency choice、添加未选择的 source、伪造
citation，或将 retrieved text 变成 condition 或 policy 的变更。outcome 拥有稳定形状，
可供后续 answer execution persistence 与 surface projection 使用，但这些后续职责不属于
本 ticket。

这项决策使 Pilot 更保守：不完整 branch、未解决 conflict、缺少 assurance 和
over-budget evidence 都会拒绝回答，而不会生成 partial recommendation。这是刻意的，
因为只有在针对声明的 QCS 与 answer shape 完整时，evidence boundary 才是 authoritative。

## 范围

本 ADR 不会 activate provider、定义 HTTP 或 SSE transport、投影 UI state，也不定义新的
publication 或 withdrawal policy。既有 tombstone operation 会脱敏 frozen evidence copy，
使该 record shape 不会泄露 withdrawn excerpt；更广的 history 和 withdrawal contract 仍属于
各自已命名的后续 ticket。
