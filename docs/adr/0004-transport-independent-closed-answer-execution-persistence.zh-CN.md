# ADR 0004：传输无关的封闭回答执行持久化

状态：已接受

日期：2026-09-06

替代：将 `ChatMessage.rag_trace` 或可变 chat message 视为 answer authority 的
transport-specific answer gate、generation/persistence 语义、snapshot slicing、
outcome inference 与 source-count heuristic。

## 背景

ADR 0003 使 evidence sufficiency 成为确定性规则，并在 provider 可以 generation 之前
冻结 Answer Evidence Set。但仅此并不能阻止 normal HTTP handler、SSE handler、
persistence path、reload path 或 history projection 各自作出独立决定。可变 message
body、legacy trace flag、source count 或后续 retrieval 都可能让同一个 user request 在
不同 surface 产生不一致的 outcome、在 interruption 后伪造 support，或重新切分历史
snapshot。

产品需要为每个 admitted question 保留一份私有 result。它必须在 admission 前让用户的
decisive condition 显式且可编辑，但在每个 completed turn 中冻结。它必须区分 execution
lifecycle 与四种 completed answer outcome；当无法证明 retained result 完整且内部一致时，
它必须以 application failure 失败。

ADR 0003 仍是确定性 sufficiency、冻结的 Answer Evidence Set、provider-visible evidence
以及 citation identity 的 authority。本 ADR 只负责 execution admission、terminal
persistence，以及对那个已经 closed 的 evidence decision 进行 cross-surface projection。

## 决策

- 每个已认证的 chat request 都创建一个私有 Answer Execution，带有不可变的
  `answer_execution_request/v1` header 和只追加的 `answer_execution_event/v1`
  record。它随用户的 private conversation 保留，并且只能由相同的、经过验证的
  conversation deletion 或 expiry path 删除。它不是全局 canonical-record aggregate。
  每个 execution 的 event sequence 唯一；terminal、redaction 或 stream-delivery lifecycle
  writer 都会先锁定该 execution，再读取并追加下一条 event。conversation deletion 与 retention
  purge 会在删除 event trail 或 header 前取得同一把 execution lock，因此 private retention
  不会留下孤立的 execution event。在 SQLite 上，admission、event writer、deletion 与 expiry
  会在各自相关 read 前共享一个 transaction-wide writer fence：fresh transaction 通过 `BEGIN
  IMMEDIATE` 开始它；已打开的 deferred read transaction 则通过一个 no-op `chat_sessions`
  write 升级到同一 fence。cleanup 会先锁定匹配的 execution header、再锁定经过验证的
  conversation session，然后在 delete 前重新扫描 header；admission 会在同一个 fence 之后锁定
  该经过验证的 session。因此 admission 或 event write 不会与 private deletion 或 expiry
  竞态，从而留下 orphaned header、event 或 message。admission 会在 execution 开始前，将 request
  header 耐久地绑定到其精确的 user-message identity。
- execution state path 为 `admitted -> queued -> running`，之后恰好进入一个 terminal
  state：`completed`、`stopped`、`failed`、`throttled` 或 `rejected`。`completed`
  不是 answer outcome。它恰好拥有一个 outcome：Evidence-Gated Answer、Insufficient
  Evidence Reply、Non-Knowledge-Base Reply 或 Generation Unavailable。其他 terminal
  state 没有 completed outcome，不能暴露伪造的 answer、evidence、citation 或 provider
  result。
- admission 会连同 normalized question 保存显式 QCS。condition 可以显式提供、为每个
  新 turn 编辑，或仅从同一 private conversation、同一 owner 最近的 completed execution
  继承。继承的 condition 会复制到绑定新 normalized question 的新 QCS 中，并保留
  source-execution provenance。即使 normalized question 相同，继承的 QCS 也会拥有新的
  per-turn identity。新的 conversation 从没有继承 condition、且 composer draft 已清空的
  状态开始。若 retry 的旧 turn 已有保留的 execution，包括 non-completed 的 SSE 或 history
  projection，它必须显式提交该 turn 冻结的 QCS。缺失 client-side terminal data 绝不能授权
  重新 inheritance：带有 inherited 标记、但没有 retained execution 的 retry 会在本地失败，
  直到该 execution 被恢复。只有 admission 前的 transport failure 才能重复原始请求的 QCS 或
  inheritance request。只有在 streamed execution 绑定到 submitted turn 后才可接受它：
  implicit QCS 必须精确等于从 submitted question 派生的 condition，inherited QCS 必须指向
  同一 private conversation 中已投影的 completed source execution，并复制其精确 condition。
  completed execution 只有在完整 terminal projection 校验成功后才取得 retry authority；
  矛盾的 terminal projection 会清除该本地 authority，直到 persisted execution 被恢复。
  hidden memory 与 global profile 都不能提供 decisive condition。
- 狭窄的 non-knowledge-base allowlist 在 retrieval 之前选择，并且不会产生 knowledge
  claim、evidence、citation 或 provider call。其他每个 admitted request 都使用 admission
  保存的 QCS。缺失 decisive condition 时只会得到 reviewed conditional branch 或结构化
  insufficiency，绝不会得到推断出的 condition。
- completed result 精确持久化一个 normalized question、QCS 与 provenance、completed
  outcome、answer text、Evidence Set identity、有序 item identity、snapshot identity
  以及 knowledge-version identity。Evidence-Gated Answer 和 Generation Unavailable
  还会持久化一个包含相同值的 provider-input identity record。仅 owner 可见的 completed
  projection 携带精确 assistant binding、冻结 answer text 与冻结 evidence summary，使
  transport 比较 retained projection 而不是推断。Insufficient Evidence Reply 会保留一条
  structured reply，其中有其 outcome、精确 reason 与 QCS identity；其他 completed outcome
  都不会保留它。任何后续 consumer 都不能替换其中任一值。
  request header 绑定精确的 user message，每个 completed result 或有 assistant 支撑的
  terminal 都绑定精确的已持久化 assistant message。若 admission 后 assistant persistence
  本身失败，completion 会 rollback，既有 user binding 只能保留一个没有 assistant、answer
  text 或 outcome 的 `failed` `ANSWER_EXECUTION_PERSISTENCE_FAILED` terminal。
  `ChatMessage.answer_execution_id`
  是可变的 lookup index，因此 reload 与 history 会在同一 private conversation 中发现并
  验证不可变 binding；清空该 index 不会将已绑定 message 降级为 legacy trace projection，
  将该 message 移到另一个 private conversation 会 fail closed，非空且矛盾的 index 也会
  fail closed。Generation Unavailable 只为证明其 closed boundary 保留冻结 identity：其
  answer projection 不包含 knowledge claim、citation、source 或 evidence preview，且绝
  不能成为 supported-answer projection。
- normal HTTP、terminal SSE event、关联的 private persistence、reload 与 history 只
  projection retained result。它们不得执行 retrieval、重新选择 evidence、重新切分
  snapshot、调用 generation，或根据 message text、gate flag、source count、score 或
  legacy trace 推断 outcome。`ChatMessage.rag_trace` 可以保留为有界的 diagnostic
  compatibility artifact，但不能作为任何关联 Answer Execution 的 semantic input。对于关联
  execution，其持久化形式只包含有界的 operational metadata，例如 gate state、step、count、
  provider identity、timing 与 error classification；它排除 question、QCS、answer text 或
  preview、evidence content、provider-visible generation envelope 与 private history。client
  只有在 assistant identity 等于 execution binding、completed execution identity/state、
  answer text、normalized question、完整且不重复的 QCS/provenance、显式 outcome 与完整
  outcome-specific evidence summary 全部等于冻结 execution projection，且 question、QCS
  与 provenance 等于 normalized submitted turn 时，才可接受 completed SSE terminal。一个
  Insufficient Evidence Reply 会携带一条 structured reply，其 outcome、reason 与 QCS identity
  必须在 terminal stream 与 execution 中完全相等；其他 outcome 都不能携带它。有效 terminal
  只能是完整 framed 的 SSE `event: done`，且 data payload 必须是未加引号的精确字面量
  `[DONE]`。该 frame 的 blank-line separator 之前 EOF、带引号或其他被修改的 marker，或在
  任何其他 event 上出现该 marker，都是缺失或相互矛盾的 terminal。重复的 `done`，或 `done`
  后任意 semantic frame，都是 application failure。terminal field 缺失、重复或相互矛盾是
  application failure，绝不能借此伪造 insufficiency 或其他替代 outcome。在 completed
  terminal field 后出现 error 或 cancellation，是 application failure 而不是 stop，client
  会从 projection 清除 completed outcome、structured reply、evidence summary 与
  diagnostics。
- authentication failure 发生在 admission 之前。未恢复的 retrieval 或 provider failure，
  以及每个 execution、stream 或 persistence failure 都是 application failure，绝不是
  insufficiency。只有 retriever 已返回实际 candidate result 时，才可以为已恢复的 fallback
  保留 diagnostic；它不能从 provider exception 伪造空 result，再将其标记为证据不足。
  execution 仍在 running 时 stream interruption 会取消并等待 durable stopped event。SSE
  admission 会在 closed result 可被 delivery 前追加 `stream_delivery_pending`。一个 completed
  result 若带有该 pending record、但没有 delivery-completion 或 interruption record，就不能通过
  reload 或 history projection：这是 application failure，而不是 completed-answer replay。normal
  HTTP 不创建 SSE delivery lifecycle record，并可在普通 persistence 完成后 projection
  同一份 closed result。若 closed result 已开始 projection 后 delivery 中断，只追加的
  `stream_delivery_interrupted`
  event 会保持 closed result 不可变，但使它不可再 projection；reload 与 history 以
  application failure 失败，而不是把 `completed` 改成另一种 terminal state。缺失
  terminal event、相互矛盾的 terminal state payload，或相互矛盾的 persisted message binding
  都会让 projection 发生 application failure。response layer 会为 ASGI send exception 或
  disconnect 执行相同的 cleanup，且只有外层 ASGI transport observer 在越过每一层
  response-buffering middleware 后成功写入 terminal body 并完成 response finalization、随后追加
  `stream_delivery_completed` 后，`done` 才算 delivered。persistence failure 会 rollback 未提交的 completion，而不会留下 partial
  completed execution；若
  assistant 仍无法持久化，owner 只能在冻结的 user turn 上看到显式的 failed persistence
  terminal。admission 后的 SSE failure 会在其 `error` 与 `done` 前 projection 已保留的
  non-completed execution 与冻结 QCS；若 assistant binding 存在则一并投影，且不会暴露
  completed outcome 或 evidence summary。在 provider call 前，完整 provider-visible input
  必须等于从冻结 question、精确 QCS identity 与有序 condition record、Evidence Set、每个
  source 的全部 field（包括 citation marker）、有序 item/citation identity、由 source
  重新计算的有序 snapshot、knowledge-version identity，以及存在时完整 response contract
  构建的同一份 record。JSON object parsing 会在任意嵌套层级拒绝 duplicate key；后出现的 key
  绝不能静默覆盖较早的 frozen field。比较不得 trim、normalize、省略或重建 provider-visible
  field。input 缺失、格式错误或不匹配，已存在但格式错误的 observed generation envelope，或
  有效 envelope 不匹配，都是 application failure，而不是 Generation Unavailable。未恢复的
  provider exception，或 route configuration 没有产生任何已完成 provider call，同样是
  application failure；Generation Unavailable 只保留给已经完成 provider call、但没有可用
  frozen-boundary answer 的情况。
- 经单独授权的 document tombstone 可以向 completed private execution 追加
  evidence-redaction event。projection 会移除历史 excerpt、将 item 标记为 withdrawn，
  同时保留 Evidence Set、item、snapshot 与 knowledge-version identity。原始 terminal
  event 永远不会被改写。completion 会在追加 terminal event 前锁定冻结 evidence 对应的
  document，tombstone 会在标记 withdrawn 前锁定同一批 document。若 completion 获取锁后
  发现 document 已 tombstoned，它会在同一 transaction 中追加 redaction event，并立即返回
  redacted projection。

## 后果

每个 transport 都暴露同一份 result，而不是各自重复 answer semantic。user 可以在其
private history 中检查 per-turn QCS 与 condition provenance、在后续 turn 修改 condition，
并确信 history 不会从 memory 或其他 conversation 静默获得 condition。stopped 或 failed
request 与 insufficiency reply 在可见状态上不同，reload 也不能将它提升。若 closed result
的 SSE delivery 被中断，它不会被 replay 为 completed answer：其保留的 interruption event
会使后续 projection fail closed。assistant persistence failure 同样不会被静默擦除或转换为
reply：保留的 failed execution 会在 admitted user turn 上可见，而不会伪造 assistant message。

私有 execution record 有意不是全局 canonical record：它们的 content 遵循 private
conversation retention 与 deletion。其 immutability 因而只在 retained 期间成立，而
append-only event 为 reload 与 cross-surface equality 提供必要的 audit boundary。历史
withdrawal redaction 在防止 retained excerpt 继续暴露 withdrawn content 的同时保留
identity。

legacy lexical heuristic migration profile 可以继续产生 diagnostic，但不能生成 product
Evidence Set、completed supported answer 或 Generation Unavailable result。因此它会冻结
显式 insufficiency，而不会重新激活 legacy gate 或 provider path。

## 范围

本 ADR 不会 activate 或 approve provider route、定义 provider fallback policy、创建
publication、replacement 或 withdrawal authority、修改 feedback retention，或增加后续 UI
workflow。它只定义 transport-independent execution boundary 以及对已授权 evidence
decision 的 projection。document tombstone authority 仍在本 ADR 之外；本 ADR 只规定一旦
该 tombstone 获得授权后，它对私有 historical-redaction 的影响。
