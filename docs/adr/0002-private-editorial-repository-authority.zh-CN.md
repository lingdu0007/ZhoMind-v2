# ADR 0002：Private Editorial Repository 权威

状态：已接受

日期：2026-09-05

更新：2026-09-06

## 背景

产品在 Reviewed Release Bundle intake、Candidate Build 和 publication 之前需要保留的
editorial authority。现有运行时 document 行和部署副本无法证明已经 review 的
revision、source definition、role-separation decision、assurance support 或 source
freshness。若允许运行时 material 回写 editorial record，就会反转这一 authority
boundary，并使后续 reconstruction 无法审计。

## 决策

- 使用现有不可变的 `canonical_records` 和只追加的 `canonical_events` 表作为
  Private Editorial Repository。不可变的 `entry`、`editorial_revision` 和 `source`
  record 保留权威 snapshot；基础层已支持这些 canonical kind 和 record class，因此
  不需要 schema migration。
- source definition 只存储一次。source availability 作为只追加的
  `editorial_source_event/v1` fact 记录，并从 event trail 派生当前 availability，
  而不是修改 source record。Author 的 availability 值只是提议：新的 source 从
  `changed_or_unreachable_awaiting_review` 开始，只有已接受责任的 Maintainer 追加
  合格的 `verified_usable` fact 后才可支持 review 或 export。该 fact 必须使用 source
  aggregate、绑定精确的已接受 Maintainer event、指明包含该 source 的 revision，并以
  已接受的 Maintainer 作为 recorder。缺失、外来、畸形或不合格的 source trail 都会
  解析为 `unknown`；T01 记录这项已认证 access decision，而不是对任意 URL 或 locator
  执行运行时 fetch。不可用的 source 会阻止 answer eligibility 和 export。decisive
  loss 是持久的 authority evidence；它可以将未来已发布的 entry 转为 Needs Re-review，
  而本 ticket 不会创建 publication path。
- 将 canonical coverage position、assurance level、source tier、source access
  scope 和 revision change kind 词汇定义为封闭的 Python contract enum。entry
  validation 保留完整的 decision schema、role fact、source link 和
  assurance-specific evidence requirement。wording-only revision 只能规范化 title 与
  body 的空白；标点、大小写、语义 token、比较运算符和结构化数据都需要 material
  review。high-impact requirement 会从支持语言中的 authored text 及客户端提交的
  claim metadata 保守推断，因此客户端不能通过重新标注内容，或在同一 section 附加
  无关的 ordinary claim，来降低 Claim-Evidence Link requirement。
- 从服务器端 active member 行派生 Author、Approving Reviewer、Maintainer 和
  Administrator 的 authority。Author、Reviewer 与 Maintainer 可以在各自的 entry
  boundary 内工作；Author 或 material reviser 不能批准其 material revision。
  被指定的 Maintainer 必须在 review、approval 或 export 之前为该 revision 追加责任
  接受。Administrator 只能接收已批准的 export，不能检查或修改私有 editorial record。
- 从第一个保留的 editorial revision 开始使用显式 entry `schema_version` 1。此前没有
  需要 migration 的 editorial schema；未来不兼容的版本必须先提供自己的 migration
  才能被接受。
- 对 Release-Assured entry，所有指名 reference 都必须解析为保留的 immutable 或
  authoritative canonical record，绝不能是 derived record 或 replaceable projection。
  frozen acceptance 必须是能通过 canonical schema 校验的不可变
  `delivery_acceptance_record/v1`。其当前 active fact 必须是从 `at_risk` 到 `active`
  的 `status_changed` event，由 canonical member 记录并带有 `checks_verified`，且为
  每一个 result 为 passed 或 carried_forward 的 selected check 保留相匹配的 verification
  attachment；其保留 scope 必须覆盖精确的 entry、contract、calibration 与 named gate。
- 确定性的、由 SHA-256 标识的 `editorial_export/v1` artifact 只能从保留的 authority
  record 和 event 重建。approval 和 lightweight acceptance 会保留绑定到精确
  entry/revision、可信 role、approval fact、qualified source definition/
  availability event hash、适用的 Release-Assured record/status hash 以及 approval
  当时 audit cutoff 的 snapshot。只有结构正确的 lifecycle、Maintainer 与 approval
  event 才能建立这些 fact。新的 export 会检查当前 authority fact；历史
  reconstruction 只使用其保留的 approval snapshot。拒绝凭据形态的 material、
  任意结构深度的非空 secret-bearing field 和 automatic-publication instruction。
  该 export 不是 bundle，不触发 intake，也不能写入 Candidate、已发布 knowledge
  version、运行时 document 或部署副本。
- 允许 T02 只能通过从本 repository 进行只读 reconstruction 来消费 export。
  `reviewed_release_bundle/v1` item 必须精确匹配保留的 approved artifact 及其 hash。
  manifest source revision 等于每个 artifact revision hash，且 item hash 覆盖其稳定
  identity、operation、artifact hash 与 artifact。intake 不接受 type coercion：
  `schema_version` 是 JSON integer 而不是 boolean，identity 必须是 JSON string。随后
  在不追加 export audit event 的前提下重新检查当前 source 与 Release-Assured authority。
  整包 integrity check 失败或不可变 identity collision 时，只记录针对新鲜
  `admission_attempt` 的有界且不含内容 audit，绝不创建 rejected bundle、item 或
  build-generation record。其不可变 bundle、item 和 build-generation record 可以创建
  独立的、可恢复 Candidate Build job，但导入只创建其计划。明确的 System Administrator
  dispatch 是唯一会记录耐久 `dispatched_at`、追加带有当前 attempt 和该管理员 `member:`
  identity 的 `dispatched` event，并让 queued work 有资格 enqueue 或由 startup recovery
  恢复的 action；明确 retry 会为其新 attempt 追加带有同样当前-attempt authority 的
  `retry_dispatched`，并建立 `cancel_or_await_candidate_build` action。runtime enqueue
  与 startup recovery 都要求当前 queued attempt 的这条只追加 evidence，绝不会只凭可变的
  `dispatched_at`。并发的 Candidate-generation allocation 争用会重新读取胜出的 generation，
  并在有界 attempt budget 内重试同一 immutable intake；耗尽时会返回 retry-required，
  不留下 admitted intake record，也不会将不同的有效 bundle 视作 immutable identity conflict。
  bundle 通过只追加的
  `received`、`validating`、`validated`、`processing` 以及 `completed` 或
  `completed_with_rejections` event 前进。其不可变 snapshot 会保留是否有任一 item 被
  rejected，因此 mixed valid/rejected work 绝不会被表示为 complete batch success。它会在
  有效 Candidate job 未 terminal 时保持 processing。每条
  worker、retry、recovery、derived write、vector
  call、Candidate finalization 与 cleanup path 都会在使用前重建并重新匹配这些 frozen
  input。Candidate finalization 会在 indexing 后、Candidate persistence 前立即重复验证
  approved export、source 与 Release-Assured authority。Candidate embedding 只基于
  configuration schema、active flag、model 与
  dimension 计算 fingerprint，因此其 Candidate-specific collection 与 normal runtime
  retrieval 分离，且不含 endpoint 或 secret。inactive frozen configuration 没有 Candidate
  vector collection；没有其 frozen Candidate fingerprint 的 cleanup 不会查询或删除任何
  collection，也绝不会回退到 normal retrieval fingerprint。build stage 是封闭的（`queued`、`parsing`、
  `chunking`、`indexing`）；terminal status 独立，worker 必须在修改 running job 前证明其
  精确 attempt、owner 与未过期 lease。Candidate 与 legacy document job 使用同一进程范围、
  有界的 build-worker slot pool，独立 dispatcher 不能放大配置的并发数。更高 generation 会 supersede unfinished work，
  但不会抹去 historical evidence；未验证或失败的 cleanup 会保留为 durable pending
  obligation，且只有重新匹配 frozen input 后才可删除。intake 和 worker 都不能回写
  private editorial authority、legacy runtime document 行或任何 published knowledge
  pointer。Candidate 仍是隔离的 derived result；Candidate inspection、publication、
  replacement 与 withdrawal 都不属于 T02 的职责。
- 将 Candidate Build 的 recovery 与并发视为耐久的 authority boundary。intake 中 source 的
  可用性只来自 verifier 对保留 authority fact 的重建，绝不相信 artifact 内 Author 声明的
  `availability` 值。bundle completion 会在重建 state 与 child-job status 前锁定 bundle
  aggregate。`parsing` 会先验证并读取已批准 export，随后 `chunking` 只消费已解析数据。
  等待外部 indexing 时，worker 会条件化续期其精确 owner/attempt lease，并让这些 heartbeat
  与 indexing coroutine 竞逐。续期失败或被接管时，会先取消并等待该 coroutine，再使 worker
  停止，既不进行 terminal mutation，也不 cleanup derived data；recovery owner 会锁定并重新检查过期 job，
  持久化带有 `derived_cleanup_pending` 的 interrupted stale-worker fence、提交该 fence，
  随后才协调 matching frozen input。Candidate job event 会快照冻结的 editorial source revision、input hash、
  结构化 failure reason 与 allowed next action。recovery 会在 mutation 前立即锁定并
  重新检查每个选中的 queued、running 或 cleanup-pending job。这些路径都不能创建或改变
  Published Knowledge Version 或 publication pointer。

## 后果

Private Editorial Repository 仍是耐久、access-controlled 的 editorial authority，
且可从保留数据证明 revision/export hash。T02 通过不可变 Reviewed Release Bundle
intake 和可恢复 Candidate Build record 消费它，而 T04 publication 仍是独立职责。
bundle verifier 会重新检查当前 authority，但绝不会回填或改写 private editorial
record。source availability 已是 fail-closed eligibility 的权威事实；后续 publication
与 maintenance path 必须消费这项保留 evidence，而不是从 runtime copy 推断。
recovery 只会重新 enqueue 带有其当前 attempt 的耐久 administrator-dispatch evidence 的
queued job，绝不会只凭可变 timestamp，并在成功 requeue 时追加 `requeued_on_startup`；
它会将缺失、过期以及由先前 runtime 持有的
candidate lease 视为 interrupted work，并且必须在 retry 前协调 derived data。这样跨
process restart 仍保持同一 authority boundary，同时不授予 recovery 任何 publication
capability。recovery 与 runtime enqueue 都会先锁定并刷新当前持久 job，再判断
eligibility；recovery 会在 mutation 前立即锁定并重新检查每个选中的 queued、running 或
cleanup-pending job，且只有在该当前 job 仍为 queued 或 running 时才记录成功 requeue。
每个只追加的 Candidate job event 都会快照冻结的 editorial source revision、input hash、
结构化 failure reason 与 allowed next action，因此 retry 不会抹去先前 attempt 的审计。
stage transition 发生在它所命名的工作之前：`parsing` 会在 `chunking` 使用已解析数据
之前验证并读取已批准 export。外部 indexing 会条件化续期精确的 worker lease；无法续期
的 worker 会先取消并等待 indexing coroutine，再停止且不做 terminal mutation 或 cleanup，
之后由已提交 fence 的 recovery 负责 interrupted/retryable transition 与
matching-input reconciliation。
administrator cancellation request 只有在 worker control 确认收到后才成功。false 的
no-task result 或 exception 会留下带结构化 reason 和 pending derived-data reconciliation
的耐久 failed Candidate，绝不会被视为 successful cancellation。
recovery 绝不能让 stale worker 或 input mismatch 将 Candidate 变成 terminal success、
删除未经验证的 derived data，或移动 Published Knowledge Version。unfinished work 被
supersede 后会携带明确 cleanup obligation，直到 matching-input reconciliation 成功。
Release-Assured reference 只有在 canonical record 属于适当的 authority record，且 frozen
delivery-acceptance record 正在 active 地覆盖精确 editorial authority 时才可通过，
否则 fail closed。
