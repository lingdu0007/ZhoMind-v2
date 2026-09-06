# 规范化产品契约

状态：规范性产品契约；ticket 13 至 19 的增量基础

## 目的

规范化契约层为后续产品切片提供稳定标识、封闭状态词汇、不可变记录和只追加
事件，同时不重新解释或删除现有运行时记录。在迁移期间，它有意与旧的
`documents`、`document_jobs`、`chat_messages` 和反馈表分离。

本文件对其定义的产品规则具有规范效力。Ticket 必须保持这些规则；若要修改，必须在
同一变更中更新对应 ADR、英文规范原文及其中文镜像。

## 标识

`StableIdentity` 是 `(kind, value)` 对。事件使用独立的 `event` kind；保留的
业务 kind 包括：`member`、`team_invitation`、`admission_attempt`、`entry`、
`source`、`bundle`、`bundle_item`、`build_generation`、`candidate`、
`published_knowledge_version`、`answer_execution`、`evidence_set`、
`evidence_snapshot`、`maintenance_item`、`delivery_acceptance_record`、
`collection`、`capability`、`configuration`、`concurrency`、`corpus`、
`data_boundary`、`deployment`、`editorial_revision`、`embedding_profile`、
`host`、`migration`、`objective`、`product_path`、`product_revision`、
`prompt_envelope`、`provider_route`、`public_claim`、`retrieval_profile` 和
`user_boundary`。标识不可变。标题、文件名、代次编号、分数、环境标签或当前
`latest` 指针都不是标识。

## 状态与事件

Python 契约模块定义 PRD 中关于条目生命周期、bundle 接收、构建阶段与终态、
回答执行与结果、维护和验收的状态词汇。`validate_transition` 在持久化前拒绝
非法转换。`canonical_records` 是按 `immutable`、`append_only`、
`authoritative`、`derived` 或 `replaceable_projection` 分类的不可变快照。
`canonical_events` 只追加，并通过追加状态变更记录而不是改写历史。

当旧记录无法证明时，任何投影都不会把 `answer_eligible`、编辑审批、来源授权、
发布资格或验收默认为 true。旧记录缺少的身份、assurance、来源、bundle、条件
和证据字段会明确保留为 `unknown`。

## 私有编辑权威

在后续 bundle intake 之前，Private Editorial Repository 是 Engineering Decision
Entry 的唯一权威。它使用不可变且具权威性的 canonical `entry`、
`editorial_revision` 和 `source` record，以及只追加的 canonical event；它不会
写入 `documents`、Candidate record、Published Knowledge Version、运行时 cache
或部署副本。不可变的 `entry_id`、每个 `editorial_revision` identity 与每个
source identity 永远保留其原始含义。source record 保留不可变的定义；当前 source
availability 从只追加的 `editorial_source_event/v1` 轨迹重建，绝不写回该 record。
合格的 availability fact 必须使用 `source` aggregate、指明该 source 所属的保留 entry
与 revision、绑定精确的 Maintainer responsibility-acceptance event，并由该已接受的
Maintainer 记录。该 source 必须属于所指名的 revision。外来的、畸形的或不合格的
event 不能覆盖既有 fact；它们与缺失的轨迹都会使 availability 解析为 `unknown`。

canonical 词汇恰好包含八个 coverage position：
`rag_source_admission_and_chunking`、
`sparse_dense_hybrid_and_reranking_choices`、
`evidence_sufficiency_refusal_and_acceptance`、
`tools_and_mcp_permissions_and_failure_behavior`、
`agent_context_state_and_memory`、
`orchestration_retry_human_intervention_and_side_effects`、
`provider_failure_and_observability` 和
`prompt_injection_isolation_and_security`。assurance level 为
`source_grounded`、`claim_linked` 与 `release_assured`。source tier 为
`primary_evidence_source`、`reproducible_engineering_evidence`、
`secondary_discovery_source` 与 `bounded_internal_case`；access scope 为
`public` 与 `controlled_internal`；revision change kind 为 `material` 与
`wording_only`。

编辑 revision 保留 title、coverage position、assurance、Author、指定的 Approving
Reviewer、负责的 Domain Knowledge Maintainer、review date、适用版本与结构化
condition、freshness trigger、source metadata、chunk strategy、supported 与
Boundary acceptance material、必需的作者决策段落、section-source relationship、
claim，以及可选的 replacement 或 supersession relationship。public source 必须有
经清理的 canonical HTTPS URL；controlled source 必须有经清理的 `controlled://`
locator。不可用、不可访问、暗中重定向或 tier 不合格的 source 不能支持 review 或
export。

由服务器派生的 active、非 Administrator member identity 是 editorial role 的
权威。Author、Approving Reviewer 与 Maintainer 被保留为独立的 role fact。Author
或 material reviser 不能审批同一 material revision；Reviewer 可以编辑，但作为
material reviewer-reviser 时必须指定不同的 reviewer。System Administrator 不能
检查或修改 Private Editorial Repository，只能接收已批准的 export。被指定的
Maintainer 必须为每个 revision 追加明确的责任接受 event，之后才能 review、approval
或 export。

私有 lifecycle 写入 `draft`、`evidence_collected` 与 `editorial_review` event。
evidence collection 和 review 都要求 revision 完整。material change 会创建新的
不可变 revision 并要求 Editorial Review；wording-only change 要求有已批准的 base
revision、独立的 lightweight acceptance，且只允许规范化 title 和 authored body 中的
空白。任何标点、大小写、token、比较/运算符或结构化权威数据的变化都是 material。
T01 不记录 Reviewed Release Bundle、
Candidate Build、publication、replacement 或 withdrawal action：这些由后续 ticket
负责。对 `unavailable_for_new_evidence` 的 availability event 会记录 decisive
source loss；若较后的产品路径已创建 published entry，则将该 entry 转为
`needs_re_review`；所有其他 source-state projection 均 fail closed。

Author 提供的 source availability 只是提议。新的 retained source 初始为
`changed_or_unreachable_awaiting_review` 的 record state，而在已接受责任的 Maintainer
通过已认证的 source-availability command 追加合格的 `verified_usable` fact 之前，
其派生 availability 仍为 `unknown`。T01 有意记录这项 access decision，而不是在
运行时抓取任意 public URL 或 controlled locator。缺失、陈旧或不合格的 source event
都会重建为 `unknown`，因此不可变 record 的 state 不能使 entry eligible。

Source-Grounded revision 要求经过 review 的 section support。Claim-Linked 会为
material claim 增加 Claim-Evidence Link。Release-Assured 还要求冻结的 canonical
contract、calibration、delivery-acceptance 与 named-gate identity；它们必须解析到
保留的 immutable 或 authoritative canonical record，绝不能是 derived record 或
replaceable projection。不可变的 delivery-acceptance record 必须通过其 canonical
schema 校验，并具有当前从 `at_risk` 到 `active` 的 `status_changed` event；该 event
必须由 canonical member 记录、使用 `checks_verified` reason，且为每一个 result 为
passed 或 carried_forward 的 selected check 保留完整 verification attachment。它还必须
在保留的 scope 中绑定这个精确 entry 以及所指名的 contract、calibration 与 gate
identity。material 的 prescriptive、numeric、version、security、privacy 及其他
high-impact claim 始终需要 Claim-Evidence Link。high-impact text 会在两种支持语言中
根据 statement 和 authored decision section 保守推断，也会参考客户端提交的
`claim_kind` 或 `material`；客户端不能通过错误标注内容，或在同一 section 附加无关的
ordinary claim，来降低该要求。

answer eligibility 是派生投影，绝不是 publication command。在 Published 或受限的
Needs Re-review state 之外它均为 false；缺少 approval、source availability、
显式 applicability、已知 contradiction、integrity defect 或 decisive source loss
时会 fail closed。Needs Re-review 最多七天可保持 eligible，且仅在没有已知 blocking
reason 时成立。approval 和 lightweight acceptance 会保留确定性的 authority snapshot，
并将其绑定到精确的 entry 与 revision：可信 role assignment、approval identity/status、
精确的 verified source definition/availability event、适用的 Release-Assured
record/status hash，以及 approval 当时的 editorial audit cutoff。只有结构正确的
lifecycle、Maintainer 与 approval event 才能建立这些 fact；畸形或外来的 event 会
fail closed。新的 administrator export 会先验证当前 source 与 assurance fact，再重建
已批准的 snapshot；历史 reconstruction 只读取保留的 snapshot，因此较后的运行时或
source-state 变化不能改变旧 artifact 或其 audit trail。确定性的
`editorial_export/v1` 包含稳定的 revision/source identity、role、approval 和 audit
history，并计算 SHA-256 hash。export 会拒绝凭据形态的值、任意结构深度的非空
secret-bearing field，以及以 field name 或祈使式文本表达的 automatic-publication
instruction。它不是 Reviewed Release Bundle，不能修改或自动发布运行时 material。
保留的 entry schema 从显式的 `schema_version` 1 起步；此前不存在需要 migration 的
editorial schema，未来不兼容的版本必须先加入明确 migration 才能被接受。

editorial export、Reviewed Release Bundle manifest/item 以及冻结 Candidate input 的
integrity hash 使用同一份 canonical JSON serialization：key 排序、紧凑 separator、
对非 ASCII code point 进行 ASCII escape，然后取 UTF-8 byte。这样即使包含 Unicode
value，也保持 Private Editorial Repository 已确立的 export byte contract。

## Reviewed Release Bundle Intake 与 Candidate Build

System Administrator 只能导入不可变的
`reviewed_release_bundle/v1` manifest。manifest 保留稳定的 bundle identity、
schema version、editorial source revision、UTC export time、bundle 级 SHA-256、
每个 bundle-item 的 SHA-256，以及显式 item operation：`create`、`replace`、
`no_op` 或 `proposed_withdrawal`。bundle integrity 会在持久化任何 bundle item、job
或 Candidate 之前 fail closed：不支持的 schema、畸形 identity/timestamp、hash、
credential、automatic-publication instruction、重复 item identity 和冲突 entry
operation 都会拒绝整个 bundle。以相同 bundle identity 和 bundle hash 重复导入是
idempotent；相同 identity 配合不同 hash 则是 conflict。manifest 的 editorial source
revision 必须等于每个 item 已批准 export 的 `revision_sha256`；bundle-item hash 是对其
稳定 item identity、operation、artifact hash 和 artifact 的规范化 hash。整包 integrity
rejection 或不可变 bundle/item identity collision 只会针对一个新鲜、不可变的
`admission_attempt` record 及其只追加 event 记录有界且不含内容的 canonical import
audit。`schema_version` 必须是 JSON 整数，不能把 boolean 视为整数；manifest、bundle
与 item identity 都必须是 JSON string，intake 绝不将 identity 或数值强制转换为可接受
输入。它绝不会留下 partial item、job、不安全的 rejected manifest copy，或 rejected 的
`bundle`、`bundle_item`、`build_generation` aggregate。
并发的 Candidate-generation allocation 争用会重新读取胜出的 generation，并对同一
immutable intake 作有界重试；耗尽时会返回 retry-required 结果且不留下 admitted intake
record，而不会将不同的有效 bundle 误报为 bundle-identity conflict。

bundle 与每个 item 都是不可变 canonical record。有效的
`editorial_export/v1` 会通过从 Private Editorial Repository 重建精确的、已批准的
保留 export，并比较其 SHA-256 和完整 artifact 来验证。这个只读验证还会检查当前
source availability 和 Release-Assured authority fact。source 的可用性只以 verifier
从这些保留 authority fact 重建的结果为准，绝不相信 artifact 内由 Author 声明的
`availability` 值。验证不会创建 export audit event，也绝不会写入 private editorial
record。metadata、approval、role snapshot、source、assurance、access、chunking 或
acceptance failure 都是 item-local 的 `rejected` 结果，带有结构化 blocking field 和
allowed next action；独立的 valid item 仍保持 admitted。`no_op` 与
`proposed_withdrawal` 保持为不可变、可见的 operation plan，
不会创建 Candidate Build、publication、replacement 或 withdrawal side effect。
verifier 的基础设施或执行 failure 不是 item defect：它会中止 intake transaction，且不
持久化 bundle、item、job 或虚假的 rejection，因此同一个 immutable bundle 可以重试。

不可变 bundle record 仍为 `received`；其 intake state 从只追加的
`received -> validating -> validated -> processing -> completed` 或
`completed_with_rejections` event trail 重建。不可变 bundle snapshot 会保留 intake
是否拒绝过任何单独 item。含 admitted Candidate work 的 bundle 会保持 `processing`，直到
每一个 Candidate Build 都成为 `candidate_ready` 或 `superseded`；failed、canceled 或
interrupted job 会让它保持 `processing`。没有 Candidate Build 的 bundle 只有在没有
rejected item 时才会在 `processing` 后立即完成显式 operation plan；否则，以及 mixed
valid/rejected Candidate work 完成后，它会进入 `completed_with_rejections`，绝不能被呈现为
complete batch success。item-local rejection 因而不会阻止独立 valid sibling，也不会坍缩成
whole-bundle integrity rejection。完成判定会在重建当前 intake state 与 child-job status
前锁定 bundle aggregate，从而串行化 terminal aggregate decision。

每个 admitted 的 `create` 或 `replace` item 都会创建一个可恢复的 Candidate Build
job 和一个不可变的 `build_generation` input record。它们保留 bundle、bundle-item、
entry、runtime-document、requested-generation、editorial-source revision、approved-artifact
`input_sha256`、完整的 `frozen_input_sha256`、chunk-strategy 以及不含 secret 的
effective embedding configuration identity。frozen-input hash 覆盖 input schema、bundle
与 item identity/hash、entry/document identity、generation、editorial revision、artifact
hash、chunk strategy 和 embedding configuration。导入只创建这项工作计划，不会 enqueue
或启动它：初始 allowed next action 为
`dispatch_candidate_build`。只有明确的 System Administrator dispatch 可以写入耐久的
`dispatched_at`、追加由该管理员 `member:` identity 记录且绑定当前 attempt 的
`dispatched` event 并将 queued job enqueue；明确的 administrator retry 会为新 attempt
追加带有同样当前-attempt authority 的 `retry_dispatched`，并建立
`cancel_or_await_candidate_build` action。runtime enqueue 与 startup recovery 只会根据
当前 attempt 的这条只追加 administrator event 授权 queued job，绝不会只根据可变的
`dispatched_at`。新的 input record 与 dispatch event 必须使用
`candidate_build_job_event/v1`、精确的 queued transition 与 action shape、冻结的
editorial revision 与两份 input hash，并且其 `member` identity 仍须解析为 active 的当前
System Administrator 及权威 identity record。早于 `0018` 的 immutable input 与 event
绝不会被改写：升级会重新计算并在 job 上持久化完整 hash；只有当历史 dispatch event 缺少
`frozen_input_sha256`、其 immutable input record 同样早于该字段、重新计算的 hash 与 job
匹配，并且其余精确的 current-attempt 与 administrator 检查全部匹配时，runtime 才可承认
该历史 event。这一兼容只承认先前明确的 authorization，绝不会从可变 timestamp 或
migration state 创建 authorization。worker、
retry、recovery、cleanup 或 completion 创建、索引、删除或 finalization Candidate-derived
data 之前，都会重建不可变 canonical record 并检查每一个可变 job binding；任何不匹配
都会 fail closed，既不会使用也不会删除 Candidate-derived data。completion 会在 indexing
后、Candidate persistence 前立即重复验证 approved export、source 与 Release-Assured
authority，并在这一期间持有 entry、保留 source 及每个 Release-Assured reference 的共享
canonical authority lock。source availability recorder 与 delivery-acceptance status writer
会取得对应的同一把 lock，因此任何 authority change 都不能在 final verification 和
Candidate commit 之间追加。finalization 必须使用 verifier 提供的 authority-fence
context，绝不允许回退到未加锁的检查。在 SQLite 上，该 context 会在 re-verification
之前取得 `BEGIN IMMEDIATE`，从而将 editorial-authority writer 串行化到 Candidate
commit 之后，等价于支持 record lock 的数据库中的对应保护。Candidate Build storage
与旧的 `documents`、`document_chunks` 分离；其 Candidate chunk 和 dense vector 都是
derived data，对普通 retrieval 不可用。Candidate 与 legacy document worker 使用同一
进程范围、有界的 build-worker capacity，新增 dispatcher 也不能放大配置的并发数。
冻结的 Candidate embedding configuration 只包含
configuration schema、active flag、model、dimension 及它们不含 secret 的 fingerprint：
绝不保留 endpoint、user info、query parameter、credential 或 secret。Candidate vector
operation 使用该 frozen Candidate fingerprint，因而写入 Candidate-specific collection，
绝不使用 active normal-retrieval collection。它在 worker 启动时仍必须匹配 active 的
non-secret configuration；不匹配会使 job 失败，而不会在不同 configuration 下静默构建。
derived-vector cleanup 使用 frozen fingerprint，而非 cleanup 时恰好 active 的 embedding
profile；cleanup backend failure 要求在 retry 前完成 reconciliation。inactive 的 frozen
configuration 没有 Candidate vector collection：没有 frozen Candidate fingerprint 的 cleanup
不会查询或删除任何 collection，也绝不会回退到 active normal-retrieval fingerprint。

封闭的 build-stage vocabulary 为 `queued -> parsing -> chunking -> indexing`；terminal
outcome 是独立的 job status 与 terminal state：`candidate_ready`、`failed`、`canceled`、
`interrupted_retryable` 或 `superseded`。stage 绝不会变成 terminal marker，
`candidate_ready` Candidate 仍然没有 published。worker 只有在其精确 owned attempt、
精确 lease owner 与未过期 lease 全部匹配时才可修改 running job；stale worker 不能
finalize Candidate、写入 terminal result 或 cleanup derived data。阶段转换发生在该阶段
所代表的工作开始之前，因此 parsing 与 chunking 的 timing 保持真实：`parsing` 会在
`chunking` 只消费已解析 artifact 之前验证并读取已批准 export。在等待外部 indexing
时，worker 会通过条件化的持久更新续期其精确 owner/attempt lease，并让 indexing
coroutine 与这些 heartbeat 竞逐。若续期失败，或已由其他 owner 或 attempt 接管，worker
会先取消并等待 indexing coroutine，再停止，既不进行 terminal mutation，也不 cleanup
derived data。随后 recovery owner 会锁定并重新检查过期 job，只有它可持久化其
`interrupted_retryable` stale-worker fence 与 `derived_cleanup_pending`、提交该 fence，
随后才只在 frozen input 匹配时协调数据。parser、chunking、
authority/source、indexing、cancellation、cleanup 和 enqueue failure 都会记录有界的
结构化 reason。每个只追加的 Candidate job event 都会快照冻结的 editorial source
revision、input SHA-256、`failure_reason`（如有）和 `allowed_next_action`，因此 retry
对可变 projection 的 reset 不会抹去先前 attempt 的诊断；因不可变 input 无效或
derived data 尚未协调而被阻止的 retry 也会追加其 blocking event。administrator
cancellation request 只有在 worker cancel dispatch 确认收到后才成功；false 的 no-task
result 与 exception 一样，会留下耐久的 `CANDIDATE_CANCELLATION_REQUEST_FAILED` job，
并标记 `derived_cleanup_pending`，要求在 retry 前 reconciliation，绝不会被展示为
successful cancellation。retry 要么以精确的 frozen accepted input 创建新的 attempt，
要么在这些不可变 input 不再可验证时要求新 bundle。retry、startup requeue 与 runtime
enqueue 都会先锁定并刷新当前持久 job 事实，再判断 eligibility。每次 recovery 选择
都会在 mutation 前立即锁定并重新检查当前 queued、running 或 cleanup-pending job。
startup 只会重新 enqueue 带有当前 attempt 的耐久 administrator dispatch evidence 的
有效 queued work，绝不会只凭可变 dispatch timestamp；并且只有刷新后的 job 仍为 queued
或 running 时才追加 `requeued_on_startup`；它会将
缺失或过期的 running lease，或由另一 runtime instance 持有的 lease 标为
interrupted/retryable。queue-dispatch failure 本身也是耐久的 failed job，不会 silent
drop。

为同一 entry 接纳更高 generation 时，会 supersede 较早的 unfinished 或
`candidate_ready` Candidate。对于 allowed next action 为 `import_new_bundle` 的较早
failed job，也会 supersede，这使原 bundle 无需重试已损坏的 immutable input 即可进入
terminal aggregate state。较早的不可变 Candidate record 保留为历史 evidence，而其 job
projection 记录它不可 publish。对 unfinished running 或 interrupted work 的
supersession 会保留耐久的 `derived_cleanup_pending` obligation；failed generation
已存在的 cleanup obligation 同样必须保留。startup 会扫描拥有该 obligation 的 terminal
job，只有在再次匹配 frozen input 后才协调其 Candidate chunk 和 vector；若不匹配，会保留
这些 asset 与可恢复 obligation，绝不在未验证 binding 下删除数据。intake、retry、
recovery、cleanup、supersession 和 Candidate completion 都不会
创建或改变 Published Knowledge Version、legacy published generation 或 runtime
publication pointer。Candidate inspection、publication、replacement switching 和
withdrawal 仍由后续职责处理。

## 检索回答策略与授权候选池

`retrieval-answer-policy/pilot-v1` 是活跃的普通用户检索 profile。它是实际的
`sparse_bm25`，使用保留字面量的 tokenization、`k1=1.5`、`b=0.75`、candidate
depth `20` 以及有版本的 `retrieval-candidate-tie-breaker/v1`。该 profile 不接受
field boost：未记录的 boost 会被拒绝，不能静默改变 effective profile。它保持
reranking、lexical-answer anchor、semantic near-duplicate removal、query expansion
和 online LLM sufficiency judging 为 disabled。每个 retrieval result 与 trace 都携带
effective profile identity。

在后续 ticket 以 canonical Published Knowledge Version 替换 legacy runtime projection
之前，Pilot 的普通 pre-sufficiency Candidate Pool 只从当前 legacy published generation
的 compatibility projection 构建。只有当 chunk 所属的 `Document` 未被 withdraw、
其 generation 等于该 document 的当前 published generation 时，它才可能进入 pool。
ranking 前，pool 会为 entry 解析当前 Private Editorial Repository authority：当前
revision、lifecycle eligibility、准确的 section-level verified source relationship、
assurance、applicability、freshness 与 team-shared access scope。compatibility metadata
只用于将 chunk 绑定到这些当前事实；缺失、畸形或不匹配的 metadata 绝不授予
eligibility。其 lifecycle 必须为 `published`，或为仍处于七日 grace interval 内的
`needs_re_review`；known contradiction、integrity defect、已过期的 grace、不可用
source、不支持的 access scope 和不合格 assurance 都会 fail closed。该 pool 保留
entry、revision、publication、section、source、assurance、applicability、freshness、
access、chunk identity、经过 review 的 `decision_query` 以及 source 未截断的 content length。
通过 authority 的 candidate 可以标记为可供后续 evidence
selection 使用，但这个 pre-sufficiency boundary 不决定 sufficiency。在返回至多 20 个
candidate 前，pool 会以确定性方式去重 exact content 和重复的 `(entry, section)` pair。

Candidate Build chunk 与 Candidate record 不是这个普通 pool 的成员。Candidate preview
是一条显式的、仅 System Administrator 可用的隔离路径：
`GET /reviewed-release-bundles/candidates/{candidate_id}/preview`。它只接受 immutable 的
`candidate_ready` Candidate 以及其匹配 current attempt 的 `candidate_ready` build
chunk，并且会重建和验证完整 immutable Candidate binding：Candidate record、build job、
frozen input、bundle/item record 与 artifact，以及连续、content-hashed 的 chunk 和其准确的
section-source metadata。preview result 仅用于 diagnostic，保留 Candidate 而非
publication identity，并设定 `answer_evidence_eligible=false`；它不能返回给普通用户，
也不能被呈现为 product answer evidence。普通 runtime trace 只保留 normalized exclusion
reason，绝不保留被排除的 Candidate 或 unpublished chunk identity。

BM25 raw score（包括兼容的 `score` field）只对已经授权的 pre-sufficiency pool 排序。
raw score 绝不建立 eligibility、sufficiency 或 generated-answer decision。Evidence Set
selection 与其他后续职责不属于本 ticket。

`retrieval-answer-policy/lexical-heuristic-migration-v1` 只保留为显式的
migration/diagnostic profile。它的 strategy 和 candidate-pool scope 分别标识为
`lexical_heuristic_migration` 和 `legacy_migration_diagnostic`；它绝不能被标记或
视为 Sparse BM25。

## 证据充分性与冻结回答证据

只有活跃的 Pilot profile 可以将 Authorized Retrieval Candidate Pool 传入证据充分性
判定。decider 要求该 profile identity 和 `published_knowledge` pool scope；缺少或
不兼容的 boundary 都是 `no_eligible_published_evidence`。candidate count、non-empty
context、原始 BM25 score、ranking position、任意 score threshold 或 model judgment
都绝不能建立 sufficiency。

确定性的输入是不可变的 Query Condition Set（QCS）：normalized question 以及按顺序
排列的显式 `field`、`operator` 和 `value` condition。每一个选中的 item 都必须匹配该
QCS 中每一个 decisive applicability condition。对于缺失 condition、未解决的 review、
material conflict、不支持的 assurance 或缺少 Claim-Evidence Link support，decider 都会
fail closed。它还要求一个 governing 的 `recommendation_or_reviewed_branches` item，
并要求 comparison、diagnosis、acceptance review 或 implementation guidance 所需的每个
由问题形状决定的 complement。governing item 必须通过经过 review 的 `decision_query`
以确定性方式覆盖 normalized question；raw retrieval score、rank、non-empty context 或
model inference 都不能替代该 coverage。Claim-Linked evidence 要求每个支持 selected item
的 claim 都具有全部经过 review 的 `(section_id, source_id)` link。decider 按 canonical
item identity 而非 retrieval order 搜索，并且只能从 authorized pool 中选择满足这些要求的、
最小的可行集合。

唯一的 insufficient-evidence reason 是
`no_eligible_published_evidence`、`decision_not_covered`、
`decisive_condition_missing`、`material_evidence_conflict`、
`assurance_support_missing`、`evidence_budget_exceeded` 和
`knowledge_needs_review`。insufficient result 是带有这些精确 code 之一的结构化 reply。
它没有 Answer Evidence Set、recommendation、citation identity 或 provider-visible
evidence payload，也不会调用 provider。

sufficient result 会在 answer generation 之前冻结一个不可变的 Answer Evidence Set。
它至多包含三个选中的 item，每个 item 至多 1200 个字符，总计至多 3000 个字符；若满足
要求的 evidence 超过 cap，decider 会拒绝，而不是静默丢弃 required evidence。被截断的
candidate preview 不能作为完整 evidence 冻结：authoritative source content
length 也必须满足 per-item cap。每一个 item 的 identity 绑定精确的 entry、editorial
revision、Published Knowledge Version、section、content-hashed chunk、source content
length 与 Evidence Excerpt Snapshot。持久化 item 包含该 canonical identity binding，
reader 在信任 item、snapshot 或 citation 前必须重新计算它。set identity 绑定 QCS、
有序的 item identity 和 governing item。每个 citation identity 都绑定该 set identity
以及恰好一个 selected item。score 与 selection diagnostic 不属于 citation identity
输入，也不会进入 provider-visible 或 user-facing citation data。

provider-visible prompt 只能从同一个冻结 set 派生。它的结构化 region 会分离
normalized question、QCS、selected evidence source 和 response contract。response
contract 只能列出 selected citation 和 governing citation。每个 provider-visible source
都携带来自同一 frozen set 的 `snapshot_id`、`item_identity`
和 `citation_identity`，但 score、chunk locator 与 selection diagnostic 仍被排除。
retrieved text 不能修改 policy、permission、provider routing、QCS condition、evidence
identity 或 citation identity。generated output 只能引用冻结的 selected item；每一个
required response section 中每条 material nonblank line 都必须引用，且不得引入 unknown 或
矛盾的 QCS assignment、secret value、unsupported quantified assurance，或将 bounded
internal case universalize。

对于活跃 Pilot 的 production decision，历史的 first-three selector、non-empty-context
gate 和 candidate-derived citation projection 已被替代。它们只能保留在显式的
`lexical_heuristic_migration` / `legacy_migration_diagnostic` profile 后面，且不能产生
product sufficiency decision、immutable Answer Evidence Set 或 product citation
identity。

## Pilot 身份权威与审计

当前的 `users` 行是授权事实：受保护的 handler 必须要求成员仍存在且 active，
并读取其当前数据库 role。JWT role claim、浏览器存储和客户端提交的 member 或
owner 值都不是授权依据。Bootstrap Administrator 仅来自服务端保存的部署配置。
公开注册必须恰好一次性消费 active invitation，且永远创建 Knowledge User；
提升为 System Administrator 只能由已有的 System Administrator 执行。若配置的
bootstrap 创建或其 audit 无法持久化，应用启动会失败，而不会在缺少该身份不变量
时继续服务。条件化 claim 发生竞争时会在 rollback 后重新读取 invitation，使其
持久化拒绝结果按最终状态保留为 `replayed`、`revoked` 或 `expired`。

不可变的 `member` 与 `team_invitation` canonical record 建立安全的身份引用。
未知 invitation code 的拒绝使用独立的、基于 hash 的 `admission_attempt` identity；
已知 invitation 的拒绝仍归入该 invitation 的 canonical aggregate。只追加的
`identity_audit/v1` event 仅包含 action、outcome、reason、时间以及最小
actor/target/reference identity。invitation 和 session 引用为 SHA-256 引用，
而不是 invitation 明文或 bearer token。审计事件绝不包含密码、JWT、request
body、问题、回答或私有 conversation history。仅管理员可查询的身份审计投影只
暴露这个最小事件形状，不会形成管理员读取成员 conversation history 的入口。

`POST /auth/logout` 只删除已认证 bearer 对应的
`auth:session:{sub}:{jti}` key。deactivation 更强：它将成员标记为 inactive，
撤销该成员的全部 session key、拒绝未来登录和受保护访问，但保留账户及其私有
history。每个动作都会先提交无内容的 `pending` audit event；随后用 `revoked`
或 `deactivated` event 记录完成，session store 失败则追加 `failed`，而不会丢失
已经持久化的审计轨迹。对已经 inactive 的成员重试时会追加终态完成或另一个失败，
因而最新的只追加轨迹不会错误地结束在历史失败。

## 兼容性

`compatibility_read_projection` 将现有记录确定性地映射为可替换投影。它只读，
携带 `legacy_id` 和 `unknown_fields`。新调用方应优先写入规范化记录，并可在
expand-contract 迁移期间继续写旧投影。

Ticket 14 的尾部 migration 将所有以前未 revoke、未 expired 的 legacy invitation
视为已消费。历史可重复使用 invitation 的 schema 无法证明某个 code 未被使用，
因此这个保守转换 fail closed；任何尚待准入的成员应由管理员签发新的 invitation。

## 迁移计划

| 后续 ticket | 旧调用方 | 规范化替代 | 移除条件 |
| --- | --- | --- | --- |
| 14 | 用户、invitation 和 Redis session 的准入/授权路径 | member/invitation 标识和不含内容的身份审计事件 | 每条 pilot 身份路径都使用一次性 invitation、数据库派生的 role 检查和只追加身份审计 |
| 16 | Markdown/front-matter 编写与运行时 document 副本 | Private Editorial Repository 的 entry、revision、source 与确定性 `editorial_export/v1` 权威 | T02 只消费经 review 的不可变 export，且 legacy/runtime 行上不再存在权威性的 editorial 写入 |
| 17 | 上传和批量构建分发 | Reviewed Release Bundle、bundle item、build generation、可恢复 Candidate Build | Reviewed bundle 是唯一新增的 authority-bearing intake；Candidate work 没有 publication side effect，legacy publication path 仍仅为 compatibility |
| 18 | 未经资格校验的 legacy retrieval 与 Candidate-derived chunk | 有版本的 Pilot Sparse BM25 与授权的当前 Published Candidate Pool | 普通 retrieval 只返回当前、已授权的 compatibility-published chunk；Candidate preview 保持仅管理员可用且仅用于 diagnostic |
| 19 | first-three selection、non-empty context gate 与 candidate-derived citation | 确定性的 evidence sufficiency 与不可变 Answer Evidence Set | 活跃 Pilot 只使用 authorized pool、精确 QCS 和 assurance rule、一个冻结 selected set 及其绑定的 citation identity |
| 20 | `ChatMessage.rag_trace` 与回答推断 | Answer execution、条件、证据集、快照 | 每个回答都持久化封闭的规范化结果 |
| 21 | HTTP、SSE 和 history 适配器 | 规范化 execution 投影 | 所有界面都读取同一规范化 execution |
| 24 | Candidate 检查与发布 | Candidate 和 Published Knowledge Version | 发布检查规范化代次、hash 和验收标识 |
| 25 | tombstone 与脱敏 | 撤回事件和保留的发布标识 | 所有撤回读写使用规范化发布标识 |
| 27 | 反馈与 review work item | Maintenance item 和 Validated Finding | 原始反馈引用过期后仍保留规范化决策 |
| 15 | 验收脚本和证据 | Delivery Acceptance Record 与状态事件 | 验收绑定精确规范化标识，不使用 branch 或 `latest` |

只有在指定后续 ticket 完成认证产品路径迁移、回放/回填检查，并且在声明的保留
窗口内不再观察到仅依赖旧模型的写入后，才能移除兼容路径。
