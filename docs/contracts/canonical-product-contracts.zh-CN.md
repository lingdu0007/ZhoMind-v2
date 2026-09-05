# 规范化产品契约

状态：规范性产品契约；ticket 13 至 17 的增量基础

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
| 17 | 上传和批量构建分发 | Reviewed Release Bundle、bundle item、build generation | Reviewed bundle 成为唯一常规入口且旧 Job 都已终态投影 |
| 20 | `ChatMessage.rag_trace` 与回答推断 | Answer execution、条件、证据集、快照 | 每个回答都持久化封闭的规范化结果 |
| 21 | HTTP、SSE 和 history 适配器 | 规范化 execution 投影 | 所有界面都读取同一规范化 execution |
| 24 | Candidate 检查与发布 | Candidate 和 Published Knowledge Version | 发布检查规范化代次、hash 和验收标识 |
| 25 | tombstone 与脱敏 | 撤回事件和保留的发布标识 | 所有撤回读写使用规范化发布标识 |
| 27 | 反馈与 review work item | Maintenance item 和 Validated Finding | 原始反馈引用过期后仍保留规范化决策 |
| 15 | 验收脚本和证据 | Delivery Acceptance Record 与状态事件 | 验收绑定精确规范化标识，不使用 branch 或 `latest` |

只有在指定后续 ticket 完成认证产品路径迁移、回放/回填检查，并且在声明的保留
窗口内不再观察到仅依赖旧模型的写入后，才能移除兼容路径。
