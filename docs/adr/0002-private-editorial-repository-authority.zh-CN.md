# ADR 0002：Private Editorial Repository 权威

状态：已接受

日期：2026-09-05

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

## 后果

T01 现在具备耐久、access-controlled 的 editorial authority，且可从保留数据证明
revision/export hash。T02 必须经由自己的不可变 Reviewed Release Bundle intake
contract 消费该 export；T03 Candidate Build 和 T04 publication 仍是独立责任。
source availability 已是 fail-closed eligibility 的权威事实；后续 publication 与
maintenance path 必须消费这项保留 evidence，而不是从 runtime copy 推断。
Release-Assured reference 只有在 canonical record 属于适当的 authority record，且 frozen
delivery-acceptance record 正在 active 地覆盖精确 editorial authority 时才可通过，
否则 fail closed。
