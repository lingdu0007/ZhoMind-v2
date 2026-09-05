# 规范化产品契约

状态：规范性产品契约；ticket 13 与 14 的增量基础

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
`evidence_snapshot`、`maintenance_item` 和 `delivery_acceptance_record`。
标识不可变。标题、文件名、代次编号、分数或当前 `latest` 指针都不是标识。

## 状态与事件

Python 契约模块定义 PRD 中关于条目生命周期、bundle 接收、构建阶段与终态、
回答执行与结果、维护和验收的状态词汇。`validate_transition` 在持久化前拒绝
非法转换。`canonical_records` 是按 `immutable`、`append_only`、
`authoritative`、`derived` 或 `replaceable_projection` 分类的不可变快照。
`canonical_events` 只追加，并通过追加状态变更记录而不是改写历史。

当旧记录无法证明时，任何投影都不会把 `answer_eligible`、编辑审批、来源授权、
发布资格或验收默认为 true。旧记录缺少的身份、assurance、来源、bundle、条件
和证据字段会明确保留为 `unknown`。

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
| 17 | 上传和批量构建分发 | Reviewed Release Bundle、bundle item、build generation | Reviewed bundle 成为唯一常规入口且旧 Job 都已终态投影 |
| 20 | `ChatMessage.rag_trace` 与回答推断 | Answer execution、条件、证据集、快照 | 每个回答都持久化封闭的规范化结果 |
| 21 | HTTP、SSE 和 history 适配器 | 规范化 execution 投影 | 所有界面都读取同一规范化 execution |
| 24 | Candidate 检查与发布 | Candidate 和 Published Knowledge Version | 发布检查规范化代次、hash 和验收标识 |
| 25 | tombstone 与脱敏 | 撤回事件和保留的发布标识 | 所有撤回读写使用规范化发布标识 |
| 27 | 反馈与 review work item | Maintenance item 和 Validated Finding | 原始反馈引用过期后仍保留规范化决策 |
| 15 | 验收脚本和证据 | Delivery Acceptance Record 与状态事件 | 验收绑定精确规范化标识，不使用 branch 或 `latest` |

只有在指定后续 ticket 完成认证产品路径迁移、回放/回填检查，并且在声明的保留
窗口内不再观察到仅依赖旧模型的写入后，才能移除兼容路径。
