# ADR 0001：服务端派生的 Pilot 身份与审计

状态：已接受

日期：2026-09-05

## 背景

Pilot 准入原本允许重复使用 invitation，浏览器或 JWT 元数据也可能被误认为
授权事实。身份生命周期证据还需要一种持久化形式，同时不能暴露凭据或私有
conversation。

## 决策

- Bootstrap Administrator 创建仅使用服务端保存的配置，并由数据库强制“只能有一个”
  的不变量。重复执行保留既有身份和密码。若已配置 bootstrap 却无法持久化，启动会
  中止，而不会允许部分初始化的 pilot 继续运行。
- 公开注册通过原子、带条件的一次性 claim 消费 active invitation，且始终创建
  Knowledge User。
- 每个受保护 handler 都从当前数据库成员派生 active 状态和 role。只有 System
  Administrator 能签发或撤销 invitation、提升、deactivate，以及使用管理员能力。
- 已认证 logout 只删除当前 `(sub, jti)` Redis session。deactivation 撤销成员全部
  session、阻止未来登录，但保留账户和 conversation history。
- 身份生命周期使用 canonical 的不可变 member/invitation record 与只追加、无内容
  的审计 event。未知 code 使用由 SHA-256 派生的 admission-attempt identity；
  已知 invitation 的拒绝保留 invitation aggregate。审计只保存最小 identity 和
  SHA-256 引用，绝不保存密码、bearer token、invitation 明文、request body 或
  conversation 内容。pending audit 会先于 Redis mutation 提交，因此 session
  store 失败仍然可见；重试会追加完成或另一个失败。条件化 claim 竞争会在 rollback
  后重新读取 invitation，再记录最终拒绝结果。

## 后果

尾部 migration 增加 invitation consumption 字段和可跨方言使用的 bootstrap
partial unique index。由于此前可重复使用 invitation 的 schema 无法证明 code
未被使用，它会将 legacy 中未 revoke、未 expired 的 invitation 标为 consumed。
部署必须先运行该 migration，才能依赖新的准入与 bootstrap 保证；尚待准入的成员
需要由管理员重新签发 invitation。
