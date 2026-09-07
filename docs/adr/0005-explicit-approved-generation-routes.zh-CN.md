# 显式批准的生成路由

Status: accepted for Ticket 22 implementation; Pilot activation still requires
controlled live evidence.

我们以一条显式批准、版本化的生成路由取代永久单供应商选择：一个主供应商和最多三个有序、
分别批准的 fallback。这样可以进行有界恢复，但不会把可用凭据、环境别名或 provider
registry 当作发送团队数据的许可。

## 权限与替换

不可变 `provider_route` record 绑定供应商与模型身份、HTTPS 端点及声明的端点类别、
`team_shared_pilot` 数据范围、供应商专属超时、总预算、最大尝试数，以及每个供应商的
`configuration` 批准身份。凭据仅以 Fernet 密文保存在不透明引用之后；凭据和引用都不会
返回。每次保存替换版本都需要显式提交凭据。

管理员保存非活动草稿，提供当前 Active 的 Delivery Acceptance Record，确保路由和每个
批准身份都绑定已验证的供应商、失败、隐私及 prompt/citation 检查，再请求激活。有界连接
验证只使用固定的非内容 prompt。最终事务重新检查当前管理员、证据、最新草稿和预期活动
指针，再追加激活事件并原子切换指针。失败或过期的激活保留旧路由。新 execution 捕获
持久化权限；已经捕获的 execution 保留此前不可变路由。验收证据被撤销或改变后，新请求
停止捕获该路由，查询也不再把它投影为 Active。

服务端 `GENERATION_DEPLOYMENT_IDENTITY`、确切的 `GENERATION_PRODUCT_REVISION`
（`product_revision:<40 位十六进制 commit>`）及 `GENERATION_VALIDATION_MODE`
必须与验收记录匹配。默认 `controlled_live` 模式拒绝 Local Development record，
要求 Editorial Preview 或更高阶段记录显式声明该模式。部署或 revision 配置为空时
拒绝激活。`local_development` 只接受 Local Development record，并在生产工厂中
禁止构建真实供应商；隔离测试必须显式安装 provider I/O 替身。浏览器不能改变这个
服务端边界。

## 尝试与结果

只有连接失败、超时、限流、临时服务错误、脱敏服务错误、确定性答案结构失败及 citation
失败可以推进路由。每个供应商最多尝试一次，同时受声明的尝试数和剩余时间预算限制。
SDK 重试和 HTTP 重定向被关闭；单次尝试的 HTTP client 在完成或取消后关闭。

证据不足不会调用供应商。取消、权限、数据范围、安全、政策或未声明供应商的决策都不会
推进路由。每次尝试接收完全相同的冻结 question/QCS/evidence payload 和有序 snapshot
hash；输出验证不会重新检索或选择证据。

规范化路由失败或耗尽返回 executor 的 completed Generation Unavailable outcome，
没有 Decision Summary 或 Knowledge Answer Citation，最多展示已有的冻结 Generation
Evidence Preview。未分类的 application failure、格式错误的 provider input，以及格式
错误或不匹配的 observed envelope 仍是 application failure，绝不能成为 insufficiency
或成功的 fallback。非内容的路由与批准身份、尝试序号、规范化原因、耗时、payload hash
和 snapshot hash 保留在既有的 30 天 Operational Event 边界内。原始异常、问题、答案
和证据文本均被排除。
请求级尝试观察独立于成功答案投影发出，包括取消和 application failure。输出隐私或
政策拒绝不能被当成答案结构重试原因。

## 兼容与证据

本决策替代 workspace 历史 ADR `0010-use-one-approved-generation-provider` 和
`0010-fail-closed-on-generation-provider-outage` 的永久单供应商限制，并替代 source
ADR 0004 中“每个未恢复的 provider exception 或没有完成的 provider call 都是
application failure”的规则。ADR 0003/0004 的冻结证据、隐私、传输及格式错误输入规则
保持不变。

旧设置仅保留为兼容配置界面，不具备生成批准权限；通用生成供应商发现被移除。迁移不会
从已有凭据创建批准路由。

确定性 fixture 只激活 Local Development 证据，并显式替换 provider I/O。它不能证明
真实供应商行为，也不能批准 Pilot 使用。Pilot 之前，owner 必须为确切声明的供应商、
模型、端点及数据边界安排受控 live smoke，并保留失败、隐私、prompt/citation 和激活
证据。环境不可用或缺少批准仍属于未完成验收义务。
