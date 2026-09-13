# Ticket 执行协议

状态：实现类 ticket 的必需工程工作流。

本协议把一张 ticket 转化为一组可审阅的工作归属、行为证据与最终 diff。
`AGENTS.md` 保存硬规则，本文定义详细执行顺序。Ticket 及其父级产品文档仍是范围与
行为的权威来源。

## 1. 预检与归属

从仓库根目录工作。修改代码前，阅读 `AGENTS.md`、被分配的 ticket、ticket 指向的
全部父 PRD 要求，以及适用的 contract 与 ADR 段落。

确认 tracker 中所有编号依赖均为 resolved；当 ticket 指定了依赖实现 commit 时，还要
确认当前源码历史包含这些 accepted commits。不能以 status label 代替源码历史证明。

记录：

```bash
BASE_SHA="$(git rev-parse HEAD)"
git branch --show-current
git status --short
```

初始 status 是预先存在的工作区清单。保留这些改动，不暂存无关路径，不使用破坏性
Git 命令。

Ticket 的 `## Comments` 必须记录活动 owner 或 session、`BASE_SHA`、branch 与开始
时间。如果 ticket 已是 claimed，且归属无法与当前任务准确对应，停止并且不要编辑。

只运行足以证明受影响测试运行器可以启动的最小环境与 baseline smoke。单独记录
pre-existing failures；除非 ticket 明确拥有它们，否则不要修复或悄悄吸收。

## 2. 验收项到 Seam 的映射

写测试前，为每一项验收标准建立映射：

| 验收标准 | 已批准的公共 seam | 第一个失败测试 | 必需负向场景 | 最终证据 |
| --- | --- | --- | --- | --- |
| Ticket 原文 | 用户或调用方可见 interface | 测试路径与名称 | 拒绝、失败或边界行为 | 命令与结果 |

测试通过公共 interface 观察行为，不测试私有方法、可变 diagnostic，或仅通过实现内部
数据库查询证明行为。

仓库 owner 已预先批准以下默认 seams：

- 用于确定性 state、identity、normalization、policy 与 validation 行为的纯 domain 或
  contract interface。
- 用于服务端派生 authority、ownership、persistence 与产品 outcome 的 authenticated
  HTTP interface。
- 用于 streaming progress、interruption 与最终结果 projection 的 SSE terminal-event
  interface。
- 用于 retained identity 与 redaction 的 persisted reload 和 history interface。
- 用于无语义重解释 canonical projection 的 frontend presentation adapter 或 store
  interface。
- 用于用户可见 workflow、authorization、failure 与 responsive layout 的
  disposable-API browser journey。
- Ticket 修改 persistence 时，用于 migration、compatibility 与 database enforcement
  的 Alembic upgrade interface。

使用其中最小的适用集合。Agent 无需再次请求用户批准这些 seams。只有 ticket 确实需要
列表外的新 seam，或两个已列 seam 会断言互相矛盾的产品行为时，才需要询问用户。

优先使用能够证明完整行为的最小 seam 集合。对于跨 surface contract，使用共享
contract vectors 验证每个必需 adapter，不要让每种语言或 transport 各自重新生成预期
语义。

## 3. Review 风险矩阵

对于受保护、持久化、异步或跨 surface 的变更，在实现前回答每个适用问题，并把答案
转化为测试：

| 风险 | 必须回答的问题 |
| --- | --- |
| Authority | 哪个当前持久化事实授权这次动作或 projection？ |
| Identity | 哪些不可变 identity、hash、version、attempt 与 configuration 必须一致？ |
| Fail closed | 必需事实缺失、过期、畸形、未授权或矛盾时会发生什么？ |
| State | 哪些 transition 合法、终结、可重试或禁止？ |
| Race 与 retry | 并发变更、取消、重复投递、lease 丢失或重试后会发生什么？ |
| Compatibility | legacy 或部分迁移数据能否错误授予当前 authority？ |
| Privacy | content、credentials、其他成员数据或管理员 diagnostic 能否越过 seam？ |
| Projection | normal、streaming、persisted、reload、history 与 frontend 是否保留同一语义结果？ |
| Scope | 实现是否避开 publication、provider、retention、recovery 或后续 ticket 行为？ |

只有 happy path 不能关闭矩阵中的风险。风险只能通过已批准 seam 上的确定性断言关闭，
或被准确分类为明确的 deployment-only obligation。

### 自动风险配置

当 ticket requirements 或 changed surfaces 匹配时，自动应用以下配置。Worker prompt
不需要重复这些检查。

**Canonical authority 或 persistence**

- 从当前持久化事实派生 authority，不能信任 client flags、可变 diagnostics、title、
  filename、environment alias 或 `latest`。
- 当 identity 影响授权时，绑定准确 aggregate identity、revision、version、hash、
  attempt、actor、configuration 与 status event。
- 拒绝 stale、malformed、foreign、partially migrated 或 contradictory facts。
- 覆盖 duplicate submission、retry、cancellation、concurrent change、partial
  transaction failure 与 finalization race。
- 使用新增 Alembic revision，并测试受支持 upgrade path。

**Retrieval、evidence、answer execution、chat 或 feedback**

- 保留 closed Answer Execution Outcome，并将 execution states 与 completed outcomes
  分开。
- Adapter 不得重新 retrieval、rerank、选择 evidence、切 snapshot，也不能根据 text、
  source count、flags 或 legacy diagnostics 推断 outcome。
- Normal HTTP、SSE、persistence、reload、history 与 feedback 必须保留相同 question
  conditions、outcome、evidence-set identity、snapshot identity、citation identity 与
  knowledge version。
- 防止 stale 或 interrupted stream 污染其他 conversation 或后续 execution。
- Insufficient Evidence 不调用 provider，不包含支持性 citation 或生成 recommendation。
- Generation Unavailable 只能暴露已经冻结的 evidence preview，不能展示 Decision
  Summary 或 Knowledge Answer Citation。
- 排除 Candidate、draft、withdrawn excerpt、inaccessible source、administrator-only
  diagnostics 与其他成员私有内容。
- 对 normalization、Unicode、punctuation、reason codes、retry limits 与 presentation
  semantics 使用共享 Python/JavaScript contract vectors。

**Frontend 或 browser workflow**

- 显式表现 loading、empty、failure、stopped、throttled、rejected 与 completed states，
  不能制造成功内容。
- 等待可观察 readiness，不能只等待 heading 出现或使用任意 sleep。
- 验证 desktop 与 narrow-screen layout，避免 overlap、clipped controls、无法访问的
  action 与由状态引起的 layout shift。
- 在每个 setup 或 launch failure window 之前注册 browser、server 与 build-process
  cleanup。
- 使用完整 authenticated browser journey 验证产品流程，不能把孤立 store、API call
  或 diagnostic page 当成产品验收。

**Publication、recovery、queue 或 background work**

- 分离 admission、build completion、inspection、acceptance、publication、withdrawal
  与 recovery authority。
- Failed replacement、partial verification、worker interruption、lease loss 或
  derived-data cleanup failure 后保留此前 accepted state。
- 每个 failed 或 interrupted operation 都必须有 structured terminal 或 retryable
  state，以及 allowed next action。
- 不可逆 finalization 前重新验证 immutable inputs 与 current authority。

**Privacy、safety 或 operations**

- 通过 authenticated interfaces 证明 cross-member 与 administrator denial。
- Diagnostics、logs、audit 与 public evidence 不得包含 question、answer、excerpt、
  credential、token、private description 或 full request body。
- 分别验证 deletion、expiry、redaction 与保留的 de-identified identity。
- 将 content-bearing telemetry 或 unauthorized projection 作为 blocking integrity
  defect。

## 4. 纵向 TDD Slice

每次实现一个 tracer slice：

1. 在已批准 seam 上新增一个失败的行为测试。
2. 确认它因为预期缺失行为而失败。
3. 添加让它通过所需的最小实现。
4. 运行受影响的 focused tests。
5. 根据该 slice 暴露的信息更新 acceptance mapping 与风险矩阵。
6. 继续下一项验收标准或负向场景。

不要先写完全部测试再一次性实现。不要把宽泛重构放进 red-green loop。保持 adapters
轻薄，让 canonical authority module 继续作为唯一语义来源。

Ticket 跨越多层时使用以下 checkpoints：

- Contract checkpoint：identity、authority、state、failure 与 compatibility。
- Product-path checkpoint：authenticated interface 与必需负向场景。
- Projection checkpoint：transport、persistence、history 与 frontend 渲染同一个
  retained result。

工作仍新鲜时就把 checkpoint 命令与结果写入 ticket。

## 5. 验证梯度

验证从窄到宽执行：

1. Environment 与 baseline smoke。
2. 当前 slice 的 focused unit 或 contract tests。
3. 受影响 integration tests。
4. Authenticated product-path 与 browser journeys，包括负向场景。
5. 受影响 surface 的 static checks 与 production build。
6. `README.md` 中适用的 deterministic repository gate。
7. 单独委托的 deployment-only checks。

早期 red-green 循环中不要反复运行完整 repository gate。Focused 与 product-path 已经
收敛后运行；之后只有可能影响完整门禁的修复才需要再次运行。

真实 embedding、Milvus、provider、PostgreSQL deployment、ingress、Compose、
persistent stack、load 与 production acceptance 都是 deployment-only，除非 ticket
明确委托。Skipped 或 unavailable check 表示限制，不是 pass。

## 6. 临时提交与双轨 Review

Focused、integration 与 product-path checks 通过后：

1. 只暂存 ticket 拥有的路径。
2. 创建一个 provisional focused commit。
3. 确认 `git diff BASE_SHA...HEAD` 只包含预期 ticket diff。
4. 针对 `BASE_SHA...HEAD` 使用仓库 code-review 工作流，同时运行 standards reviewer
   与 specification reviewer。
5. 将每条 finding 分类为 blocking、可执行的 non-blocking，或附带理由的 rejected。
6. 每个行为 finding 都先增加一个能在被审 commit 上失败的 regression，再修复行为。
7. 重跑受影响 focused 与 product-path checks，并 amend provisional commit。
8. 对 amend 后的 diff 重新获得 standards 与 specification review。

不能用 generic reviewer 替代缺失的 specialist reviewer。必需 reviewer 不可用时，
保持 ticket 非终态，或按照 tracker 规则使用 blocked status 并记录准确的必需动作。

Review 收敛后运行最终 deterministic gate。如果门禁要求修改代码或行为测试，amend
后必须重新运行两个 review axes。最终 `HEAD` 必须正是没有 blocking finding 的已审
diff。

## 7. Ticket 关闭

Ticket resolved 前：

- 只有对应映射证据存在时才勾选验收标准。
- 记录准确验证命令与结果。
- 记录 review findings、regression tests、修复与新一轮 review 结果。
- 记录最终 commit，并确认已审 diff 仍与 `HEAD` 一致。
- 分开记录本地 deterministic evidence 与 deployment-only limits。
- 记录 residual risks，但不能将其表述为已接受行为。
- 确认没有暂存或提交无关路径。

验收标准只从代码推断、必需 reviewer 不可用、仍有 blocking finding、最终 diff 未重新
review，或把 remote-only check 表述为本地 pass 时，ticket 不能 resolved。

## 8. Worker Prompt 结构

普通 worker prompt 应当只有一个短段落。Ticket 的 dependency、requirement coverage、
acceptance criteria 与 exclusions 从 tracker 读取；默认 seams、风险配置、TDD loop、
验证梯度、review 顺序、输出语言与关闭规则来自 `AGENTS.md` 和本协议。

使用以下默认结构：

```text
在 /home/lingdu/workspace/agent/Zhomind/sources/ZhoMind-v2 实施 Ticket <NN>。
读取并严格遵循仓库 AGENTS.md、docs/agents/ticket-execution.md、对应 ticket
及其引用的 PRD、contract 与 ADR。确认依赖和归属后，按纵向 TDD 完成全部验收项，
执行适用的产品路径与确定性验证，完成 standards/spec 双轨 review 和 re-review，
更新 ticket 证据，并提交仅包含本票的聚焦变更。全程用中文汇报，不扩大 ticket 范围。
```

只有信息尚未存在于 ticket 或仓库规范中时才补充 prompt，例如用户提供的 dependency
commit、临时外部环境、新批准的 seam 或显式 scope override。不要在每票 prompt 中复制
长期产品 contract、风险清单或完整 deterministic 命令列表。
