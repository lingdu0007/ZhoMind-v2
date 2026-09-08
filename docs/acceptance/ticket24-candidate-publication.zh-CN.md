# Ticket 24 Candidate 发布验收

状态：Ticket 24 全部十五项条件已验证，范围内实现完成
日期：2026-09-08
实现提交：`59064b77ff8dd6adef09823604c15799c4e1bf94`
固定基线：`4c8a5437ec1bded09df89435dd13b28cc7925e0d`
分支：`ticket24-candidate-publication`

## 范围

本记录覆盖工作区 ticket
`.scratch/usable-team-knowledge-base/issues/24-inspect-accept-publish-candidates.md`
的全部十五项验收条件，即 KB-RL-004、KB-RL-005 和 KB-AC-004 的 Candidate 部分。
它包含 `4bcad79` 的已有实现，以及 `b189f85`、`59064b7` 的接续工作。
本记录不声称完成 Pilot Entry Baseline、真实 provider 激活、生产部署、撤回实现或
Portfolio Release 验收。

编辑批准、Candidate inspection、经过认证的 Candidate acceptance 和显式发布仍是
分离的操作。Candidate acceptance 执行确定性的 closed evidence contract，不调用
generation provider。临时浏览器 API 另行使用确定性测试 provider，验证已发布内容
经过认证的 chat 回答路径。

## 验收映射

主要 HTTP 回归文件为
`backend/tests/integration/test_ticket24_candidate_publication_api.py`；
浏览器旅程为 `frontend/tests/reviewed-bundles-workspace.test.mjs`。
下表列出代表性测试，完整套件还包含其他负向用例。

| 验收条件 | 保留的验证 |
| --- | --- |
| 授权查看 metadata、chunks、bundle/revision/hash/generation/configuration | `test_inspection_is_durable_and_exactly_bound`、角色拒绝矩阵和真实浏览器 metadata inspection |
| 聚焦的替换比较 | `test_replacement_failure_is_isolated_and_preserves_the_existing_pointer`、`test_multi_source_runtime_projection_does_not_invent_replacement_diff_changes` |
| 持久且精确绑定的 inspection | `test_publication_rejects_corrupted_retained_inspection_or_acceptance`，覆盖外来 identity 和 JSON 布尔／数值替换 |
| 对精确 Candidate 执行 Supported 与 Boundary Query | `test_candidate_acceptance_requires_exact_supported_and_boundary_records` 和下述真实 Release-Assured 链路 |
| 保留 governing entry/section、Evidence Set、citations、snapshots | Supported 结果断言、完整确定性结果校验、多来源 Claim-Linked 测试和可重载的浏览器验收记录 |
| Boundary 为结构化不足，无 provider 调用或 citation | Boundary 结果断言，以及 QCS identity、provider count 篡改用例 |
| diagnostic、旧输入或变化的 configuration 不能替代验收 | 冻结输入／chunk metadata 检查、configuration 变化、旧 generation 和选择不匹配测试 |
| 只有批准、inspection、acceptance 和当前精确身份均有效才能选择 | `test_publication_requires_exact_eligible_confirmation_and_creates_a_version` 和留存记录篡改矩阵 |
| 确认逐项列出 Candidate 与创建／替换效果 | HTTP 精确选择断言和浏览器确认旅程 |
| 分开报告 published/failed/skipped，不虚报完整批次成功 | 混合批次 HTTP 断言和浏览器部分发布提示 |
| 每个选中 entry 原子发布 | pointer 切换前注入持久化失败、独立成功兄弟项和持久化发布重载 |
| 替换成功前保留旧 eligible version | 替换失败与未发布后继修订检索测试，包括新增未验证来源的真实后继 revision |
| 失败和重复确认保留唯一 pointer，且可重试 | 新 confirmation 重试成功；重复 confirmation 重放原结果；租约执行和中断结果恢复测试 |
| Candidate 和未选择内容不进入普通检索 | 普通池发布测试、pointer 篡改测试和浏览器角色边界 |
| 为后续 acceptance 与 reconstruction 保留精确身份 | `test_delivery_acceptance_persists_an_exact_candidate_publication_binding` 和不可变 publication/inspection/acceptance 重载 |

`backend/tests/unit/test_editorial_authority.py` 还执行真实的 Release-Assured
编辑批准／导出、bundle intake、Candidate build、inspection、acceptance 和 publication。
该测试证明 snapshot 字节和 export hash 保持不变，且未发布后继修订新增的未验证来源
不会撤销仍然有效的 pointed version。

## 验证结果

各命令分别在 `backend/`、`frontend/` 或仓库根目录执行。

| 命令或检查 | 结果 |
| --- | --- |
| `uv run ruff check .` | 通过 |
| `uv run pyright` | 0 errors、0 warnings |
| 在实现提交上执行 `uv run pytest -q -ra --tb=short --disable-warnings` | 833 passed、1 skipped |
| `npm test` 最终实现复跑 | 通过：45 项单元测试和 123 项浏览器／门禁测试（基础 108、reviewed-bundles 9、settings 6）；生产构建通过 |
| `uv run alembic heads` | 唯一 head 为 `20260908_t24_candidate_pub`；后端套件包含迁移回归 |
| `python3 scripts/check-docs-parity.py` | 通过，包含本双语验收记录 |
| `python3 scripts/validate-evidence-bundle.py --all` | 七个保留的 bundle 全部通过 |
| `python3 scripts/scan-secrets.py` | 通过 |
| Playwright 1440、1024、390 像素视口 | 截图非空；文档宽度等于视口宽度；移动端保留已声明的桌面专用管理边界 |

首轮后端全量测试与浏览器工作并发执行，结果为 822 passed、1 failed、1 skipped。
唯一失败是使用 50ms 租约的 `test_long_running_indexing_renews_the_candidate_worker_lease`。
该用例单独重跑及最终后端全量执行均通过；没有为获得通过结果而放宽租约、心跳或生产
时序行为。既有依赖弃用警告和前端 bundle 大小警告仍明确保留。

## Standards

规定的 `standards_reviewer` 审查完整基线差异，并复审至 `59064b7`。
三个规范问题和一个 possible Duplicated Code 建议均已解决：

- 历史 decisive source loss 不再被后继 revision 的 approval 截断。
- 普通检索完整验证 pointer 的 entry、document 和 generation。
- 留存 inspection/configuration 校验区分 JSON 布尔值与数字。
- current 与 historical authority 共用来源校验和 section projection。

最终 Standards 复审：无剩余可执行问题。

## Spec

规定的 `spec_reviewer` 审查同一完整差异及其修复。发现的两个 P1 均已解决：

- 合法 Release-Assured export 可以使用其保留的 canonical event UUID reference，
  而不改写冻结 snapshot。
- 未发布后继 revision 自身的来源问题不会移除 eligible predecessor；
  predecessor 自身的来源失效仍按 fail-closed 处理。

最终 Spec 复审：无剩余可执行问题或范围回归。

## 边界与已知失败

- 未设置 `RUN_MILVUS_E2E=1`，因此明确跳过 live Milvus 测试。本任务没有执行真实
  embedding、generation provider 或持久化生产栈验收。
- 历史命令
  `python3 scripts/verify-portfolio-release.py --expected-source-revision 91753f1c1ff6fc07bc262dfa50fb719a63210e0b --expected-release-revision HEAD`
  仍为**失败**：Public Evidence Bundle source 不是 release artifact revision 的祖先。
  在 Ticket 24 开始前的固定基线上，该祖先检查同样失败。本任务没有改写证据，也没有
  声称 Portfolio Release 通过或获得豁免。
- 本地预览使用独立的临时 SQLite 数据库和合成编辑材料，包含一个有意注入的项目失败。
  它不是生产验收，也不是永久服务。
