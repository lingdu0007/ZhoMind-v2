# Public Evidence Bundle 契约

版本：**1.1.0**（schema `evidence-bundle.schema.json`，sections schema 位于 `sections/`）

## 目的

**Public Evidence Bundle** 是与 ZhoMind-v2 Portfolio Release 一起发布的、经过审查的非敏感证据子集。它将每一条公开声明与 source revision、run identities 和 corpus/query hashes 关联起来，同时不暴露凭据、原始环境值、私有问题、完整模型回答、来源摘录、主机地址或运维凭据。

本契约是唯一的公开证据格式。未来的证据生产者（检索、确定性回答、Prompt Injection、性能、生产验收）必须把数据写入本契约定义的 sections；不得另造公开证据格式。确定性 validator `scripts/validate-evidence-bundle.py` 在 PR Gate 中强制实施本契约。

## Bundle 布局

一个 bundle 是一个目录，包含 manifest、类型化 section 文件和双语报告对：

```
<bundle-dir>/
  manifest.json                     # 必需：allowlist manifest（schema 1.1.0）
  sections/
    retrieval.json                  # 可选类型化 section
    answer.json                     # 可选类型化 section
    prompt-injection.json           # 可选类型化 section
    performance-*.json              # 可选类型化 sections；已接受 candidate 使用 c1 与 c5 profiles
    production-acceptance.json      # 可选类型化 section
  REPORT.md                         # 必需：English canonical 报告
  REPORT.zh-CN.md                   # 必需：完整中文镜像
```

一个 bundle 至少包含一个类型化 section。section 文件和报告必须在 `manifest.artifacts` 中声明并附带 sha256；validator 会重新计算每个哈希。

## Manifest 字段（显式 allowlist）

`manifest.json` 是一个对象，顶层字段严格限定为以下字段（`additionalProperties: false`）：

| 字段 | 类型 | 规则 |
| --- | --- | --- |
| `schema_version` | string | legacy bundle 使用 `1.0.0`；受控三模式检索对比必须使用 `1.1.0`。 |
| `bundle_id` | string | `[a-z0-9][a-z0-9-]{0,127}`，在同一次调用（`--all` 或显式目录）检查的 bundle 间唯一。 |
| `kind` | string | 必须等于 `public-evidence-bundle`。 |
| `canonical_language` | string | 必须等于 `en`。 |
| `mirror_language` | string | 必须等于 `zh-CN`。 |
| `source_revision` | string | bundle 描述的完整 40 位十六进制 source commit id。 |
| `release_candidate` | object | `identity`、`revision`（必须等于 `source_revision`）、`status`（`candidate`/`accepted`/`released`）、`created_at`（ISO 8601 UTC）。 |
| `provenance` | object | `runs`、`corpora`、`query_sets`、`revisions`（见下文）。 |
| `artifacts` | array | 每个 bundle 文件的清单：`path`、`kind`、`sha256`、`role`。 |
| `limits` | array | 声明的限制，每条为 `name`/`kind`（`boundary`/`qualification`/`exclusion`/`target`）/`statement`。至少需要一条。 |

`provenance.runs` 记录非敏感 run identities：`run_id`、`kind`（`retrieval-smoke`、`generation-smoke`、`retrieval-evaluation`、`answer-acceptance`、`prompt-injection-run`、`performance-run`、`production-acceptance`）、`source_revision`、`started_at`、`finished_at`、`outcome`（`passed`/`failed`/`completed-with-exceptions`）。

`provenance.corpora` 和 `provenance.query_sets` 记录以 `sha256` 锚定的身份（version + 确定性哈希）。`provenance.revisions` 列出 bundle 中任何位置引用的每个 revision。

## 类型化 Sections

每个 section 是一个 JSON 文件，按各自的 allowlist schema 校验。section 文件现在是可选的；生产者可以随时间增加，但一旦存在就必须符合规范。

| Section 文件 | Schema | 必需内容 |
| --- | --- | --- |
| `sections/retrieval.json` | `retrieval.schema.json` | `run_ids`、`corpus_id`、`query_set_id`、`modes`、`metrics`（Evidence Recall@3/@5/@10、Context Precision@3/@5/@10、first Gold Evidence rank、retrieval duration）、`boundary_query_diagnostics`、`conditions`（chunking、output depths、corpus/query-set versions）。 |
| `sections/answer.json` | `answer.schema.json` | 确定性回答 cases：`case_id`、`outcome`（四个闭值 Answer Execution Outcomes 之一）、`citations_count`、`contract`（`passed`/`failed`）。 |
| `sections/prompt-injection.json` | `prompt-injection.schema.json` | 对抗 cases：`case_id`、`kind`、`outcome`、`pass_fail`、`citation_counts`、`failure_classification`。 |
| `sections/performance-*.json` | `performance.schema.json` | `load`（concurrency、requests）、`metrics`（SSE TTFT 与 total P50/P95/P99、error rate、retrieval/generation-provider/embedding-provider/persistence durations、application-controlled time）、`target`（12 秒 P95 target、`met`、`note`）。`accepted` candidate 必须包含恰好 c1 与 c5 profiles，以及基于观测方差的 regression envelope。 |
| `sections/production-acceptance.json` | `production-acceptance.schema.json` | `path`（必须等于 `/api/v1/chat`）、`results`（有界的逐 check 结果与限制）。 |

每个声明的评估 mode（`modes` 中除 `migration` 之外的所有 mode；`migration` 是单独报告的可用性机制）都必须携带完整的八项指标族。值有单位约束：`ratio` 在 [0, 1]、`rank` >= 1、`ms` >= 0。缺失指标族或越界值会以 `metric-completeness` 诊断失败。只报告 `migration` 可用性的 retrieval section 可以携带空的 `metrics` 数组。

精确三模式对比（`sparse_bm25`、`dense` 与 `hybrid_rrf`，且不含 `migration` mode）属于 `1.1.0` retrieval section。它必须记录 `conditions.candidate_depth = 20`、一个非敏感 `model_identity`，以及每个 mode 一项 `mode_provenance`。每项携带 mode 的 run identity、source revision、corpus hash、query-set hash 和 embedding identity；validator 会拒绝缺失字段或跨 mode drift。

## 敏感字段策略

两层机制确保排除敏感材料。

1. **字段 allowlist。** schema 全面使用 `additionalProperties: false`。会携带敏感内容的字段（`credentials`、`api_key`、`password`、`token`、`secret`、`answer`、`excerpt`、`prompt`、`question`、`environment`、`host`、`address`、`private_content`）不属于契约，因此任何出现都构成 schema 违规并给出具体诊断。

2. **值形状拒绝。** validator 会扫描 manifest、每个 artifact 文件的文本内容（每文件 4 MiB 文本上限；二进制或非 UTF-8 文件无法做文本扫描）和两份报告中的敏感形状，发现即拒绝该 bundle：
   - Provider/API 密钥形状（例如长 `key-prefix-...` 令牌、AWS 风格访问密钥、GitHub 令牌）。
   - 私钥块和 SSH 密钥引用（`id_ed25519`、`id_rsa`、`.ssh/` 路径）。
   - JSON Web Tokens 和 bearer tokens。
   - 内嵌凭据的 URL（`scheme://user:password@host`）。
   - IPv4 地址和 `host:port` 形式。
   - 环境赋值（`KEY=value` 行）和 `${VAR}` 引用。
   - `user@ip-address` SSH 风格登录形式。

敏感值永远不会被回显；诊断只命名形状类别和违规文件。

## 跨制品引用规则

validator 拒绝任何引用了 bundle 自身 provenance 中不存在对象的报告、section 或 artifact：

1. `release_candidate.revision` 必须等于 `source_revision`。
2. `source_revision` 必须出现在 `provenance.revisions` 中。
3. 每个 `run.source_revision` 必须出现在 `provenance.revisions`（或等于 `source_revision`）。
4. `run_id`、`corpus_id` 和 `query_set_id` 在其数组内唯一。
5. 每个 section 的 `run_ids` 条目必须存在于 `provenance.runs`；`corpus_id` 和 `query_set_id` 必须存在于 `provenance.corpora` 和 `provenance.query_sets`。
6. `manifest.json` 自身必须列在 `artifacts` 中（kind `manifest`，role `manifest`）。其 `sha256` 是自引用占位值，validator 不会重新计算。
7. 其他每个 artifact path 必须解析到 bundle 目录内部、存在，且与声明的 sha256 匹配。
8. section artifacts 必须存在，且其 JSON `section` 判别值必须等于 artifact `kind`。
9. `REPORT.md`（role `report-en`）和 `REPORT.zh-CN.md`（role `report-zh`）必须同时存在并被声明。

## 双语 Parity

`REPORT.md` 是 canonical 版本；`REPORT.zh-CN.md` 是在同一变更中更新的完整镜像。validator 强制两条机器可检查规则：

1. **结构 parity。** H1/H2/H3 标题大纲计数相同，fenced code block 计数相同。
2. **字面量 parity。** 标识符、指标名、状态值、模型名和版本以字面量保留。validator 从英文报告中提取字面量集合（snake_case 标识符、`Metric@N` 名称、`x.y.z` 版本、以及固定缩写表，如 TTFT、P50、P95、P99、RRF、BM25、QA、SSE、API、JSON、CLI、PR），并要求每个提取出的字面量都出现在中文报告中。指标名和状态值永远不翻译。

## Validator

```bash
# 验证示例 bundle 和每个 tracked release bundle
python3 scripts/validate-evidence-bundle.py --all

# 验证指定的 bundle 目录
python3 scripts/validate-evidence-bundle.py public-evidence/example

# 列出 --all 会发现的 bundle（不验证）
python3 scripts/validate-evidence-bundle.py --list
```

退出码 `0` 表示每个被检查的 bundle 均符合规范；`1` 表示至少一个检查失败（stdout 上会给出逐 bundle 诊断）；`2` 是用法错误。validator 是纯 Python 3 标准库实现，无需安装依赖，与 `check-docs-parity.py` 和 `scan-secrets.py` 一致。

## 版本化

契约通过每个 schema 内的 `schema_version` 进行版本化。版本 `1.1.0` 在本次变更中加入受控三模式检索对比规则及 validator 支持。既有 `1.0.0` bundle 使用显式 legacy 校验路径；只有三模式对比必须使用 `1.1.0`，因此 legacy evidence 不会被静默重解释为 comparison evidence。

## 术语

领域词汇遵循 `CONTEXT.md`：Evidence Package、Evidence Manifest、Public Evidence Bundle、Portfolio Release、Stable Experimental Baseline、Evaluation Query Set、Boundary Query、Answer Evidence Set、Answer Execution Outcome、Production Answer Acceptance 均保持 canonical 定义。
