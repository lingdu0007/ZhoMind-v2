# RAG 生成生产诊断与烟测总结

**记录日期：**2026-08-01
**已核验的生产修订：**`36f6e1cb57c65130ba5c0efa8d18f0ff625e4643`

本文是带引用 RAG 生成闭环验收的运行复盘和操作手册。只记录配置**状态**，绝不记录 `.env` 值、凭据、提示词、回答或来源摘录。

## 结果

真实生产验收已通过完整的已批准服务商闭环：

1. Bootstrap Administrator 身份验证。
2. 实时 Embedding、Milvus 建索引，以及检索刚发布的来源。
3. 带已发布来源引用的普通聊天响应。
4. 满足相同引用来源契约的 SSE 聊天响应。
5. 经 Caddy 的公网 HTTPS 健康检查和静态前端交付。

最终生产清单中，`retrieval-evidence generation-smoke` 的 `outcome=passed`，其中 `normal_contract` 和 `stream_contract` 均为通过。清单只包含源码修订、非敏感契约结果、数量和归一化失败信息。

## 事件：已批准的 Ark 生成超时

### 用户可见症状

最初的真实服务商生成验收在 45 秒后抛出 `TimeoutError`，因此无法证明普通带引用聊天和 SSE 完成；检索、认证、实时 Embedding 和索引当时已经通过。

### 使用的反馈环

可变红的反馈环是下面的真实服务商 Compose 命令：

```bash
python -m app.retrieval_evidence generation-smoke \
  --base-url http://backend:8000 \
  --output-dir /evidence \
  --source-revision <revision> \
  --run-id <unique-run-id>
```

它由 [`deploy/production/acceptance.sh`](../../deploy/production/acceptance.sh) 调用，且仅在服务、HTTPS、Bootstrap Administrator 和检索检查通过后执行。该命令上传并显式发布专用来源，走部署中的真实检索与已批准服务商路径；精确验收契约失败时以明确的失败检查项非零退出。

为了最小化失败场景，分别运行了直接短服务商请求、服务商适配器和 RAG 图调用，均正常完成。重放包含合成 `retrieval-evidence-<run-id>` 标识的原始组合提示词会稳定复现超时；仅替换为自然语言夹具即可恢复。第一次使用过于通用的自然语言替代品后，又暴露了现有语料中的独立检索未命中问题，因此最终夹具同时做到自然语言化和检索可区分。

### 已测试的假设

| 假设 | 预测 | 结果 |
| --- | --- | --- |
| 网络、凭据或服务商整体故障 | 直接短服务商调用也失败或超时。 | 已证伪。 |
| Ark 适配器或基础 RAG 图故障 | 普通输入的适配器/图调用失败。 | 已证伪。 |
| 合成机器标识提示词触发服务商长尾 | 旧 sentinel 组合请求超时，而自然语言等价物成功。 | 已确认。 |
| 自然语言夹具对检索不够有区分度 | 通用问题无法取回刚发布的来源。 | 已确认并修正。 |

### 修复与回归覆盖

烟测夹具现在发布一条关于“蓝松石版本”的自然、可区分中文事实，并以自然问题询问该事实。机器标识仍在上传夹具的文件名/内容中用于运行隔离，但不再作为模型问题发送。

[`backend/tests/unit/test_retrieval_evidence.py`](../../backend/tests/unit/test_retrieval_evidence.py) 固化了这一边界和 SSE 引用契约：

- 聊天问题为自然语言，且不包含合成 sentinel；
- 夹具包含用于检索的可区分事实；
- 流响应遗漏被引用来源时，以 `STREAM_CITATION_MISSING` 失败；
- 非敏感清单不包含凭据、服务商 URL、回答或摘录。

修复后完成的验证：

```text
backend 测试套件：191 passed, 1 skipped
生产 generation-smoke：passed
```

没有添加临时调试日志；源码扫描未发现 `[DEBUG-...]` 调试标记。

## 烟测操作指南

### Retrieval Smoke

当只需验证接入与检索半环时，运行 `./retrieval-evidence smoke`。它验证健康检查、Bootstrap Administrator 登录、文档构建、显式发布、实时 Embedding、Milvus 索引，以及取回刚接入的已发布文档；**不会**调用聊天服务商。

### Generation Smoke

当还需验证当前已批准服务商时，运行 `./retrieval-evidence generation-smoke`。它包含全部 Retrieval Smoke 检查，随后验证普通聊天和 SSE 的 evidence-summary 契约。它才是带引用回答的正确生产信号；仅做服务商 HTTP ping 并不等价。

### 生产验收

使用服务器端封装脚本，不要把运行时配置值拷入 shell：

```bash
ssh -i "$HOME/.ssh/zhomind_ops_ed25519" \
  -o BatchMode=yes -o StrictHostKeyChecking=yes \
  ops@45.207.207.65 \
  'sudo -n env \
    DEPLOY_APP_DIR=/opt/zhomind-v2 \
    DEPLOY_CADDY_SITE_ADDRESS=zhomind.kedayoung.cc \
    SOURCE_REVISION=<deployed-revision> \
    bash /opt/zhomind-v2/source/deploy/production/acceptance.sh'
```

封装脚本会自行读取 root 管理的服务器运行时文件。不要对该文件执行 `cat`、`printenv`、shell trace，或渲染 Compose 完整配置。证据清单写入 `/opt/zhomind-v2/evidence/`；只查看其中的通过/失败字段，绝不将来源内容或配置值复制进 issue 评论。

每次烟测都会在运行中的知识库创建并发布验收来源。应把它视为运行测试数据：使用可识别的 run ID，保留清单供追溯；不再需要时通过正常管理员工作流删除或撤回测试内容。

## 服务器连接与已核验的配置状态

### 连接方式

| 项目 | 已核验值或做法 |
| --- | --- |
| 主机 | `45.207.207.65`（`zhomind.kedayoung.cc`） |
| SSH 账号 | `ops` |
| 本地私钥路径 | `$HOME/.ssh/zhomind_ops_ed25519` |
| 主机密钥校验 | 使用 `StrictHostKeyChecking=yes`；预期 ED25519 SHA-256 指纹：`SHA256:gEWi26JW7nC5PLm8012eRHnlouqHoFFu8NH4aY9BeYs` |
| 权限边界 | 对 `/opt/zhomind-v2` 和 Compose 操作使用 `sudo -n`；不得放宽文件权限。 |

常规的非敏感状态检查：

```bash
ssh -i "$HOME/.ssh/zhomind_ops_ed25519" \
  -o BatchMode=yes -o StrictHostKeyChecking=yes \
  ops@45.207.207.65 \
  'sudo -n git -C /opt/zhomind-v2/source rev-parse HEAD && \
   sudo -n env DEPLOY_CADDY_SITE_ADDRESS=zhomind.kedayoung.cc \
     docker compose --env-file /opt/zhomind-v2/.env \
     -f /opt/zhomind-v2/source/deploy/production/compose.yml \
     ps --status running --services'
```

### 2026-08-01 观察到的状态

- 应用目录：`/opt/zhomind-v2`。
- 运行时文件：`/opt/zhomind-v2/.env`，所有者 `root:root`，权限 `0600`；本文未读取其内容。
- 已部署源码修订：`36f6e1cb57c65130ba5c0efa8d18f0ff625e4643`。
- 运行服务：`postgres`、`redis`、`etcd`、`minio`、`milvus`、`backend` 和 `caddy`。
- 网络结构：Caddy 是公网 HTTPS 边缘；Postgres、Redis、etcd、MinIO、Milvus 和 backend 在内部 Compose 网络通信。backend 保有调用已批准外部服务商所需的受控出口。
- 以不暴露值的方式检查了必需运行时配置：bootstrap 管理员、设置加密、数据库/对象存储凭据、Ark 服务商、Embedding 服务商、Embedding 维度及主服务商选择均完整。
- 当前主生成服务商是唯一已批准的 Ark 服务商。证据闸门由带引用生成闭环的生产契约保持启用；不存在服务商回退链。
- 系统设置草稿/应用控制和文档工作者均已配置。工具使用虽已配置，但其启用状态应以应用契约为准，不应由本文推断。
- `https://zhomind.kedayoung.cc/api/health` 返回通过状态。

## 运行边界与后续事项

- 首版容量 ADR 规定：4 vCPU/3.8 GiB 主机、最多 25 名活跃成员、5 路并发问答、500 个已发布版本、25 MiB 上传限制和单个文档构建工作者。常规问答端到端 P95 目标不超过 12 秒；真实服务商调用仍可能失败或超出目标，届时必须 fail-closed。
- 服务商故障时“不回退”的行为由后端契约测试覆盖。不要为证明该点而把生产来源材料发送给无效或替代服务商。
- 发布脚本末尾的浏览器验收依赖本机 `node`；记录的发布中该可选本地步骤不可用。服务器侧 Caddy 前端交付、认证、Retrieval Smoke、Generation Smoke 和公网健康检查均已通过。未来发布前应恢复本机 Node/Playwright，才能依赖这一步最终浏览器验收。
