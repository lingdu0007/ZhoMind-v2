# 配置与部署

ZhoMind-v2 后端基于 Python 3.11 与 FastAPI，前端基于 Vite 构建的浏览器应用。本地开发栈由 Docker Compose 编排：Postgres 15 存储权威数据，Redis 7-alpine 承载任务队列，etcd v3.5.18 与 MinIO 支撑 Milvus v2.5.14 独立部署，Attu v2.5.11 提供 Milvus 管理界面。后端启动时先执行 `alembic upgrade head` 迁移，再以 uvicorn 监听 8000 端口。

## 依赖锁定

后端依赖以 `uv.lock` 锁定，安装通过 `uv sync --frozen` 进行；镜像内安装使用 `--no-dev`，锁文件与 `pyproject.toml` 不一致时构建失败，而不是在锁外静默解析。前端依赖以 `package-lock.json` 锁定，安装使用 `npm ci`。PR Gate 与发布构建使用相同的锁定输入。

## 确定性 PR Gate

`.github/workflows/pr-gate.yml` 定义公开的 PR Gate：Ruff 检查后端源码与测试、Pyright basic 检查后端应用代码、完整确定性后端测试套件（其中 live-Milvus 稠密 E2E 测试保持显式 opt-in，未启用时报告为 skip，绝不作为 live 证据）、前端单元测试、生产构建、disposable-API 浏览器验收、英文/中文文档对齐检查、Public Evidence Bundle 校验与秘密扫描。PR Gate 不调用付费模型 provider，不依赖持久化生产服务或私有主机。

## 隔离的 Retrieval Smoke

`./retrieval-evidence smoke` 在每次运行时启动独立的 Compose project（项目名带 run id 前缀）与独立的运行时 volumes，映射到隔离的主机端口，结束运行时清理全部隔离资源。该命令先验证健康检查与 Bootstrap Administrator 登录，上传并显式发布一个带 run id sentinel 的验收来源文档，然后通过既有稠密检索路径验证：文档构建、live Qwen embedding、Milvus 索引与刚发布文档的检索命中，全程不调用聊天模型。成功时向 `evidence/runs/<run-id>/manifest.json` 写入非敏感 manifest，记录 source revision、provider 身份、嵌入合约指纹、有界计数与规范化结果；任何验收条件失败时命令以非零状态退出。

## 来源

- Compose 服务与镜像版本：`docker-compose.yml`
- 后端镜像构建与锁定安装：`backend/Dockerfile`
- PR Gate：`.github/workflows/pr-gate.yml`
- Retrieval Smoke 行为：`retrieval-evidence`、`backend/app/retrieval_evidence.py`
- 运行配置：`backend/.env.example`
