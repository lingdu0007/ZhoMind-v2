# 文档发布生命周期

ZhoMind-v2 使用 Manual Publication Workflow：管理员上传或替换材料、构建、检查 Candidate Build、显式发布已批准的 generation。这是首个版本中源内容变化的唯一途径。

## 构建与发布步骤

管理员通过 `POST /api/v1/documents/upload` 上传 Markdown 文件，接口只接受 `file` 字段，默认使用 `general` 分块策略，返回 `document_id` 与 `job_id`。替换上传按文件名去重，同名文件的上传会创建新的候选 generation。

构建任务的状态机为 `queued` → `running` → `succeeded` / `failed` / `canceled`。`POST /api/v1/documents/{document_id}/build` 与批量构建 `POST /api/v1/documents/batch-build` 可显式指定 `chunk_strategy`。构建成功后产生候选 generation 及其文档块。

`POST /api/v1/documents/{document_id}/publish` 把候选 generation 提升为已发布 generation：Candidate Build 成为 Published Knowledge Version，此时 Knowledge User 才可以检索到它。构建成功本身不构成发布。发布后的文档通过 `GET /api/v1/documents/{document_id}/chunks` 分页查看文档块与总数。

## 撤回与检索排除

管理员撤下某个已发布文档后，该 Published Knowledge Version 成为 Withdrawn Source：立即从未来的检索中排除，历史回答只保留撤下通知，不再暴露其摘录。撤下不影响其他文档的检索与回答。

## 来源

- 上传、构建、发布接口：`backend/app/api/v1/documents.py`
- 构建任务状态机：`backend/app/service/build_service.py`
- Manual Publication Workflow：ADR-0009 `docs/adr/0009-use-manual-source-updates-for-the-first-release.md`
- Candidate Build 与 Published Knowledge Version 语义：CONTEXT 领域词表
- Withdrawn Source：ADR-0005 `docs/adr/0005-retire-withdrawn-sources-from-retrieval.md`
