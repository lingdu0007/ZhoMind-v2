# 产品概览：证据门控的知识库回答

ZhoMind-v2 是一个面向团队的证据门控 RAG 应用。它运行在一个 Team-Shared Knowledge Base 之上：整个团队共享单一知识库，已发布的 Published Knowledge Version 对每一位已准入的 Knowledge User 可检索，不提供个人私库、按文档的访问控制或混合敏感度语料。

Knowledge User 提问后，系统先执行 Evidence-Gated Answer Execution 这一统一边界：规范化问题、在检索之前先为少量社交提示词选择 Non-Knowledge-Base Reply 例外、检索并形成唯一的 Answer Evidence Set、应用证据门控、仅在允许时调用 Approved Generation Provider，最终返回唯一的 Answer Execution Outcome。所有 HTTP、SSE 与对话历史适配层只负责持久化和投影这一结果，不允许重新解释。

Answer Execution Outcome 恰好有四种结果：Evidence-Gated Answer、Insufficient Evidence Reply、Non-Knowledge-Base Reply、Generation Unavailable。调用方消费这一封闭结果，而不是根据门控标志、原因字符串、证据或 provider 文本来自行推断结果。

Evidence-Gated Answer 的引用来源集合是 Answer Evidence Set：在首次发布策略下，它按最终检索顺序最多取前三个有效 passage 组成；数量上限是配置项而非领域常量；该集合在一次请求内不可变。证据门控、回答生成、持久化引用与用户可见引用都引用这同一个集合，集合之外的检索结果只作为管理员诊断候选，不得进入回答。Answer Evidence Set 中的每一项携带一个 Evidence Excerpt Snapshot，同一快照进入生成 prompt、持久化引用、普通响应、SSE 响应与对话历史，任何调用方不得重新切片或替换其文本。

当检索形成的 Answer Evidence Set 为空时，系统返回 Insufficient Evidence Reply：不调用任何生成 provider、不带任何引用，调用方不能把它升级为知识回答。当 Approved Generation Provider 不可用时，系统返回 Generation Unavailable：检索可能已经产生来源，但已检索到的来源不会被发送给其他 provider，这是 Fail-Closed Generation 策略。Non-Knowledge-Base Reply 是唯一的生产例外：它只在检索之前对一小部分社交提示词开放，明确标注为非知识库回复，不做知识库声明、Answer Evidence Set 为空、不带来源引用、不调用任何生成 provider。

## 来源

- 证据门控回答边界与四种结果：`backend/app/rag/answer_execution.py`、ADR-0015 `docs/adr/0015-unify-evidence-gated-answer-execution.md`
- Answer Evidence Set 与 Evidence Excerpt Snapshot 语义：`docs/adr/0015-unify-evidence-gated-answer-execution.md`、CONTEXT 领域词表
- Team-Shared Knowledge Base：ADR-0006 `docs/adr/0006-keep-v1-knowledge-base-team-shared.md`
- Insufficient Evidence Reply：`backend/app/rag/answer_execution.py`、ADR-0002
- Generation Unavailable 与 Fail-Closed Generation：ADR-0010 `docs/adr/0010-fail-closed-on-generation-provider-outage.md`
- Non-Knowledge-Base Reply：`backend/app/service/chat_service.py` 的社交提示词 allowlist
