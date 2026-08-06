# 认证与成员准入

ZhoMind-v2 不提供公开注册路径，成员准入由系统管理员控制。

## 角色与准入

Bootstrap Administrator 在部署时创建，是系统管理员的初始来源。团队邀请（Team Invitation）是可复用的、由管理员签发的注册凭据，只在其配置的过期时间之前允许准入一位 Knowledge User，默认过期时间为七天。邀请可以被撤销而不移除已有成员；它不是公开注册路径，也不是一次性邀请码。

Administrator Promotion 是已有 System Administrator 将已准入的 Knowledge User 提升为 System Administrator 的动作，是 Bootstrap Administrator 之外获得管理员权限的唯一途径；不存在自助注册管理员或基于邀请的管理员创建。

Deactivated Member 是被停用的已准入成员：其活动会话被吊销、未来访问被阻止，但账户与对话记录不会被立即删除。

## 知识用户提问

已准入的 Knowledge User 通过 `/api/v1/chat` 或 `/api/v1/chat/stream` 提问。对话记录对成员本人私密：任何管理员不得通过 Administrator Operations Surface 查看他人的 Private Conversation Record，成员本人可以删除自己的记录，记录从持久化存储中三十天后自动移除。

## 来源

- Bootstrap Administrator 与成员准入：`backend/app/service/member_admission_service.py`、`backend/app/api/v1/auth.py`
- Team Invitation：ADR-0004 `docs/adr/0004-gate-knowledge-user-registration-with-time-limited-invitations.md`
- Administrator Promotion：ADR-0007 `docs/adr/0007-restrict-administrator-admission.md`
- Deactivated Member：ADR-0008 `docs/adr/0008-deactivate-members-instead-of-deleting-them.md`
- 对话私密与保留期限：ADR-0012 `docs/adr/0012-keep-conversations-private-and-time-limited.md`
