# Admin-managed system settings

Status: accepted

Model, retrieval, storage, and API parameters are System Settings, not browser-local preferences. They will be represented as an administrator-only workspace and later stored and served through administrator-protected backend APIs, because the current localStorage-only form neither changes server behavior nor provides an appropriate boundary for sensitive operational values.

## Considered Options

- Keep the current browser-local form and rename it as local preferences.
- Treat the settings as server-managed system state protected by administrator authorization.

## Consequences

The UI must not present browser localStorage as authoritative system state. Follow-up implementation requires a protected configuration API and server-side persistence before the settings workspace can claim to modify the running system.
