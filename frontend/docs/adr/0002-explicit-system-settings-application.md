# Explicit system-settings application

Status: accepted

System Settings are edited as a draft and take effect only after an administrator explicitly saves and applies them. The interface will distinguish saved from active settings and show application status and audit metadata, preventing an input edit from silently changing live retrieval or model behavior.

## Considered Options

- Apply every field immediately as it changes.
- Save a draft and require an explicit apply action.

## Consequences

The future configuration API must expose enough state to report the saved and active versions, application outcome, and audit information. The UI must represent pending or asynchronous application honestly rather than assuming instant success.
