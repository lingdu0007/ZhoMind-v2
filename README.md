# ZhoMind-v2

## Minimal Local Run

Run these commands from the repository root:

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` before startup. At minimum, fill the blank required values such as `JWT_SECRET`, `ADMIN_INVITE_CODE`, provider API keys, and any provider-specific base URLs or tokens you actually use.

Start the backend stack with your local ignored runtime config:

```bash
docker compose up -d backend
```

Run the smoke script after the backend is healthy. If `backend/.env` sets `ADMIN_INVITE_CODE`, export the same value through `SMOKE_ADMIN_CODE` first:

```bash
SMOKE_ADMIN_CODE='<same as backend/.env ADMIN_INVITE_CODE>' node frontend/tests/user-path-smoke.mjs
```
