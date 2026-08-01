#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '%s\n' "$*" >&2
  exit 1
}

as_root() {
  if [[ $(id -u) -eq 0 ]]; then
    "$@"
  else
    sudo -n "$@"
  fi
}

load_env() {
  local line key value
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    [[ "$line" == *=* ]] || fail "invalid .env line: $line"
    key=${line%%=*}
    value=${line#*=}
    [[ "$key" =~ ^[A-Z][A-Z0-9_]*$ ]] || fail "invalid .env variable name: $key"
    export "$key=$value"
  done < <(as_root cat "$DEPLOY_APP_DIR/.env")
}

require_value() {
  local name=$1
  [[ -n "${!name:-}" ]] || fail "missing required runtime value: $name"
}

[[ -n "${DEPLOY_APP_DIR:-}" ]] || fail "DEPLOY_APP_DIR is required"
[[ -n "${DEPLOY_CADDY_SITE_ADDRESS:-}" ]] || fail "DEPLOY_CADDY_SITE_ADDRESS is required"
[[ -n "${SOURCE_REVISION:-}" ]] || fail "SOURCE_REVISION is required"
load_env

for variable in BOOTSTRAP_ADMIN_USERNAME BOOTSTRAP_ADMIN_PASSWORD; do
  require_value "$variable"
done

command -v openssl >/dev/null 2>&1 || fail "openssl is required for acceptance credentials"

source_directory="$DEPLOY_APP_DIR/source"
compose=(as_root env DEPLOY_CADDY_SITE_ADDRESS="$DEPLOY_CADDY_SITE_ADDRESS" docker compose --env-file "$DEPLOY_APP_DIR/.env" -f "$source_directory/deploy/production/compose.yml")

for service in postgres redis etcd minio milvus backend caddy; do
  "${compose[@]}" ps --status running --services | grep -Fqx "$service" || fail "service is not running: $service"
done
printf 'compose services are running\n'

base_url="https://$DEPLOY_CADDY_SITE_ADDRESS"
curl_args=(--fail --silent --show-error --resolve "$DEPLOY_CADDY_SITE_ADDRESS:443:127.0.0.1")
health_json=$(curl "${curl_args[@]}" "$base_url/api/health")
printf '%s' "$health_json" | python3 -c '
import json
import sys
payload = json.load(sys.stdin)
if payload.get("data", {}).get("status") != "up":
    raise SystemExit("health response did not report status=up")
'
printf 'Caddy to API health path passed\n'

homepage=$(curl "${curl_args[@]}" "$base_url/")
[[ "$homepage" == *'id="app"'* ]] || fail "frontend application shell is missing"
printf 'Caddy static frontend path passed\n'

login_json=$(curl "${curl_args[@]}" -H 'Content-Type: application/json' \
  --data "{\"username\":\"$BOOTSTRAP_ADMIN_USERNAME\",\"password\":\"$BOOTSTRAP_ADMIN_PASSWORD\"}" \
  "$base_url/api/auth/login")
access_token=$(printf '%s' "$login_json" | python3 -c '
import json
import sys
payload = json.load(sys.stdin)
token = payload.get("data", {}).get("access_token")
if not isinstance(token, str) or not token:
    raise SystemExit("login response did not include an access token")
print(token)
')
identity_json=$(curl "${curl_args[@]}" -H "Authorization: Bearer $access_token" "$base_url/api/auth/me")
printf '%s' "$identity_json" | python3 -c '
import json
import sys
payload = json.load(sys.stdin)
if payload.get("data", {}).get("role") != "admin":
    raise SystemExit("authenticated identity did not report admin role")
'
printf 'Bootstrap Administrator login and identity path passed\n'

evidence_directory="$DEPLOY_APP_DIR/evidence"
as_root install -d -m 0750 "$evidence_directory"
run_id="production-$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"

if [[ -n "${EMBEDDING_API_KEY:-}" && -n "${EMBEDDING_BASE_URL:-}" && -n "${EMBEDDING_MODEL:-}" && "${DENSE_EMBEDDING_DIM:-0}" != "0" ]]; then
  "${compose[@]}" run --rm --no-deps \
    --volume "$evidence_directory:/evidence" \
    backend python -m app.retrieval_evidence smoke \
    --base-url http://backend:8000 \
    --output-dir /evidence \
    --source-revision "$SOURCE_REVISION" \
    --run-id "$run_id" >/dev/null
  printf 'live embedding, Milvus indexing, and Retrieval Smoke passed\n'
else
  printf 'live Retrieval Smoke skipped because embedding configuration is incomplete\n'
fi

if [[ -n "${EMBEDDING_API_KEY:-}" && -n "${EMBEDDING_BASE_URL:-}" && -n "${EMBEDDING_MODEL:-}" && "${DENSE_EMBEDDING_DIM:-0}" != "0" && "${RAG_PRIMARY_LLM_PROVIDER:-ark}" == "ark" && -n "${ARK_API_KEY:-}" && -n "${BASE_URL:-}" && -n "${MODEL:-}" ]]; then
  generation_run_id="${run_id}-generation"
  "${compose[@]}" run --rm --no-deps \
    --volume "$evidence_directory:/evidence" \
    backend python -m app.retrieval_evidence generation-smoke \
    --base-url http://backend:8000 \
    --output-dir /evidence \
    --source-revision "$SOURCE_REVISION" \
    --run-id "$generation_run_id" >/dev/null
  printf 'live approved-provider cited chat and stream Generation Smoke passed\n'
else
  fail 'live Generation Smoke requires complete approved-provider configuration'
fi
