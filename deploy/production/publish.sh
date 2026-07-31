#!/usr/bin/env bash
set -euo pipefail

repository_root=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)

# Deployment target values are supplied only for this invocation, never via .env.

fail() {
  printf '%s\n' "$*" >&2
  exit 1
}

require_value() {
  local name=$1
  [[ -n "${!name:-}" ]] || fail "missing required deployment value: $name"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command is unavailable: $1"
}

DEPLOY_SSH_PORT=${DEPLOY_SSH_PORT:-22}
DEPLOY_APP_DIR=${DEPLOY_APP_DIR:-/opt/zhomind-v2}
DEPLOY_GIT_REF=${DEPLOY_GIT_REF:-refs/heads/experiment/retrieval-evidence}
DEPLOY_CONFIGURE_UFW=${DEPLOY_CONFIGURE_UFW:-false}

for variable in \
  DEPLOY_HOST DEPLOY_SSH_USER DEPLOY_SSH_IDENTITY_FILE DEPLOY_SSH_HOST_KEY_SHA256 \
  DEPLOY_CADDY_SITE_ADDRESS; do
  require_value "$variable"
done

[[ "$DEPLOY_SSH_PORT" =~ ^[0-9]+$ ]] || fail "DEPLOY_SSH_PORT must be numeric"
[[ "$DEPLOY_HOST" =~ ^[A-Za-z0-9.:-]+$ ]] || fail "DEPLOY_HOST contains unsupported characters"
[[ "$DEPLOY_SSH_USER" =~ ^[A-Za-z_][A-Za-z0-9_-]*$ ]] || fail "DEPLOY_SSH_USER contains unsupported characters"
[[ "$DEPLOY_APP_DIR" =~ ^/[A-Za-z0-9._/-]+$ ]] || fail "DEPLOY_APP_DIR must be an absolute path without spaces"
[[ "$DEPLOY_GIT_REF" =~ ^refs/heads/[A-Za-z0-9._/-]+$ ]] || fail "DEPLOY_GIT_REF contains unsupported characters"
[[ "$DEPLOY_CONFIGURE_UFW" == "true" || "$DEPLOY_CONFIGURE_UFW" == "false" ]] || \
  fail "DEPLOY_CONFIGURE_UFW must be true or false"
[[ "$DEPLOY_CADDY_SITE_ADDRESS" != *://* && "$DEPLOY_CADDY_SITE_ADDRESS" != */* ]] || \
  fail "DEPLOY_CADDY_SITE_ADDRESS must be a DNS name, without a scheme or path"
[[ "$DEPLOY_CADDY_SITE_ADDRESS" =~ ^[A-Za-z0-9.-]+$ && "$DEPLOY_CADDY_SITE_ADDRESS" == *.* ]] || \
  fail "DEPLOY_CADDY_SITE_ADDRESS must be a DNS name"
[[ -f "$DEPLOY_SSH_IDENTITY_FILE" ]] || fail "DEPLOY_SSH_IDENTITY_FILE does not exist"
[[ "$DEPLOY_GIT_REF" == refs/heads/* ]] || fail "DEPLOY_GIT_REF must name a local branch"
git -C "$repository_root" show-ref --verify --quiet "$DEPLOY_GIT_REF" || \
  fail "DEPLOY_GIT_REF is not present locally: $DEPLOY_GIT_REF"

for command in git ssh scp ssh-keyscan ssh-keygen curl; do
  require_command "$command"
done

temporary_directory=$(mktemp -d "${TMPDIR:-/tmp}/zhomind-publish.XXXXXX")
cleanup() {
  rm -rf "$temporary_directory"
}
trap cleanup EXIT

known_hosts="$temporary_directory/known_hosts"
bundle="$temporary_directory/zhomind.bundle"
ssh-keyscan -T 10 -p "$DEPLOY_SSH_PORT" "$DEPLOY_HOST" > "$known_hosts" 2>/dev/null || \
  fail "could not obtain an SSH host key from $DEPLOY_HOST:$DEPLOY_SSH_PORT"

if ! ssh-keygen -lf "$known_hosts" -E sha256 | awk '{print $2}' | grep -Fqx "$DEPLOY_SSH_HOST_KEY_SHA256"; then
  printf 'received SSH host-key fingerprints:\n' >&2
  ssh-keygen -lf "$known_hosts" -E sha256 >&2
  fail "SSH host-key fingerprint does not match DEPLOY_SSH_HOST_KEY_SHA256"
fi

git -C "$repository_root" bundle create "$bundle" "$DEPLOY_GIT_REF"
git bundle verify "$bundle" >/dev/null
source_revision=$(git -C "$repository_root" rev-parse "$DEPLOY_GIT_REF")

ssh_options=(
  -i "$DEPLOY_SSH_IDENTITY_FILE"
  -o BatchMode=yes
  -o StrictHostKeyChecking=yes
  -o UserKnownHostsFile="$known_hosts"
  -p "$DEPLOY_SSH_PORT"
)
scp_options=(
  -i "$DEPLOY_SSH_IDENTITY_FILE"
  -o BatchMode=yes
  -o StrictHostKeyChecking=yes
  -o UserKnownHostsFile="$known_hosts"
  -P "$DEPLOY_SSH_PORT"
)
remote_target="$DEPLOY_SSH_USER@$DEPLOY_HOST"

scp "${scp_options[@]}" "$bundle" "$remote_target:/tmp/"

ssh "${ssh_options[@]}" "$remote_target" \
  "DEPLOY_APP_DIR='$DEPLOY_APP_DIR' DEPLOY_GIT_REF='$DEPLOY_GIT_REF' DEPLOY_CONFIGURE_UFW='$DEPLOY_CONFIGURE_UFW' DEPLOY_CADDY_SITE_ADDRESS='$DEPLOY_CADDY_SITE_ADDRESS' SOURCE_REVISION='$source_revision' bash -s" <<'REMOTE_SCRIPT'
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

install_docker() {
  if command -v docker >/dev/null 2>&1 && as_root docker compose version >/dev/null 2>&1; then
    return
  fi
  as_root apt-get update
  as_root apt-get install -y ca-certificates curl git docker.io docker-compose-v2 ufw
  as_root systemctl enable --now docker
}

install_docker

if [[ "$DEPLOY_CONFIGURE_UFW" == "true" ]]; then
  as_root ufw default deny incoming
  as_root ufw default allow outgoing
  as_root ufw allow "${SSH_CONNECTION##* }"/tcp
  as_root ufw allow 80/tcp
  as_root ufw allow 443/tcp
  as_root ufw allow 443/udp
  as_root ufw --force enable
fi

as_root install -d -m 0750 "$DEPLOY_APP_DIR"
# The application directory is intentionally root-owned and not traversable by
# the deployment user, so this check must use the same privilege boundary as
# the subsequent Compose invocation.
as_root test -f "$DEPLOY_APP_DIR/.env" || fail "missing server runtime configuration: $DEPLOY_APP_DIR/.env"

source_directory="$DEPLOY_APP_DIR/source"
bundle=/tmp/zhomind.bundle
if [[ -d "$source_directory/.git" ]]; then
  as_root git -C "$source_directory" diff --quiet
  as_root git -C "$source_directory" diff --cached --quiet
  as_root git -C "$source_directory" fetch "$bundle" "$DEPLOY_GIT_REF:$DEPLOY_GIT_REF"
else
  [[ ! -e "$source_directory" ]] || fail "refusing to replace a non-Git source directory: $source_directory"
  as_root git clone --branch "${DEPLOY_GIT_REF#refs/heads/}" "$bundle" "$source_directory"
fi
as_root git -C "$source_directory" checkout --detach "$SOURCE_REVISION"
as_root git -C "$source_directory" diff --quiet
as_root git -C "$source_directory" diff --cached --quiet

compose=(as_root env DEPLOY_CADDY_SITE_ADDRESS="$DEPLOY_CADDY_SITE_ADDRESS" docker compose --env-file "$DEPLOY_APP_DIR/.env" -f "$source_directory/deploy/production/compose.yml")
"${compose[@]}" up -d --build --remove-orphans
"${compose[@]}" ps

as_root rm -f /tmp/zhomind.bundle
REMOTE_SCRIPT

ssh "${ssh_options[@]}" "$remote_target" \
  "DEPLOY_APP_DIR='$DEPLOY_APP_DIR' DEPLOY_CADDY_SITE_ADDRESS='$DEPLOY_CADDY_SITE_ADDRESS' SOURCE_REVISION='$source_revision' bash -s" \
  < "$repository_root/deploy/production/acceptance.sh"

public_base_url="https://$DEPLOY_CADDY_SITE_ADDRESS"
for attempt in $(seq 1 30); do
  if curl --fail --silent --show-error "$public_base_url/api/health" > "$temporary_directory/public-health.json"; then
    python3 - "$temporary_directory/public-health.json" <<'PYTHON'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    payload = json.load(source)
if payload.get("data", {}).get("status") != "up":
    raise SystemExit("public health response did not report status=up")
PYTHON
    printf 'public HTTPS health check passed\n'
    DEPLOY_CADDY_SITE_ADDRESS="$DEPLOY_CADDY_SITE_ADDRESS" \
      node "$repository_root/deploy/production/browser-acceptance.mjs"
    exit 0
  fi
  sleep 2
done

fail "deployment started, but the public HTTPS health check did not pass"
