#!/usr/bin/env bash
# Use WorldFoundry-local Codex home (config/auth/sessions isolated from global shell-config).
set -euo pipefail

if [[ -n "${ZSH_VERSION:-}" ]]; then
  _script="${(%):-%x}"
elif [[ -n "${BASH_VERSION:-}" ]]; then
  _script="${BASH_SOURCE[0]}"
else
  _script="$0"
fi

ROOT="$(cd "$(dirname "$_script")/../.." && pwd)"
export CODEX_HOME="$ROOT/.codex"
unset CODEX_PERSIST_HOME

mkdir -p \
  "$CODEX_HOME/skills" \
  "$CODEX_HOME/sessions" \
  "$CODEX_HOME/state/sqlite"

# Drop stale symlinks to a previously shared Codex home (if any).
# Never hardcode a host-specific absolute path here; only honor an explicit override.
_persist="${CODEX_PERSIST_HOME:-}"
for _item in auth.json config.toml sessions skills; do
  if [[ -L "$CODEX_HOME/$_item" ]]; then
    _target="$(readlink "$CODEX_HOME/$_item" || true)"
    if [[ -n "$_persist" && ( "$_target" == "$_persist/"* || "$_target" == "$_persist" ) ]]; then
      rm "$CODEX_HOME/$_item"
    elif [[ "$_target" == "$CODEX_HOME/$_item" || "$_target" == "$CODEX_HOME/$_item/"* ]]; then
      rm "$CODEX_HOME/$_item"
    fi
  fi
done
unset _item _persist _target

if [[ ! -f "$CODEX_HOME/config.toml" ]]; then
  cat >"$CODEX_HOME/config.toml" <<EOF
sqlite_home = "$CODEX_HOME/state/sqlite"
cli_auth_credentials_store = "file"
model = "gpt-5.6-luna"
model_reasoning_effort = "high"
approvals_reviewer = "user"
service_tier = "default"

[projects."$ROOT"]
trust_level = "trusted"

[notice]
hide_full_access_warning = true
hide_rate_limit_model_nudge = true
EOF
fi

if (return 0 2>/dev/null); then
  :
else
  if [[ $# -eq 0 ]]; then
    echo "CODEX_HOME=$CODEX_HOME"
    echo "Next: codex login --device-auth"
    exit 0
  fi
  exec "$@"
fi
