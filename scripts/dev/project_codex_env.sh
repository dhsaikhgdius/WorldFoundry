#!/usr/bin/env bash
# Use WorldFoundry-local Codex home (config/auth/sessions isolated from any shared Codex home).
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

# Capture an explicit persist home for stale-link cleanup, then isolate this shell
# from shared Codex state. Never hard-code a host path here (path-hygiene gate).
_persist_home="${CODEX_PERSIST_HOME:-}"
unset CODEX_PERSIST_HOME

mkdir -p "$CODEX_HOME"

# Drop stale symlinks that point at a shared/external Codex home before creating
# the local directory tree (so mkdir does not follow a bad link).
for _item in auth.json config.toml sessions skills; do
  if [[ -L "$CODEX_HOME/$_item" ]]; then
    _target="$(readlink "$CODEX_HOME/$_item" || true)"
    _remove=0
    if [[ -n "$_persist_home" && ( "$_target" == "$_persist_home" || "$_target" == "$_persist_home/"* ) ]]; then
      _remove=1
    elif [[ "$_target" == /* && "$_target" != "$CODEX_HOME" && "$_target" != "$CODEX_HOME/"* ]]; then
      # Absolute symlink leaving the repo-local Codex home.
      _remove=1
    elif [[ "$_target" == "$CODEX_HOME/$_item" || "$_target" == "$CODEX_HOME/$_item/"* ]]; then
      _remove=1
    fi
    if [[ "$_remove" -eq 1 ]]; then
      rm "$CODEX_HOME/$_item"
    fi
  fi
done
unset _item _persist_home _target _remove

mkdir -p \
  "$CODEX_HOME/skills" \
  "$CODEX_HOME/sessions" \
  "$CODEX_HOME/state/sqlite"

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
