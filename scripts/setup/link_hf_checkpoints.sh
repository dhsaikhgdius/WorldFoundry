#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WORLDFOUNDRY_HOME_DEFAULT="${XDG_CACHE_HOME:-$HOME/.cache}/worldfoundry"
WORLDFOUNDRY_HOME_VALUE="${WORLDFOUNDRY_HOME:-$WORLDFOUNDRY_HOME_DEFAULT}"
MODEL_ROOT_VALUE="${WORLDFOUNDRY_MODEL_DIR:-${WORLDFOUNDRY_HOME_VALUE}/models}"
CKPT_DIR="${WORLDFOUNDRY_CKPT_DIR:-${MODEL_ROOT_VALUE}/checkpoints}"
HFD_ROOT="${WORLDFOUNDRY_HFD_ROOT:-${CKPT_DIR}/hfd}"
HF_HOME_VALUE="${HF_HOME:-${WORLDFOUNDRY_HOME_VALUE}/huggingface}"
HF_HUB_CACHE_VALUE="${HF_HUB_CACHE:-${HF_HOME_VALUE}/hub}"
FORCE=0
HFD_ROOT_EXPLICIT=0
HFD_ROOT_FROM_ENV=0
REPO_SPECS=()

if [[ -n "${WORLDFOUNDRY_HFD_ROOT:-}" ]]; then
  HFD_ROOT_FROM_ENV=1
fi

usage() {
  cat <<'EOF'
Usage: bash scripts/setup/link_hf_checkpoints.sh [options]

Create no-copy checkpoint aliases for machines that already have model weights.
The script links each local checkpoint directory into both supported layouts:

  1. HFD-style alias:
     ${WORLDFOUNDRY_HFD_ROOT}/OWNER--REPO -> ${WORLDFOUNDRY_CKPT_DIR}/LOCAL_DIR

  2. Native Hugging Face cache alias:
     ${HF_HUB_CACHE}/models--OWNER--REPO/snapshots/<40-hex-local-revision> -> LOCAL_DIR
     ${HF_HUB_CACHE}/models--OWNER--REPO/refs/main = <40-hex-local-revision>

New users can ignore this script and let Hugging Face download normally.
Cluster users with shared checkpoints can run it to reuse existing weights
without copying large directories.

Options:
  --ckpt-dir PATH       Existing checkpoint root. Default: $WORLDFOUNDRY_CKPT_DIR.
  --hfd-root PATH       HFD alias root. Default: $WORLDFOUNDRY_HFD_ROOT or <ckpt-dir>/hfd.
  --hf-hub-cache PATH   Native Hugging Face hub cache root. Default: $HF_HUB_CACHE.
  --repo REPO=DIR       Add one repo mapping, for example
                        --repo THUDM/CogVideoX-5b-I2V=CogVideoX-5b-I2V.
                        DIR may be absolute or relative to --ckpt-dir.
  --force               Replace existing symlinks/files owned by this layout.
  --default-world       Link common WorldFoundry world/video checkpoints if present
                        (from scripts/setup/default_world_checkpoint_links.yaml).
  -h, --help            Show this help.
EOF
}

add_default_world_repos() {
  local manifest="${WORLDFOUNDRY_DEFAULT_WORLD_LINKS_MANIFEST:-${ROOT}/scripts/setup/default_world_checkpoint_links.yaml}"
  if [[ ! -f "${manifest}" ]]; then
    echo "Error: default-world links manifest not found: ${manifest}" >&2
    exit 1
  fi
  local python_bin="${PYTHON:-}"
  if [[ -z "${python_bin}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
      python_bin="python3"
    elif command -v python >/dev/null 2>&1; then
      python_bin="python"
    else
      echo "Error: python is required to load ${manifest}" >&2
      exit 1
    fi
  fi
  local mapped=""
  mapped="$("${python_bin}" - "${manifest}" <<'PY'
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"PyYAML is required to load {sys.argv[1]}: {exc}") from exc

payload = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
repos = payload.get("repos") or []
lines = []
for item in repos:
    if not isinstance(item, dict):
        raise SystemExit(f"invalid repo entry in {sys.argv[1]}: {item!r}")
    repo_id = str(item.get("repo_id") or "").strip()
    local_dir = str(item.get("local_dir") or "").strip()
    if not repo_id or not local_dir:
        raise SystemExit(f"repo_id/local_dir required in {sys.argv[1]}: {item!r}")
    if "=" in repo_id or "=" in local_dir:
        raise SystemExit(f"repo mapping must not contain '=': {item!r}")
    lines.append(f"{repo_id}={local_dir}")
print("\n".join(lines))
PY
)" || return 1
  if [[ -z "${mapped}" ]]; then
    echo "Warning: ${manifest} produced no repo mappings." >&2
    return 0
  fi
  while IFS= read -r spec; do
    [[ -n "${spec}" ]] || continue
    REPO_SPECS+=("${spec}")
  done <<<"${mapped}"
}

while (($#)); do
  case "$1" in
    --ckpt-dir)
      CKPT_DIR="$2"
      shift 2
      ;;
    --hfd-root)
      HFD_ROOT="$2"
      HFD_ROOT_EXPLICIT=1
      shift 2
      ;;
    --hf-hub-cache)
      HF_HUB_CACHE_VALUE="$2"
      shift 2
      ;;
    --repo)
      REPO_SPECS+=("$2")
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --default-world)
      add_default_world_repos
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$HFD_ROOT_EXPLICIT" == "0" && "$HFD_ROOT_FROM_ENV" == "0" ]]; then
  HFD_ROOT="${CKPT_DIR}/hfd"
fi

if [[ "${#REPO_SPECS[@]}" == "0" ]]; then
  add_default_world_repos
fi

mkdir -p "$HFD_ROOT" "$HF_HUB_CACHE_VALUE"

repo_cache_name() {
  local repo_id="$1"
  printf 'models--%s\n' "${repo_id//\//--}"
}

hfd_name() {
  local repo_id="$1"
  printf '%s\n' "${repo_id//\//--}"
}

hf_local_revision() {
  local repo_id="$1"
  printf 'worldfoundry-local:%s' "$repo_id" | sha1sum | awk '{print $1}'
}

resolve_local_dir() {
  local repo_id="$1"
  local local_ref="$2"
  local hfd_ref
  hfd_ref="$(hfd_name "$repo_id")"
  local cache_ref
  cache_ref="$(repo_cache_name "$repo_id")"
  local repo_basename="${repo_id##*/}"
  local candidates=()
  if [[ "$local_ref" = /* ]]; then
    candidates+=("$local_ref")
  else
    candidates+=("${CKPT_DIR}/${local_ref}")
  fi
  candidates+=(
    "${MODEL_ROOT_VALUE}/${repo_id}"
    "${MODEL_ROOT_VALUE}/${hfd_ref}"
    "${MODEL_ROOT_VALUE}/${cache_ref}"
    "${CKPT_DIR}/${repo_basename}"
    "${CKPT_DIR}/${hfd_ref}"
    "${CKPT_DIR}/${cache_ref}"
    "${HFD_ROOT}/${hfd_ref}"
    "${HFD_ROOT}/${cache_ref}"
  )

  local candidate
  local seen=":"
  for candidate in "${candidates[@]}"; do
    if [[ "$seen" == *":${candidate}:"* ]]; then
      continue
    fi
    seen="${seen}${candidate}:"
    if [[ -d "$candidate" ]]; then
      realpath -m "$candidate"
      return 0
    fi
  done

  realpath -m "${candidates[0]}"
  return 1
}

replace_path_if_allowed() {
  local path="$1"
  if [[ -L "$path" || -f "$path" ]]; then
    if [[ "$FORCE" == "1" ]]; then
      rm -f "$path"
    else
      return 1
    fi
  elif [[ -e "$path" ]]; then
    return 1
  fi
  return 0
}

link_one_repo() {
  local spec="$1"
  local repo_id="${spec%%=*}"
  local local_ref="${spec#*=}"
  if [[ "$repo_id" == "$spec" || -z "$repo_id" || -z "$local_ref" || "$repo_id" != */* ]]; then
    echo "Invalid --repo mapping: ${spec}" >&2
    return 2
  fi

  local local_dir
  if ! local_dir="$(resolve_local_dir "$repo_id" "$local_ref")"; then
    echo "skip ${repo_id}: local checkpoint not found at ${local_dir}"
    return 0
  fi

  local hfd_link="${HFD_ROOT}/$(hfd_name "$repo_id")"
  if [[ "$(realpath -m "$hfd_link")" == "$(realpath -m "$local_dir")" ]]; then
    echo "hfd source already present ${hfd_link}"
  elif replace_path_if_allowed "$hfd_link"; then
    ln -s "$local_dir" "$hfd_link"
    echo "linked hfd ${hfd_link} -> ${local_dir}"
  else
    echo "keep existing hfd ${hfd_link}"
  fi

  local repo_cache="${HF_HUB_CACHE_VALUE}/$(repo_cache_name "$repo_id")"
  local snapshot_dir="${repo_cache}/snapshots"
  local refs_dir="${repo_cache}/refs"
  local revision
  revision="$(hf_local_revision "$repo_id")"
  local snapshot_link="${snapshot_dir}/${revision}"
  local refs_main="${refs_dir}/main"
  mkdir -p "$snapshot_dir" "$refs_dir"

  if replace_path_if_allowed "$snapshot_link"; then
    ln -s "$local_dir" "$snapshot_link"
    echo "linked hf snapshot ${snapshot_link} -> ${local_dir}"
  else
    echo "keep existing hf snapshot ${snapshot_link}"
  fi

  local current_ref=""
  if [[ -f "$refs_main" || -L "$refs_main" ]]; then
    current_ref="$(tr -d '\r\n' <"$refs_main" || true)"
  fi
  if [[ "$FORCE" == "1" ]]; then
    rm -f "$refs_main"
    printf '%s' "$revision" >"$refs_main"
    echo "wrote hf ref ${refs_main}"
  elif [[ ! -e "$refs_main" ]]; then
    printf '%s' "$revision" >"$refs_main"
    echo "wrote hf ref ${refs_main}"
  elif [[ "$current_ref" == "local" || "$current_ref" == "worldfoundry-local" || -z "$current_ref" || ! -e "${snapshot_dir}/${current_ref}" ]]; then
    if [[ -f "$refs_main" || -L "$refs_main" ]]; then
      printf '%s' "$revision" >"$refs_main"
      echo "updated hf ref ${refs_main}"
    else
      echo "keep existing hf ref ${refs_main}"
    fi
  else
    echo "keep existing hf ref ${refs_main}"
  fi
}

echo "WorldFoundry checkpoint root: ${CKPT_DIR}"
echo "WorldFoundry HFD root: ${HFD_ROOT}"
echo "Hugging Face hub cache: ${HF_HUB_CACHE_VALUE}"

for spec in "${REPO_SPECS[@]}"; do
  link_one_repo "$spec"
done
