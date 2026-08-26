#!/usr/bin/env bash
# Compile (or drift-check) a per-CUDA-tier lockfile for
# requirements/worldfoundry-unified.txt (plan I-05).
#
# Lock bodies are never invented in-repo: this script is the only sanctioned
# way to (re)generate them, on a host that can reach PyPI and the PyTorch CUDA
# wheel index. The resolve is constrained by the per-tier torch stubs in
# requirements/cuda/<tier>-torch.txt so a lock can never drift outside the
# I-03 TIER_TORCH_SPECS matrix.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UV_BIN="${UV:-uv}"

usage() {
  cat <<'EOF'
Usage: bash scripts/setup/compile_unified_lock.sh <cu121|cu124|cu128> [--check]

Modes:
  (default)  Compile requirements/lock/worldfoundry-unified.<tier>.lock.txt
             with `uv pip compile`, constrained by requirements/cuda/<tier>-torch.txt.
  --check    Recompile to a temp file and diff against the committed lock.
             Placeholder locks (comment-only) skip with a notice and exit 0,
             so CI can wire this up before real locks are committed.

Requires uv on PATH (or set UV=/path/to/uv) for compiles and non-placeholder
checks. Environment overrides: WORLDFOUNDRY_TORCH_INDEX_URL,
WORLDFOUNDRY_PYPI_INDEX_URL.
EOF
}

TIER=""
CHECK_MODE=0
while (($#)); do
  case "$1" in
    cu121|cu124|cu128)
      TIER="$1"
      shift
      ;;
    --check)
      CHECK_MODE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unsupported argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$TIER" ]]; then
  usage >&2
  exit 2
fi

LOCK_PATH="${ROOT}/requirements/lock/worldfoundry-unified.${TIER}.lock.txt"
TIER_CONSTRAINTS="${ROOT}/requirements/cuda/${TIER}-torch.txt"
INDEX_URL="${WORLDFOUNDRY_TORCH_INDEX_URL:-https://download.pytorch.org/whl/${TIER}}"
PYPI_URL="${WORLDFOUNDRY_PYPI_INDEX_URL:-https://pypi.org/simple}"

if [[ ! -f "$TIER_CONSTRAINTS" ]]; then
  echo "Missing I-03 tier constraint stub: ${TIER_CONSTRAINTS}" >&2
  exit 1
fi

lock_is_populated() {
  # A lock counts as populated once it carries any non-comment requirement line.
  [[ -f "$1" ]] && grep -Eq '^[[:space:]]*[^#[:space:]]' "$1"
}

if [[ "$CHECK_MODE" == "1" ]] && ! lock_is_populated "$LOCK_PATH"; then
  echo "NOTICE: ${LOCK_PATH#"${ROOT}"/} is a placeholder (no resolved pins yet); nothing to drift-check."
  echo "Populate it with: make lock-unified TIER=${TIER}"
  exit 0
fi

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  echo "uv is required to compile/check lockfiles. Install uv or set UV=/path/to/uv." >&2
  exit 1
fi

compile_lock() {
  local out="$1"
  {
    printf '%s\n' \
      "# WorldFoundry unified lock for CUDA tier ${TIER} (plan I-05)." \
      "# Regenerate with: make lock-unified TIER=${TIER}" \
      "# --index-url ${INDEX_URL}" \
      "# --extra-index-url ${PYPI_URL}" \
      "# --constraint requirements/cuda/${TIER}-torch.txt (I-03 TIER_TORCH_SPECS)" \
      "#"
    "$UV_BIN" pip compile \
      --quiet \
      --python-version 3.11 \
      --index-url "$INDEX_URL" \
      --extra-index-url "$PYPI_URL" \
      --index-strategy unsafe-best-match \
      --constraint "$TIER_CONSTRAINTS" \
      --emit-index-url \
      --no-strip-extras \
      --custom-compile-command "make lock-unified TIER=${TIER}" \
      "${ROOT}/requirements/worldfoundry-unified.txt"
  } >"$out"
}

tmp="$(mktemp "${TMPDIR:-/tmp}/worldfoundry-lock-${TIER}.XXXXXX")"
trap 'rm -f "${tmp}"' EXIT

echo "Resolving ${TIER} lock (torch index: ${INDEX_URL})"
compile_lock "$tmp"

if [[ "$CHECK_MODE" == "1" ]]; then
  if diff -u "$LOCK_PATH" "$tmp"; then
    echo "OK: ${LOCK_PATH#"${ROOT}"/} matches a fresh resolve."
    exit 0
  fi
  echo "Lock drift detected for ${TIER}. Regenerate with: make lock-unified TIER=${TIER}" >&2
  exit 1
fi

mv "$tmp" "$LOCK_PATH"
trap - EXIT
echo "Wrote ${LOCK_PATH}"
