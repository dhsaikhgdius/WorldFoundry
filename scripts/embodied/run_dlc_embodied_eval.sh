#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 CONFIG [OUTPUT_DIR]" >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

CONFIG=$1
OUTPUT_DIR=${2:-tmp/embodied_dlc/run}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${WORLDFOUNDRY_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
PYTHON_BIN=${WF_EMBODIED_PYTHON:-python}
CONDA_ENV=${WF_EMBODIED_CONDA_ENV:-}
BOOTSTRAP=${WF_EMBODIED_BOOTSTRAP:-0}
BOOTSTRAP_PACKAGES=${WF_EMBODIED_BOOTSTRAP_PACKAGES:-pyyaml msgpack packaging tqdm websockets}
SERVER_URL=${WF_EMBODIED_SERVER_URL:-}
SERVE_CONFIG=${WF_EMBODIED_SERVE_CONFIG:-}
# Default to loopback; non-loopback requires WF_EMBODIED_SERVER_TOKEN in model_server.
SERVE_HOST=${WF_EMBODIED_SERVE_HOST:-127.0.0.1}
SERVE_PORT=${WF_EMBODIED_SERVE_PORT:-8000}
READY_TIMEOUT=${WF_EMBODIED_SERVE_READY_TIMEOUT:-1800}
PLAN_ONLY=${WF_EMBODIED_PLAN_ONLY:-0}

cd "${REPO_ROOT}"
export WORLDFOUNDRY_REPO_ROOT="${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

if [[ -n "${CONDA_ENV}" && -n "$(command -v conda || true)" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

if [[ "${BOOTSTRAP}" == "1" ]]; then
  read -r -a BOOTSTRAP_PACKAGE_ARGS <<< "${BOOTSTRAP_PACKAGES}"
  if [[ ${#BOOTSTRAP_PACKAGE_ARGS[@]} -gt 0 ]]; then
    "${PYTHON_BIN}" -m pip install --no-cache-dir "${BOOTSTRAP_PACKAGE_ARGS[@]}"
  fi
fi

echo "WORLDFOUNDRY_REPO_ROOT=${WORLDFOUNDRY_REPO_ROOT}"
echo "CONFIG=${CONFIG}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "BOOTSTRAP=${BOOTSTRAP}"
echo "MASTER_ADDR=${MASTER_ADDR:-}"
echo "MASTER_PORT=${MASTER_PORT:-}"
echo "WORLD_SIZE=${WORLD_SIZE:-}"
echo "RANK=${RANK:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L
fi

SERVER_PID=
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    # Kill the process group started via setsid when available.
    if kill -0 -- "-${SERVER_PID}" >/dev/null 2>&1; then
      kill -- "-${SERVER_PID}" >/dev/null 2>&1 || true
      wait "${SERVER_PID}" >/dev/null 2>&1 || true
    else
      kill "${SERVER_PID}" >/dev/null 2>&1 || true
      wait "${SERVER_PID}" >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup EXIT

if [[ -n "${SERVE_CONFIG}" ]]; then
  if command -v setsid >/dev/null 2>&1; then
    setsid "${PYTHON_BIN}" -m worldfoundry.evaluation.tasks.embodied.model_server.serve \
      --config "${SERVE_CONFIG}" \
      --host "${SERVE_HOST}" \
      --port "${SERVE_PORT}" &
  else
    "${PYTHON_BIN}" -m worldfoundry.evaluation.tasks.embodied.model_server.serve \
      --config "${SERVE_CONFIG}" \
      --host "${SERVE_HOST}" \
      --port "${SERVE_PORT}" &
  fi
  SERVER_PID=$!
  SERVER_URL=${SERVER_URL:-ws://127.0.0.1:${SERVE_PORT}}
  # TCP connect readiness; protocol HELLO ping is a follow-up improvement.
  "${PYTHON_BIN}" - "${SERVE_PORT}" "${READY_TIMEOUT}" <<'PY'
import socket
import sys
import time

port = int(sys.argv[1])
deadline = time.time() + int(sys.argv[2])
while time.time() < deadline:
    with socket.socket() as sock:
        sock.settimeout(2)
        try:
            sock.connect(("127.0.0.1", port))
            raise SystemExit(0)
        except OSError:
            time.sleep(2)
raise SystemExit(f"server did not open port {port}")
PY
fi

mkdir -p "${OUTPUT_DIR}"
"${PYTHON_BIN}" - "${CONFIG}" "${OUTPUT_DIR}" "${SERVER_URL}" <<'PY'
import json
import sys
from pathlib import Path

from worldfoundry.evaluation.tasks.embodied.config_loader import load_canonical_embodied_config
from worldfoundry.evaluation.tasks.embodied.materialize_rollouts import materialize_embodied_rollout_requests

config = load_canonical_embodied_config(sys.argv[1], output_dir=sys.argv[2], server_url=sys.argv[3] or None)
output_dir = Path(sys.argv[2]).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
benchmarks = []
request_count = 0
for bench_cfg in config.get("benchmarks") or ():
    requests = materialize_embodied_rollout_requests(bench_cfg)
    request_count += len(requests)
    benchmarks.append(
        {
            "id": str(bench_cfg.get("id") or bench_cfg.get("benchmark_id") or "benchmark"),
            "benchmark_id": str(bench_cfg.get("benchmark_id") or bench_cfg.get("id") or "libero"),
            "request_count": len(requests),
        }
    )
payload = {
    "schema_version": "worldfoundry-embodied-eval-plan",
    "config_path": str(Path(sys.argv[1]).resolve()),
    "output_dir": str(output_dir),
    "model_id": str((config.get("model") or {}).get("id") or config.get("model_id") or "openvla"),
    "server_url": sys.argv[3] or None,
    "benchmark_count": len(benchmarks),
    "request_count": request_count,
    "benchmarks": benchmarks,
}
(output_dir / "embodied_plan.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY

if [[ "${PLAN_ONLY}" == "1" ]]; then
  exit 0
fi

"${PYTHON_BIN}" - "${CONFIG}" "${OUTPUT_DIR}" "${SERVER_URL}" "${WF_EMBODIED_SHARD_ID:-}" "${WF_EMBODIED_NUM_SHARDS:-}" "${WF_EMBODIED_EVAL_ID:-}" "${WF_EMBODIED_NO_SAVE:-0}" <<'PY'
import json
import sys

from worldfoundry.evaluation.tasks.embodied.orchestrator import run_embodied_eval_config


def _optional_int(value: str):
    return int(value) if value else None


result = run_embodied_eval_config(
    sys.argv[1],
    output_dir=sys.argv[2],
    server_url=sys.argv[3] or None,
    shard_id=_optional_int(sys.argv[4]),
    num_shards=_optional_int(sys.argv[5]),
    eval_id=sys.argv[6] or None,
    no_docker=True,
    no_save=sys.argv[7] == "1",
)
print(json.dumps(result.to_dict(), sort_keys=True))
raise SystemExit(int(result.evaluate_result.exit_code))
PY
