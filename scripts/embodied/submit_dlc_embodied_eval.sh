#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: submit_dlc_embodied_eval.sh CONFIG [OUTPUT_DIR]

Builds a PAI DLC pytorchjob that runs scripts/embodied/run_dlc_embodied_eval.sh
for the given embodied eval config. Set WF_DLC_SUBMIT=1 to submit; without it,
the script only prints the DLC command (dry run).

Required environment variables (account/network scoped, no public defaults):
  WF_DLC_RESOURCE_ID        DLC resource quota id
  WF_DLC_WORKSPACE_ID       PAI workspace id
  WF_DLC_VPC_ID             VPC id for the job network
  WF_DLC_SWITCH_ID          vSwitch id
  WF_DLC_SECURITY_GROUP_ID  Security group id
  WF_DLC_EXTENDED_CIDRS     Comma-separated extended CIDR list

Optional overrides:
  WF_DLC_JOB_NAME       Job name (default: wf-embodied-<config-id>-<UTC timestamp>)
  WF_DLC_WORKER_IMAGE   PAI-accessible worker image (default: docker.image from CONFIG)
  DLC_BIN               Path to the dlc CLI (default: /etc/dsw/runtime/export_bin/dlc
                        when present, otherwise `dlc` found on PATH)
  WF_DLC_PRIORITY, WF_DLC_WORKERS, WF_DLC_WORKER_CPU, WF_DLC_WORKER_MEMORY,
  WF_DLC_WORKER_SHARED_MEMORY, WF_DLC_WORKER_GPU, WF_DLC_MAX_RUNNING_MINUTES
                        Job sizing. Defaults are example machine-shape values;
                        review them before submitting.

Registry credentials:
  Prefer configuring credentials once via `dlc config` so this script never
  handles secrets. If you must submit inline against a private registry, set
  both WF_DLC_IMAGE_REPO_USERNAME and WF_DLC_IMAGE_REPO_PASSWORD.
  WARNING: the dlc CLI receives the password as a process argument (argv),
  which other local users can read via /proc/<pid>/cmdline. This script only
  redacts the password in its own printed copy of the command.

Migration note (2026-08): the resource/workspace/VPC/vSwitch/security-group/CIDR
values were previously hardcoded account-level defaults and are now required
environment variables -- export the values for YOUR account before running.
The default job name changed from sft_innovator_science_data to
wf-embodied-<config-id>-<UTC timestamp>.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${WORLDFOUNDRY_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
CONFIG=$1
if [[ "${CONFIG}" != /* ]]; then
  CONFIG="${REPO_ROOT}/${CONFIG}"
fi
OUTPUT_DIR=${2:-${REPO_ROOT}/tmp/embodied_dlc/$(basename "${CONFIG}" .yaml)-$(date -u +%Y%m%dT%H%M%SZ)}

# DLC CLI resolution: explicit DLC_BIN override, then the DSW runtime path,
# then whatever `dlc` is on PATH.
if [[ -z "${DLC_BIN:-}" ]]; then
  if [[ -x /etc/dsw/runtime/export_bin/dlc ]]; then
    DLC_BIN=/etc/dsw/runtime/export_bin/dlc
  else
    DLC_BIN=$(command -v dlc || true)
  fi
fi
if [[ -z "${DLC_BIN}" || ! -x "${DLC_BIN}" ]]; then
  DLC_BIN_MSG="dlc CLI not found or not executable (resolved DLC_BIN='${DLC_BIN}'). Set DLC_BIN or install dlc on PATH."
  if [[ "${WF_DLC_SUBMIT:-0}" == "1" ]]; then
    echo "${DLC_BIN_MSG}" >&2
    exit 2
  fi
  echo "Warning: ${DLC_BIN_MSG} Continuing because this is a dry run (WF_DLC_SUBMIT != 1)." >&2
  DLC_BIN=${DLC_BIN:-dlc}
fi

# Account/network-scoped settings must come from the environment. There are
# intentionally no defaults so a misconfigured run fails locally instead of
# submitting to someone else's cloud quota/VPC.
REQUIRED_DLC_ENV_VARS=(
  WF_DLC_RESOURCE_ID
  WF_DLC_WORKSPACE_ID
  WF_DLC_VPC_ID
  WF_DLC_SWITCH_ID
  WF_DLC_SECURITY_GROUP_ID
  WF_DLC_EXTENDED_CIDRS
)
MISSING_DLC_ENV_VARS=()
for _var in "${REQUIRED_DLC_ENV_VARS[@]}"; do
  [[ -n "${!_var:-}" ]] || MISSING_DLC_ENV_VARS+=("${_var}")
done
if (( ${#MISSING_DLC_ENV_VARS[@]} > 0 )); then
  {
    echo "Missing required environment variable(s): ${MISSING_DLC_ENV_VARS[*]}"
    echo "These are account/network scoped and have no public defaults."
    echo "Export the values for your DLC account, then re-run."
    echo
  } >&2
  usage
  exit 2
fi

readarray -t CONFIG_VALUES < <(
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 - "${CONFIG}" <<'PY'
import sys
from worldfoundry.evaluation.tasks.embodied.config_loader import (
    load_canonical_embodied_config,
    load_embodied_config,
)

config = load_canonical_embodied_config(sys.argv[1])
# The canonical loader injects a placeholder id; read the raw payload so the
# shell side can fall back to the config file stem when no real id is set.
raw = load_embodied_config(sys.argv[1])
docker = config.get("docker") or {}
print(docker.get("image") or "")
print(docker.get("source_image") or "")
print(docker.get("python_env") or "")
print(raw.get("id") or raw.get("name") or "")
PY
)

CONFIG_IMAGE=${CONFIG_VALUES[0]}
CONFIG_SOURCE_IMAGE=${CONFIG_VALUES[1]}
CONFIG_CONDA_ENV=${CONFIG_VALUES[2]}
CONFIG_ID=${CONFIG_VALUES[3]}
if [[ -z "${CONFIG_ID}" ]]; then
  CONFIG_ID=$(basename "${CONFIG}")
  CONFIG_ID=${CONFIG_ID%.*}
fi
if [[ "${WF_DLC_USE_SOURCE_IMAGE:-0}" == "1" ]]; then
  DEFAULT_WORKER_IMAGE=${CONFIG_SOURCE_IMAGE:-${CONFIG_IMAGE}}
else
  DEFAULT_WORKER_IMAGE=${CONFIG_IMAGE}
fi
WORKER_IMAGE=${WF_DLC_WORKER_IMAGE:-${DEFAULT_WORKER_IMAGE}}
CONDA_ENV=${WF_EMBODIED_CONDA_ENV-${CONFIG_CONDA_ENV}}
# Keep the job name to characters DLC accepts regardless of config id contents.
DEFAULT_JOB_NAME=$(printf 'wf-embodied-%s-%s' "${CONFIG_ID}" "$(date -u +%Y%m%dT%H%M%SZ)" | tr -c 'a-zA-Z0-9._-' '-')
JOB_NAME=${WF_DLC_JOB_NAME:-${DEFAULT_JOB_NAME}}

if [[ -z "${WORKER_IMAGE}" ]]; then
  echo "No worker image found. Set WF_DLC_WORKER_IMAGE or add docker.image to the config." >&2
  exit 2
fi

INNER_COMMAND=$(cat <<EOF
set -euo pipefail
cd "${REPO_ROOT}"
export WORLDFOUNDRY_REPO_ROOT="${REPO_ROOT}"
export WF_EMBODIED_CONDA_ENV="${CONDA_ENV}"
export WF_EMBODIED_SERVER_URL="${WF_EMBODIED_SERVER_URL:-}"
export WF_EMBODIED_SERVE_CONFIG="${WF_EMBODIED_SERVE_CONFIG:-}"
export WF_EMBODIED_SERVE_PORT="${WF_EMBODIED_SERVE_PORT:-8000}"
export WF_EMBODIED_PLAN_ONLY="${WF_EMBODIED_PLAN_ONLY:-0}"
export WF_EMBODIED_NO_SAVE="${WF_EMBODIED_NO_SAVE:-0}"
export WF_EMBODIED_BOOTSTRAP="${WF_EMBODIED_BOOTSTRAP:-0}"
export WF_EMBODIED_BOOTSTRAP_PACKAGES="${WF_EMBODIED_BOOTSTRAP_PACKAGES:-pyyaml msgpack packaging tqdm websockets}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
bash scripts/embodied/run_dlc_embodied_eval.sh "${CONFIG}" "${OUTPUT_DIR}"
EOF
)
COMMAND=$(cat <<EOF
bash <<'WF_DLC_COMMAND'
${INNER_COMMAND}
WF_DLC_COMMAND
EOF
)

CMD=(
  "${DLC_BIN}" submit pytorchjob
  --name="${JOB_NAME}"
  --command="${COMMAND}"
  --resource_id="${WF_DLC_RESOURCE_ID}"
  --workspace_id="${WF_DLC_WORKSPACE_ID}"
  --vpc_id="${WF_DLC_VPC_ID}"
  --switch_id="${WF_DLC_SWITCH_ID}"
  --security_group_id="${WF_DLC_SECURITY_GROUP_ID}"
  --extended_cidrs="${WF_DLC_EXTENDED_CIDRS}"
  --priority="${WF_DLC_PRIORITY:-1}"
  --workers="${WF_DLC_WORKERS:-1}"
  --worker_image="${WORKER_IMAGE}"
  --worker_cpu="${WF_DLC_WORKER_CPU:-116}"
  --worker_memory="${WF_DLC_WORKER_MEMORY:-1800Gi}"
  --worker_shared_memory="${WF_DLC_WORKER_SHARED_MEMORY:-1800Gi}"
  --worker_gpu="${WF_DLC_WORKER_GPU:-8}"
  --job_max_running_time_minutes="${WF_DLC_MAX_RUNNING_MINUTES:-1440}"
)

if [[ -n "${WF_DLC_DATA_SOURCE_URIS:-}" ]]; then
  CMD+=(--data_source_uris="${WF_DLC_DATA_SOURCE_URIS}")
fi

# Registry credentials: prefer `dlc config` so no secret ever reaches this
# script. When passed inline, the dlc CLI gets the password via argv, which is
# visible to other local users through /proc/<pid>/cmdline; we can only redact
# the printed copy below.
if [[ -n "${WF_DLC_IMAGE_REPO_USERNAME:-}" || -n "${WF_DLC_IMAGE_REPO_PASSWORD:-}" ]]; then
  if [[ -z "${WF_DLC_IMAGE_REPO_USERNAME:-}" || -z "${WF_DLC_IMAGE_REPO_PASSWORD:-}" ]]; then
    echo "Set both WF_DLC_IMAGE_REPO_USERNAME and WF_DLC_IMAGE_REPO_PASSWORD for private image registries." >&2
    exit 2
  fi
  echo "Warning: passing the registry password on the dlc argv (visible via /proc/<pid>/cmdline). Prefer 'dlc config'." >&2
  CMD+=(--image_repo_username="${WF_DLC_IMAGE_REPO_USERNAME}" --image_repo_password="${WF_DLC_IMAGE_REPO_PASSWORD}")
fi

PRINT_CMD=("${CMD[@]}")
for i in "${!PRINT_CMD[@]}"; do
  if [[ "${PRINT_CMD[$i]}" == --image_repo_password=* ]]; then
    PRINT_CMD[$i]="--image_repo_password=REDACTED"
  fi
done

printf 'DLC job name: %s\n' "${JOB_NAME}"
printf 'DLC worker image: %s\n' "${WORKER_IMAGE}"
printf 'DLC conda env: %s\n' "${CONDA_ENV}"
printf 'DLC output dir: %s\n' "${OUTPUT_DIR}"
printf '+'
printf ' %q' "${PRINT_CMD[@]}"
printf '\n'

if [[ "${WF_DLC_SUBMIT:-0}" == "1" ]]; then
  exec "${CMD[@]}"
fi
