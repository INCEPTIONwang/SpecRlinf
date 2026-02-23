#!/usr/bin/env bash
set -euo pipefail

EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "$EMBODIED_PATH")")"
EVAL_SCRIPT="${EMBODIED_PATH}/eval_embodiment.sh"

# Optional args:
#   $1 config name (default: libero_spatial_grpo_openpi_pi05-15sp)
#   $2 number of runs (default: 12)
#   $3 base seed for threshold sampling (default: 1234)
#   $4 robot platform (default: LIBERO)
CONFIG_NAME="${1:-libero_spatial_grpo_openpi_pi05-15sp}"
NUM_RUNS="${2:-12}"
BASE_SEED="${3:-1234}"
ROBOT_PLATFORM_ARG="${4:-LIBERO}"

# Keep same runtime env style as existing embodiment scripts.
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export HYDRA_FULL_ERROR=1

echo "Config=${CONFIG_NAME}, runs=${NUM_RUNS}, base_seed=${BASE_SEED}, robot_platform=${ROBOT_PLATFORM_ARG}"

STAMP="$(date +'%Y%m%d-%H%M%S')"
SWEEP_ROOT="${REPO_PATH}/logs/spec_delta_sweep_${CONFIG_NAME}_${STAMP}"
mkdir -p "${SWEEP_ROOT}"

CSV_FILE="${SWEEP_ROOT}/spec_delta_sweep.csv"
echo "run_idx,spec_delta_thresholds" > "${CSV_FILE}"

for ((i=0; i<NUM_RUNS; i++)); do
  THRESHOLDS="$(python - "$BASE_SEED" "$i" <<'PY'
import random
import sys

base_seed = int(sys.argv[1])
run_idx = int(sys.argv[2])

center = [0.2, 0.2, 0.2, 0.03, 0.03, 0.03]
radius = [0.04, 0.04, 0.04, 0.01, 0.01, 0.01]

# Keep run 0 as the exact center point for a clean baseline.
if run_idx == 0:
    vals = center
else:
    rng = random.Random(base_seed + run_idx)
    vals = []
    for c, r in zip(center, radius):
        v = c + rng.uniform(-r, r)
        vals.append(max(1e-4, v))

print(",".join(f"{v:.5f}" for v in vals))
PY
)"

  RUN_NAME="run_$(printf '%02d' "$i")"
  LOG_DIR="${SWEEP_ROOT}/${RUN_NAME}"
  mkdir -p "${LOG_DIR}"

  echo "${i},\"[${THRESHOLDS}]\"" >> "${CSV_FILE}"
  echo "[$(date +'%F %T')] ${RUN_NAME} thresholds=[${THRESHOLDS}]"

  bash "${EVAL_SCRIPT}" "${CONFIG_NAME}" "${ROBOT_PLATFORM_ARG}" \
    "runner.logger.log_path=${LOG_DIR}" \
    "runner.logger.experiment_name=${CONFIG_NAME}_${RUN_NAME}" \
    "actor.model.openpi.spec_delta_thresholds=[${THRESHOLDS}]" \
    2>&1 | tee "${LOG_DIR}/sweep_wrapper.log"
done

echo "Sweep completed. Root: ${SWEEP_ROOT}"
echo "Threshold table: ${CSV_FILE}"
