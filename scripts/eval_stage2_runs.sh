#!/usr/bin/env bash
# Evaluate Stage 2 run checkpoints with Tier A sweep.
#
# Usage:
#   cd /home/szb9536/genai_test_4
#   DATA_ROOT=/home/szb9536/genai_test_4/data bash scripts/eval_stage2_runs.sh
#
# Runs list must match notebooks/stage2_hyperparam_search.ipynb RUN_MATRIX_STAGE_A names.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
RUNS=(
  "H00_baseline"
  "H01_lr_low"
  "H02_lr_high"
  "H03_batch_16"
  "H04_epochs_plus"
  "H05_drop_sketch_high"
  "H06_guidance_high"
  "H07_sample_steps_fast"
  "H08_lpips_late"
  "H09_color_strong"
  "H10_linear_schedule"
  "H11_min_snr_5"
)

for r in "${RUNS[@]}"; do
  ckpt="${PROJECT_ROOT}/checkpoints/hp_${r}/ckpt_best.pt"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[skip] missing ${ckpt}"
    continue
  fi
  echo "[eval] ${r}"
  CKPT="${ckpt}" \
  GENAI_ROOT="${PROJECT_ROOT}" \
  EVAL_ROOT="${PROJECT_ROOT}" \
  DATA_ROOT="${DATA_ROOT}" \
  GRID_SUFFIX="hp_${r}" \
  bash "${PROJECT_ROOT}/scripts/run_tier_a_eval_grid.sh"
done
