#!/usr/bin/env bash
# Tier A: eval-only sweep (no retraining) — sample-steps × guidance-scale grid.
#
# Works in genai_test_3 and can be copied to genai_test_2 / Quest unchanged.
#
# Usage:
#   cd /path/to/genai_test_clone
#   CKPT=checkpoints/ckpt_best.pt bash scripts/run_tier_a_eval_grid.sh
#
# Env:
#   CKPT              — required; path to ckpt_best.pt (absolute paths OK).
#   GENAI_ROOT        — default: parent of scripts/.
#   DATA_ROOT         — default: $GENAI_ROOT/data (override when CelebA lives elsewhere).
#   EVAL_ROOT         — repo whose src/eval_test.py should run. Default: $GENAI_ROOT.
#                       Set to old tree for old checkpoints (e.g. /home/.../genai_test_2).
#   PYTHON            — interpreter to run (default: python). Examples:
#                           PYTHON=python3
#                           PYTHON=/path/to/.venv/bin/python
#   GRID_SUFFIX       — output bucket name eval_grid_tier_a_<GRID_SUFFIX>; if unset:
#                           basename of ckpt directory (often "checkpoints" — pass
#                           GRID_SUFFIX=genai_test_2 to avoid collisions.)
#   FID=1             — add --fid per cell (much slower).
#   MAX_TEST          — if set → --max-test-images (skipped in legacy eval mode.)
#   TIER_A_LEGACY_EVAL — set to 1 when using older src/eval_test.py with no CLI for
#                           --batch-size, --workers, or --max-test-images (Quest genai_test_2).
#
# Recommended on Quest genai_test_2:
#   module load cuda ...   # site-specific
#   conda activate YOUR_ENV
#   cd /home/.../genai_test_2
#   TIER_A_LEGACY_EVAL=1 GRID_SUFFIX=genai_test_2 \
#   CKPT=/path/to/ckpt_best.pt DATA_ROOT=/path/to/celeba-root \
#   bash scripts/run_tier_a_eval_grid.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENAI_ROOT="$(cd "${GENAI_ROOT:-$SCRIPT_DIR/..}" && pwd)"
DATA_ROOT="${DATA_ROOT:-$GENAI_ROOT/data}"
EVAL_ROOT="${EVAL_ROOT:-$GENAI_ROOT}"
PYTHON="${PYTHON:-python}"

if [[ -z "${CKPT:-}" ]]; then
  echo "Set CKPT to your ckpt_best.pt e.g.:" >&2
  echo "  CKPT=/path/to/ckpt_best.pt bash scripts/run_tier_a_eval_grid.sh" >&2
  exit 1
fi
if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  exit 1
fi

cd "$GENAI_ROOT"

GRID_SUFFIX="${GRID_SUFFIX:-}"
if [[ -z "${GRID_SUFFIX}" ]]; then
  GRID_SUFFIX="$(basename "$(cd "$(dirname "$CKPT")" && pwd)")"
fi
GRID_ROOT="${GENAI_ROOT}/checkpoints/eval_grid_tier_a_${GRID_SUFFIX}"
mkdir -p "$GRID_ROOT"
CSV="${GRID_ROOT}/tier_a_aggregate.csv"
echo "steps,guidance_scale,test_ddpm_loss,test_psnr_db,out_dir,ckpt" >"$CSV"

FID_FLAG=()
if [[ "${FID:-}" == "1" ]]; then
  FID_FLAG=(--fid)
fi

EXTRA=( )
if [[ "${TIER_A_LEGACY_EVAL:-}" == "1" ]]; then
  echo "[tier_a] TIER_A_LEGACY_EVAL=1: omitting --batch-size/--workers/--max-test-images for older eval_test.py" >&2
  if [[ -n "${MAX_TEST:-}" ]]; then
    echo "[tier_a] warning: MAX_TEST is set but unsupported in legacy mode; ignoring." >&2
  fi
else
  BATCH_SIZE="${BATCH_SIZE:-24}"
  WORKERS="${WORKERS:-4}"
  EXTRA=(--batch-size "$BATCH_SIZE" --workers "$WORKERS")
  if [[ -n "${MAX_TEST:-}" ]]; then
    EXTRA+=(--max-test-images "$MAX_TEST")
  fi
fi

echo "[tier_a] GENAI_ROOT=$GENAI_ROOT DATA_ROOT=$DATA_ROOT PYTHON=$PYTHON" >&2
echo "[tier_a] EVAL_ROOT=$EVAL_ROOT" >&2
echo "[tier_a] outputs under $GRID_ROOT" >&2

for steps in 200 300 400; do
  for gs in 1.0 1.2 1.35 1.5; do
    tag="steps${steps}_gs${gs/./_}"
    out="${GRID_ROOT}/${tag}"
    mkdir -p "$out"
    echo "[tier_a] sample_steps=${steps} guidance_scale=${gs} -> ${out}" >&2

    (
      cd "$EVAL_ROOT"
      "$PYTHON" -m src.eval_test \
      --ckpt "$CKPT" \
      --data-root "$DATA_ROOT" \
      --out-dir "$out" \
      --sample-steps "$steps" \
      --guidance-scale "$gs" \
      "${EXTRA[@]}" \
      "${FID_FLAG[@]}" \
      --triplet-png "${out}/triplets.png"
    )

    jj="${out}/test_eval_summary.json"
    loss="$("$PYTHON" -c "import json; print(json.load(open(\"${jj}\"))[\"test_ddpm_loss\"])")"
    psnr="$("$PYTHON" -c "import json; d=json.load(open(\"${jj}\")); print(d.get(\"test_psnr_db\", \"\"))")"
    printf '%s,%s,%s,%s,"%s","%s"\n' "$steps" "$gs" "$loss" "$psnr" "$out" "$CKPT" >>"$CSV"
  done
done

echo "[tier_a] done. Aggregate: ${CSV}" >&2
