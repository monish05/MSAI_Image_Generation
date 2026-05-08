# MSAI_Image_Generation Devlog

Short version of what we did, what failed, and what worked.

- Main docs: [`../README.md`](../README.md)
- Hyperparameter matrix: [`./hyperparam_run_matrix.md`](./hyperparam_run_matrix.md)

## One-minute summary

- Started with **Sketchy** (human sketches -> photos).
- Tried many training improvements (EMA, CFG dropout, cosine schedule, Min-SNR, LPIPS, wider UNet, distributed training).
- Quality still poor because **human sketches were noisy/misaligned** and class diversity was high.
- Pivoted to **CelebA + synthetic sketches** (generated from each photo). Big quality jump.
- Built a second iteration to improve:
  - cleaner conditioning signals
  - color faithfulness (Lab color loss)
  - repeatable hyperparameter search workflow

---

## Timeline at a glance

| Window | Phase | Outcome |
|---|---|---|
| Apr 26 -> Apr 30 | Sketchy era (`38d7657` -> `b7aac8b`) | Better training stack, weak visuals |
| May 2 | CelebA pivot (`b698b35`) | Major improvement from aligned conditioning |
| Post-pivot | CelebA runs (first iteration) | Multiple checkpoint runs + Tier-A sweeps |
| Next | Second iteration | Color-loss + Stage-2 search |

---

## Phase 1 - Bootstrap

### `95dc356` Initial commit

- Placeholder README only.
- No model/data pipeline yet.

---

## Phase 2 - Sketchy era (why it failed)

### Why Sketchy was chosen

Sketchy matched the original goal (structure-guided generation across many categories), so it was a reasonable first choice.  
Human-sketch dataset source: [Kaggle - Sketch to Image (ankitsheoran23)](https://www.kaggle.com/datasets/ankitsheoran23/sketch-to-image).

### Major things we tried

- Built full training/eval stack for Sketchy.
- Fixed dataset paths and split leakage issues.
- Added sampling + eval scripts.
- Added modern training knobs:
  - EMA
  - CFG dropout
  - cosine LR + warmup
  - Min-SNR weighting
  - LPIPS aux loss
  - wider UNet
- Tried distributed training (`torchrun`) for throughput.

### Why quality stayed bad

1. **Too much category diversity** for available per-class data density.
2. **Human sketch mismatch** (geometry/style drift from target photos).
3. **More compute helped speed, not information quality**.

### Visual evidence (human sketch failure)

This example shows how weak the human-sketch conditioning results were:

![Human sketch to real: poor results](../assets/human_sketch_to_real.png)

---

## Phase 3 - Pivot to CelebA (`b698b35`)

### What changed

Sketchy-specific files removed, CelebA pipeline added:

- Added: `src/celeba.py`, `src/sketch.py`, `src/train.py`, `src/unet.py`
- Removed: `build_sketchy_splits.py`, `sketchy_dataset.py`, `train_ddpm.py`, `unet_conditional.py`, and old eval/sample scripts

### Why this worked better

- CelebA is single-domain and aligned.
- Synthetic sketches are generated from the exact photo (strong alignment).
- Existing training improvements became effective once conditioning noise dropped.

---

## Phase 4 - CelebA iteration (first iteration)

### Runs used for comparison

- `checkpoints/`
- `checkpoints_2/`
- `checkpoints_3/`

### Eval/training workflow

- Tier-A sweep script
- Quest SLURM trainer

### What improved and what remained

- 128x128 deeper UNet runs clearly outperformed early 64x64 runs.
- Remaining issues before the second iteration:
  - background clutter leaking into sketch signal
  - color drift (skin/hair) despite LPIPS

---

## Phase 5 - Second CelebA iteration

Focus: clean conditioning + better color realism + structured run matrix.

### New ingredients

1. **Color-faithfulness loss**
   - `color_ab_l1` in [`../src/train.py`](../src/train.py)
   - Flags: `--color-loss-weight`, `--color-loss-start-frac`, `--color-loss-ramp-steps`
   - Not present in the first-iteration `train.py`

2. **Two-stage execution**
   - Stage 1 baseline: [`../slurm/quest_stage1_baseline.sh`](../slurm/quest_stage1_baseline.sh)
   - Stage 2 search: [`../slurm/quest_stage2_hyperparam_search.sh`](../slurm/quest_stage2_hyperparam_search.sh)

3. **Per-case + analysis notebooks**
   - [`../notebooks/stage2_cases/`](../notebooks/stage2_cases/)

4. **Run matrix + rubric**
   - [`./hyperparam_run_matrix.md`](./hyperparam_run_matrix.md)

---

## Stage 2 run matrix (quick reference)

Overrides from [`./hyperparam_run_matrix.md`](./hyperparam_run_matrix.md):

- `H00_baseline`: none
- `H01_lr_low`: `--lr 1e-4`
- `H02_lr_high`: `--lr 3e-4`
- `H03_batch_16`: `--batch-size 16`
- `H04_epochs_plus`: `--epochs 55`
- `H05_drop_sketch_high`: `--drop-sketch-prob 0.2`
- `H06_guidance_high`: `--guidance-scale 2.0`
- `H07_sample_steps_fast`: `--sample-steps 120`
- `H08_lpips_late`: `--lpips-start-frac 0.25`
- `H09_color_strong`: `--color-loss-weight 0.05`, `--color-loss-start-frac 0.45`
- `H10_linear_schedule`: `--beta-schedule linear`
- `H11_min_snr_5`: `--min-snr-gamma 5`

Shared defaults currently documented there include:

`--image-size 64 --batch-size 32 --epochs 75 --lr 2e-4 --seed 42 --timesteps 1000 --beta-schedule cosine --min-snr-gamma 0 --guidance-scale 1.5 --sample-steps 200 --sample-every 2000 --drop-sketch-prob 0.1 --lpips-start-frac 0.1 --color-loss-weight 0.02 --color-loss-start-frac 0.6 --color-loss-ramp-steps 5000 --early-stop-patience 0 --early-stop-min-delta 0 --max-train-images 70000 --amp`

---

## Hyperparameter results (fill when run finishes)

<!-- TODO: Fill placeholders after final run analysis -->

### Stage 1 baseline

Source: [`../checkpoints/stage1_baseline/`](../checkpoints/stage1_baseline/)

| Metric | Value |
|---|---|
| `last_step` | TODO |
| `val_loss` | TODO |
| `val_psnr` | TODO |
| `test_psnr_db` | TODO |
| `test_ddpm_loss` | TODO |
| Triplet PNG | [`../checkpoints/stage1_baseline/eval_test/test_triplets.png`](../checkpoints/stage1_baseline/eval_test/test_triplets.png) |
| Notes | TODO |

### Stage 2 table

| Run | Override | val_loss | val_psnr | test_psnr_db | Notes |
|---|---|---:|---:|---:|---|
| H00_baseline | none | TODO | TODO | TODO | TODO |
| H01_lr_low | `--lr 1e-4` | TODO | TODO | TODO | TODO |
| H02_lr_high | `--lr 3e-4` | TODO | TODO | TODO | TODO |
| H03_batch_16 | `--batch-size 16` | TODO | TODO | TODO | TODO |
| H04_epochs_plus | `--epochs 55` | TODO | TODO | TODO | TODO |
| H05_drop_sketch_high | `--drop-sketch-prob 0.2` | TODO | TODO | TODO | TODO |
| H06_guidance_high | `--guidance-scale 2.0` | TODO | TODO | TODO | TODO |
| H07_sample_steps_fast | `--sample-steps 120` | TODO | TODO | TODO | TODO |
| H08_lpips_late | `--lpips-start-frac 0.25` | TODO | TODO | TODO | TODO |
| H09_color_strong | `--color-loss-weight 0.05`, `--color-loss-start-frac 0.45` | TODO | TODO | TODO | TODO |
| H10_linear_schedule | `--beta-schedule linear` | TODO | TODO | TODO | TODO |
| H11_min_snr_5 | `--min-snr-gamma 5` | TODO | TODO | TODO | TODO |

### Selection placeholders

- Best numeric run: `TODO`
- Best visual run (rubric): `TODO`
- Stage-B follow-up plan: `TODO`
- Best Tier-A combo (`sample_steps`, `guidance_scale`): `TODO`

---

## Crisp takeaways

1. **Conditioning quality beat optimizer complexity.**
2. **Aligned synthetic sketches beat human sketches for this setup.**
3. **Color-aware loss is a practical win for skin/hair consistency.**
4. **Use run-matrix + rubric, not a single scalar, to pick final checkpoints.**
