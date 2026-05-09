# Hyperparameter Run Matrix (Stage 2)

All runs use the same data split and seed. **Authoritative list**: `notebooks/stage2_hyperparam_search.ipynb` (`RUN_MATRIX_STAGE_A` + `base_args`).

## Shared defaults (`base_args` in notebook)

- `--data-root /home/szb9536/genai_test_4/data`
- `--image-root /home/szb9536/genai_test_4/data/img_align_celeba_white`
- `--image-size 64`
- `--batch-size 32`
- `--epochs 75`
- `--lr 2e-4`
- `--seed 42`
- `--timesteps 1000`
- `--beta-schedule cosine`
- `--min-snr-gamma 0`
- `--guidance-scale 1.5`
- `--sample-steps 200`
- `--sample-every 2000`
- `--drop-sketch-prob 0.1`
- `--lpips-start-frac 0.1`
- `--color-loss-weight 0.02`
- `--color-loss-start-frac 0.6`
- `--color-loss-ramp-steps 5000`
- `--early-stop-patience 0`
- `--early-stop-min-delta 0`
- `--max-train-images 70000`
- `--amp`

## Stage A runs (medium search; 12 bundles)

| Run name | Overrides (beyond defaults) |
|---------|------------------------------|
| `H00_baseline` | *(none)* |
| `H01_lr_low` | `--lr 1e-4` |
| `H02_lr_high` | `--lr 3e-4` |
| `H03_batch_16` | `--batch-size 16` |
| `H04_epochs_plus` | `--epochs 55` |
| `H05_drop_sketch_high` | `--drop-sketch-prob 0.2` |
| `H06_guidance_high` | `--guidance-scale 2.0` |
| `H07_sample_steps_fast` | `--sample-steps 120` |
| `H08_lpips_late` | `--lpips-start-frac 0.25` |
| `H09_color_strong` | `--color-loss-weight 0.05`, `--color-loss-start-frac 0.45` |
| `H10_linear_schedule` | `--beta-schedule linear` |
| `H11_min_snr_5` | `--min-snr-gamma 5` |

Artifacts:

- `/home/szb9536/genai_test_4/checkpoints/hp_<RUN_NAME>/`

Stage B optional follow-ups: configure `STAGE_B_FOLLOWUP` in the notebook after reviewing summaries.

## Evaluation and selection rubric

Use `python -m src.eval_test` (vary `--guidance-scale` / `--sample-steps` as needed) for each run, or the sample commands printed by `notebooks/stage2_cases/analyze_stage2_cases.ipynb`, then rank by:

1. Hair-color correctness
2. Skin-tone realism
3. Smoothness / grain
4. Artifact penalty

Use scalar metrics (`val_loss` / `val_psnr`, optional `test_*` from `eval_test`) as tie-breakers — same ordering as the notebook **best-run** cell.
