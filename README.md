# MSAI_395_Image_Generation

Structure-Guided Diffusion with Noise-Space Steering for Controllable Image Generation

All Python code for this part of the project lives under `src/`. This README walks through setup and training from zero.

## 0. Get the project

```bash
cd /path/to/your/clone
# optional: use a venv
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

## 1. Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

This pulls **CUDA-enabled** PyTorch for NVIDIA GPUs (see comments in `requirements.txt`). If you use another CUDA line, adjust the `--index-url` and reinstall.

Quick check (optional):

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print('cuda:', torch.cuda.is_available())"
```

## 2. Add Sketchy data (local only)

`data/` is gitignored. You need the Sketchy database (or a repack) with 256×256 images laid out as:

- `data/photo/<tx>/...`
- `data/sketch/<tx>/...`

A public Sketchy-style bundle you can start from: [Sketch to Image (Kaggle)](https://www.kaggle.com/datasets/ankitsheoran23/sketch-to-image). If the extracted layout differs, move or symlink files so the paths above match (and keep `photo/` / `sketch/` tx folders consistent).

The default in this project uses `tx_000000000000` (see `data/README.txt` for what each `tx` means).

## 3. Build train/val/test CSVs (not in git)

`metadata/` is gitignored. Generate manifests from your local `data/`:

```bash
python -m src.build_sketchy_splits
# optional flags: --data-root ... --out-dir ... --tx tx_000000000000
```

This writes `metadata/sketchy_tx000/train.csv`, `val.csv`, `test.csv` (and `split_stats.json`) under the project root, with paths **relative to the project root** so training finds images under `data/`.

## 4. Train (single or multi-GPU)

From the project root (same directory as `requirements.txt`):

```bash
# one GPU
python -m src.train_ddpm --batch-size 8 --epochs 1 --max-train-steps 500
# two GPUs
torchrun --nproc_per_node=2 -m src.train_ddpm --batch-size 8 --epochs 1 --amp
# resume from explicit checkpoint
python -m src.train_ddpm --resume checkpoints/ckpt_last.pt --epochs 1
# resume from best validation checkpoint
python -m src.train_ddpm --resume-best --epochs 1
```

Checkpoints and samples go under `checkpoints/` (gitignored). Training now tracks validation loss each epoch and writes `checkpoints/ckpt_best.pt` when it improves, so you can continue from the best model. It also keeps an EMA copy of weights and uses CFG-style conditioning dropout for stronger sketch conditioning at inference. TensorBoard logs are under `checkpoints/tb/` and step-wise loss CSV logs are in `checkpoints/metrics.csv`. Override `--train-csv` / `--val-csv` if you used a custom `--out-dir` when building splits.

To sample from a checkpoint:

```bash
# defaults to checkpoints/ckpt_best.pt (falls back to ckpt_last.pt)
python -m src.sample_from_checkpoint --guidance-scale 2.0 --sampler ddim --sample-steps 100
```

EMA weights are used automatically when present (add `--no-use-ema` to disable).

## 5. Evaluate checkpoint quality on fixed validation examples

```bash
python -m src.evaluate_checkpoint \
  --checkpoint checkpoints/ckpt_best.pt \
  --num-eval 128 \
  --num-grid 8 \
  --sampler ddim \
  --sample-steps 100 \
  --guidance-scale 2.0
```

Outputs:
- `checkpoints/eval/fixed_val_grid.png`: fixed triplet grid `[sketch | generated | ground-truth]`
- `checkpoints/eval/eval_metrics.csv`: append-only evaluation log
- `checkpoints/eval/eval_summary.json`: JSON summary for the latest run

Optional FID/KID (depends on local `torchmetrics` image extras):

```bash
python -m src.evaluate_checkpoint --checkpoint checkpoints/ckpt_best.pt --compute-fid-kid
```
