# CelebA sketch-conditioned diffusion

Trains \(\varepsilon_\theta(x_t,\,t\,|\,\texttt{sketch})\) on aligned Celeb faces; sketches use the dodge pipeline like [`../sketch-to-image/image2sketch.ipynb`](../sketch-to-image/image2sketch.ipynb).

Put CelebA under [`data/`](data/) (`list_eval_partition.csv`, `img_align_celeba/` — nested layout auto-detected). Gitignored.

**From `genai_test/`:**

```bash
python -m src.train --data-root ./data --batch-size 8 --drop-sketch-prob 0.1 --sample-every 2000
```

At startup training prints **`planned_steps`** (= `epochs × batches_per_epoch`, or capped by **`--max-steps`**). **LPIPS** turns on automatically at **half that count** (`λ` ramp fixed in code). Use **`--no-lpips`** to train with noise loss only.

| Flag | Meaning |
|------|---------|
| `--drop-sketch-prob` | CFG dropout on sketch conditioning. |
| `--guidance-scale` | Sampling / FID DDIM CFG weight. |
| `--fid-every N` | Optional FID log → `checkpoints/fid_history.csv`. |
| `--sample-every N` | Triplet PNGs in `checkpoints/samples/`. |
| `--amp` | Mixed precision on CUDA. |
| `--no-lpips` | Disable midpoint LPIPS. |

Defaults: **`--epochs 50`**, **`--data-root`** = `./data`. Install PyTorch for your CUDA, then `pip install -r requirements.txt`.
