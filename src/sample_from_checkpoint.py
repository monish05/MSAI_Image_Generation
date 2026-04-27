#!/usr/bin/env python3
"""
Generate sample images from a saved training checkpoint (no training).

From project root:  python3 -m src.sample_from_checkpoint --checkpoint checkpoints/ckpt_last.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

from .ddpm import GaussianDDPM
from .sketchy_dataset import SketchyPairDataset
from .unet_conditional import ConditionalUNet

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument(
        "--val-csv",
        type=Path,
        default=ROOT / "metadata" / "sketchy_tx000" / "val.csv",
    )
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "samples")
    p.add_argument("--num-batches", type=int, default=2, help="Batches to draw from val set")
    p.add_argument("--limit-per-batch", type=int, default=4, help="Max sketches per grid")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Missing checkpoint: {args.checkpoint.resolve()}")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)

    saved = ckpt.get("args") or {}
    timesteps = int(saved.get("timesteps", 1000))
    beta_start = float(saved.get("beta_start", 1e-4))
    beta_end = float(saved.get("beta_end", 2e-2))

    model = ConditionalUNet().to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    diffusion = GaussianDDPM(
        timesteps, beta_start=beta_start, beta_end=beta_end
    ).to(device)

    val_ds = SketchyPairDataset(
        args.val_csv, args.project_root, image_size=args.image_size
    )
    val_loader = DataLoader(val_ds, batch_size=max(1, args.limit_per_batch), shuffle=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    it = iter(val_loader)
    for b in tqdm(range(args.num_batches), desc="batches"):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(val_loader)
            batch = next(it)
        n = min(args.limit_per_batch, batch["sketch"].shape[0])
        sk = batch["sketch"].to(device)[:n]
        with torch.no_grad():
            x_gen = diffusion.sample(model, sk)
        vis = (x_gen.clamp(-1, 1) + 1) / 2
        out = args.out_dir / f"sample_batch{b:03d}.png"
        save_image(vis, out, nrow=min(2, n))
        print(f"saved {out}")


if __name__ == "__main__":
    main()
