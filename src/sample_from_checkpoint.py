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


def _infer_class_vocab(ckpt: dict, saved: dict) -> dict[str, int] | None:
    if not any(k.startswith("class_emb.") for k in ckpt["model"]):
        return None
    raw = ckpt.get("class_vocab") or saved.get("class_vocab") or saved.get(
        "class_to_idx"
    )
    if not isinstance(raw, dict) or not raw:
        raise SystemExit(
            "Checkpoint expects class embedding but lacks class_vocab. "
            "Re-save checkpoints from training with an updated codebase."
        )
    return {str(k): int(v) for k, v in raw.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument(
        "--val-csv",
        type=Path,
        default=ROOT / "metadata" / "sketchy_tx000" / "val.csv",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Must match training --image-size.",
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "samples")
    p.add_argument("--num-batches", type=int, default=2, help="Batches to draw from val set")
    p.add_argument("--limit-per-batch", type=int, default=4, help="Max sketches per grid")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--guidance-scale", type=float, default=2.0)
    p.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    p.add_argument("--sample-steps", type=int, default=100, help="Used for DDIM sampling.")
    p.add_argument("--ddim-eta", type=float, default=0.0, help="DDIM stochasticity; 0 = deterministic.")
    p.add_argument(
        "--base-channels",
        type=int,
        default=None,
        help="U-Net width (default: from checkpoint train args, else 64 for legacy).",
    )
    p.add_argument(
        "--no-use-ema",
        action="store_true",
        help="Disable EMA weights even if present in checkpoint.",
    )
    args = p.parse_args()

    if args.checkpoint is None:
        best = ROOT / "checkpoints" / "ckpt_best.pt"
        last = ROOT / "checkpoints" / "ckpt_last.pt"
        args.checkpoint = best if best.is_file() else last
    if not args.checkpoint.is_file():
        raise SystemExit(
            f"Missing checkpoint: {args.checkpoint.resolve()} (tried ckpt_best.pt/ckpt_last.pt defaults)"
        )

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
    base_ch = (
        int(args.base_channels)
        if args.base_channels is not None
        else int(saved.get("base_channels", 64))
    )
    use_cross_attn = any(
        k.startswith("mid_cross_attn.") for k in ckpt["model"]
    )
    class_vocab = _infer_class_vocab(ckpt, saved)
    num_cls = len(class_vocab) if class_vocab is not None else 0

    model = ConditionalUNet(
        base_channels=base_ch,
        use_cross_attention=use_cross_attn,
        num_semantic_classes=num_cls,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    ema_model = None
    if not args.no_use_ema and ckpt.get("ema_model") is not None:
        ema_model = ConditionalUNet(
            base_channels=base_ch,
            use_cross_attention=use_cross_attn,
            num_semantic_classes=num_cls,
        ).to(device)
        ema_model.load_state_dict(ckpt["ema_model"], strict=True)
        ema_model.eval()
    diffusion = GaussianDDPM(
        timesteps, beta_start=beta_start, beta_end=beta_end
    ).to(device)

    val_ds = SketchyPairDataset(
        args.val_csv,
        args.project_root,
        image_size=args.image_size,
        class_to_idx=class_vocab,
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
        cid = (
            batch["class_id"].to(device)[:n] if num_cls > 0 else None
        )
        with torch.no_grad():
            if args.sampler == "ddim":
                used_model = ema_model or model
                x_gen = diffusion.sample_ddim(
                    used_model,
                    sk,
                    guidance_scale=args.guidance_scale,
                    sample_steps=args.sample_steps,
                    eta=args.ddim_eta,
                    class_id=cid,
                )
            else:
                x_gen = diffusion.sample(
                    model,
                    sk,
                    use_ema_model=ema_model,
                    guidance_scale=args.guidance_scale,
                    sampler="ddpm",
                    class_id=cid,
                )
        vis = (x_gen.clamp(-1, 1) + 1) / 2
        out = args.out_dir / f"sample_batch{b:03d}.png"
        save_image(vis, out, nrow=min(2, n))
        print(f"saved {out}")


if __name__ == "__main__":
    main()
