#!/usr/bin/env python3
"""
Evaluate a checkpoint with fixed validation samples and optional FID/KID.

Example:
python -m src.evaluate_checkpoint --checkpoint checkpoints/ckpt_best.pt --num-eval 128 --compute-fid-kid
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from .ddpm import GaussianDDPM
from .sketchy_dataset import SketchyPairDataset
from .unet_conditional import ConditionalUNet

ROOT = Path(__file__).resolve().parent.parent


def _infer_class_vocab_eval(ckpt: dict, saved: dict) -> dict[str, int] | None:
    if not any(k.startswith("class_emb.") for k in ckpt["model"]):
        return None
    raw = ckpt.get("class_vocab") or saved.get("class_vocab") or saved.get(
        "class_to_idx"
    )
    if not isinstance(raw, dict) or not raw:
        raise SystemExit(
            "Checkpoint expects class embedding but lacks class_vocab. "
            "Re-save from training with class_vocab in the checkpoint file."
        )
    return {str(k): int(v) for k, v in raw.items()}


def _to_uint8(x_m11: torch.Tensor) -> torch.Tensor:
    x01 = (x_m11.clamp(-1, 1) + 1.0) / 2.0
    return (x01 * 255.0).round().clamp(0, 255).to(torch.uint8)


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
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
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-grid", type=int, default=8, help="Fixed examples for visual triplet grid.")
    p.add_argument("--num-eval", type=int, default=128, help="Validation examples to evaluate.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=123, help="Seed for fixed eval subset.")
    p.add_argument("--guidance-scale", type=float, default=2.0)
    p.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    p.add_argument("--sample-steps", type=int, default=100)
    p.add_argument("--ddim-eta", type=float, default=0.0)
    p.add_argument(
        "--base-channels",
        type=int,
        default=None,
        help="U-Net width (default: from checkpoint train args, else 64 for legacy).",
    )
    p.add_argument("--compute-fid-kid", action="store_true")
    p.add_argument("--out-dir", type=Path, default=ROOT / "checkpoints" / "eval")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if not args.checkpoint.is_file():
        raise SystemExit(f"Missing checkpoint: {args.checkpoint.resolve()}")
    ckpt = _load_checkpoint(args.checkpoint, device)
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
    class_vocab = _infer_class_vocab_eval(ckpt, saved)
    num_cls = len(class_vocab) if class_vocab is not None else 0

    model = ConditionalUNet(
        base_channels=base_ch,
        use_cross_attention=use_cross_attn,
        num_semantic_classes=num_cls,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    ema_model = None
    if ckpt.get("ema_model") is not None:
        ema_model = ConditionalUNet(
            base_channels=base_ch,
            use_cross_attention=use_cross_attn,
            num_semantic_classes=num_cls,
        ).to(device)
        ema_model.load_state_dict(ckpt["ema_model"], strict=True)
        ema_model.eval()
    use_model = ema_model or model

    diffusion = GaussianDDPM(
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
    ).to(device)

    ds = SketchyPairDataset(
        args.val_csv,
        args.project_root,
        image_size=args.image_size,
        class_to_idx=class_vocab,
    )
    if len(ds) == 0:
        raise SystemExit(f"Validation CSV contains no rows: {args.val_csv}")

    g = torch.Generator().manual_seed(args.seed)
    num_eval = min(args.num_eval, len(ds))
    perm = torch.randperm(len(ds), generator=g).tolist()
    eval_indices = perm[:num_eval]
    grid_indices = eval_indices[: min(args.num_grid, num_eval)]
    eval_loader = DataLoader(
        Subset(ds, eval_indices),
        batch_size=args.batch_size,
        shuffle=False,
    )
    grid_loader = DataLoader(Subset(ds, grid_indices), batch_size=len(grid_indices), shuffle=False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "eval_metrics.csv"
    summary_path = args.out_dir / "eval_summary.json"
    grid_path = args.out_dir / "fixed_val_grid.png"

    fid = kid = None
    fid_value = kid_mean = kid_std = None
    if args.compute_fid_kid:
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            from torchmetrics.image.kid import KernelInceptionDistance
        except Exception as e:
            print(f"Could not import FID/KID metrics ({e}); continuing without FID/KID.")
        else:
            fid = FrechetInceptionDistance(feature=2048).to(device)
            kid = KernelInceptionDistance(subset_size=50).to(device)

    abs_l1_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch in eval_loader:
            sk = batch["sketch"].to(device)
            gt = batch["photo"].to(device)
            cid = batch["class_id"].to(device) if num_cls > 0 else None
            if args.sampler == "ddim":
                pred = diffusion.sample_ddim(
                    use_model,
                    sk,
                    guidance_scale=args.guidance_scale,
                    sample_steps=args.sample_steps,
                    eta=args.ddim_eta,
                    class_id=cid,
                )
            else:
                pred = diffusion.sample(
                    use_model,
                    sk,
                    guidance_scale=args.guidance_scale,
                    sampler="ddpm",
                    class_id=cid,
                )
            abs_l1_sum += torch.abs(pred - gt).mean(dim=(1, 2, 3)).sum().item()
            count += pred.shape[0]

            if fid is not None and kid is not None:
                pred_u8 = _to_uint8(pred)
                gt_u8 = _to_uint8(gt)
                fid.update(gt_u8, real=True)
                fid.update(pred_u8, real=False)
                kid.update(gt_u8, real=True)
                kid.update(pred_u8, real=False)

    mean_l1 = abs_l1_sum / max(count, 1)
    if fid is not None and kid is not None:
        fid_value = float(fid.compute().item())
        kid_mean_t, kid_std_t = kid.compute()
        kid_mean = float(kid_mean_t.item())
        kid_std = float(kid_std_t.item())

    with torch.no_grad():
        grid_batch = next(iter(grid_loader))
        sk = grid_batch["sketch"].to(device)
        gt = grid_batch["photo"].to(device)
        gid = (
            grid_batch["class_id"].to(device) if num_cls > 0 else None
        )
        if args.sampler == "ddim":
            pred = diffusion.sample_ddim(
                use_model,
                sk,
                guidance_scale=args.guidance_scale,
                sample_steps=args.sample_steps,
                eta=args.ddim_eta,
                class_id=gid,
            )
        else:
            pred = diffusion.sample(
                use_model,
                sk,
                guidance_scale=args.guidance_scale,
                sampler="ddpm",
                class_id=gid,
            )
        vis_sk = ((sk.clamp(-1, 1) + 1) / 2).repeat(1, 3, 1, 1)
        vis_pred = (pred.clamp(-1, 1) + 1) / 2
        vis_gt = (gt.clamp(-1, 1) + 1) / 2
        triplets = torch.stack([vis_sk, vis_pred, vis_gt], dim=1).flatten(0, 1).cpu()
        save_image(triplets, grid_path, nrow=3)

    row = {
        "checkpoint": str(args.checkpoint),
        "num_eval": str(num_eval),
        "mean_l1_m11": f"{mean_l1:.6f}",
        "guidance_scale": f"{args.guidance_scale:.3f}",
        "sampler": args.sampler,
        "sample_steps": str(args.sample_steps),
        "ddim_eta": f"{args.ddim_eta:.3f}",
        "fid": "" if fid_value is None else f"{fid_value:.6f}",
        "kid_mean": "" if kid_mean is None else f"{kid_mean:.6f}",
        "kid_std": "" if kid_std is None else f"{kid_std:.6f}",
    }
    header = list(row.keys())
    write_header = not metrics_path.exists()
    with metrics_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "grid_path": str(grid_path.resolve()),
        "num_eval": num_eval,
        "mean_l1_m11": mean_l1,
        "sampler": args.sampler,
        "sample_steps": args.sample_steps,
        "ddim_eta": args.ddim_eta,
        "guidance_scale": args.guidance_scale,
        "fid": fid_value,
        "kid_mean": kid_mean,
        "kid_std": kid_std,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Appended metrics row to {metrics_path}")


if __name__ == "__main__":
    main()
