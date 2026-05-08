#!/usr/bin/env python3
"""Evaluate a trained checkpoint on CelebA test (partition 2): DDPM loss, optional FID, triplet PNGs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent


def _load_ckpt(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    from .celeba import PART_TEST, CelebSketchDataset, sample_fixed_manifest
    from .ddpm import GaussianDDPM
    from .train import run_fid_manifest, save_triplets, set_seed, validation_pass
    from .unet import SketchEpsilonUNet

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--image-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "checkpoints" / "eval_test")
    ap.add_argument("--image-size", type=int, default=None, help="Default: from checkpoint args, else 64.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-test-images", type=int, default=None)
    ap.add_argument("--guidance-scale", type=float, default=None)
    ap.add_argument("--sample-steps", type=int, default=None)
    ap.add_argument("--fid", action="store_true", help="Compute FID on a fixed random subset of test IDs (slow).")
    ap.add_argument("--fid-max", type=int, default=1024)
    ap.add_argument(
        "--fid-strict",
        action="store_true",
        help="With --fid, exit with code 2 if FID cannot be computed (e.g. torchmetrics missing).",
    )
    ap.add_argument("--manifest-seed", type=int, default=123)
    ap.add_argument(
        "--triplet-png",
        type=Path,
        default=None,
        help="Save sketch|fake|GT grid from the first test batch (e.g. out-dir/test_triplets.png).",
    )
    ap.add_argument(
        "--triplet-channel-swap-debug",
        action="store_true",
        help="With --triplet-png, also write *_chswap.png (4-column R-B swap comparison).",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    ck = _load_ckpt(args.ckpt, device)
    saved: dict = ck.get("args") or {}

    def pick(key: str, cli_val, default):
        if cli_val is not None:
            return cli_val
        v = saved.get(key)
        return v if v is not None else default

    image_size = int(pick("image_size", args.image_size, 64))
    timesteps = int(pick("timesteps", None, 1000))
    beta_start = float(pick("beta_start", None, 1e-4))
    beta_end = float(pick("beta_end", None, 2e-2))
    beta_schedule = str(pick("beta_schedule", None, "linear"))
    base_ch = int(pick("base_channels", None, 96))
    gs = float(pick("guidance_scale", args.guidance_scale, 1.5))
    sample_steps = int(pick("sample_steps", args.sample_steps, 80))
    msg = float(pick("min_snr_gamma", None, 5.0))
    min_snr = msg if msg > 0 else None

    ema = SketchEpsilonUNet(base=base_ch).to(device)
    ema.load_state_dict(ck["ema"])
    ema.eval()
    ddpm = GaussianDDPM(
        timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
    ).to(device)

    ds = CelebSketchDataset(
        args.data_root, PART_TEST, image_size, image_root=args.image_root, max_images=args.max_test_images
    )
    if len(ds) == 0:
        raise SystemExit("Test split is empty — check data paths and list_eval_partition.csv.")
    dl = DataLoader(
        ds,
        batch_size=min(args.batch_size, len(ds)),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tloss, tpsnr = validation_pass(ema, ddpm, dl, device, min_snr)
    print(
        f"test_ddpm_loss={tloss:.6f}  test_psnr={tpsnr:.4f}dB  n={len(ds)}  image_size={image_size}",
        flush=True,
    )

    summary: dict = {
        "test_ddpm_loss": tloss,
        "test_psnr_db": tpsnr,
        "n_test": len(ds),
        "ckpt": str(args.ckpt.resolve()),
        "image_size": image_size,
        "guidance_scale": gs,
        "sample_steps": sample_steps,
        "beta_schedule": beta_schedule,
    }

    if args.fid:
        manifest = args.out_dir / "test_fid_manifest.json"
        ids = sample_fixed_manifest(
            args.data_root,
            PART_TEST,
            image_root=args.image_root,
            max_count=min(args.fid_max, len(ds)),
            seed=args.manifest_seed,
            out_json=manifest,
        )
        ds_fid = CelebSketchDataset(
            args.data_root, PART_TEST, image_size, image_root=args.image_root, only_filenames=ids
        )
        fid_v = run_fid_manifest(
            ema,
            ddpm,
            ds_fid,
            device,
            guidance_scale=gs,
            steps=sample_steps,
            batch_sz=min(8, args.batch_size),
        )
        if fid_v is None:
            print("FID skipped (install torchmetrics or fix FID path).", flush=True)
            summary["fid"] = None
            if args.fid_strict:
                print(
                    "eval_test: --fid --fid-strict requires FID (e.g. pip install torchmetrics; see requirements.txt).",
                    flush=True,
                )
                raise SystemExit(2)
        else:
            summary["fid"] = fid_v
            print(f"test_fid={fid_v:.4f}  n_fid={len(ds_fid)}", flush=True)

    if args.triplet_png is not None:
        batch = next(iter(dl))
        save_triplets(
            ema,
            ddpm,
            batch,
            args.triplet_png,
            device,
            gs=gs,
            steps=sample_steps,
            channel_swap_debug=args.triplet_channel_swap_debug,
        )

    out_json = args.out_dir / "test_eval_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
