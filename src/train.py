#!/usr/bin/env python3
"""Train sketch-conditioned diffusion on CelebA (dodge sketch)."""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import torch
from contextlib import nullcontext
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

# LPIPS aux: starts at half of planned optimizer steps (printed at startup), linear ramp then hold
LPIPS_LAMBDA_MAX = 0.05
LPIPS_RAMP_STEPS = 5000


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ema_sync(dst: nn.Module, src: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for kd, ks in zip(dst.state_dict().values(), src.state_dict().values()):
            kd.mul_(decay).add_(ks, alpha=1.0 - decay)


def sketch_cfg_dropout(sketch: torch.Tensor, p: float) -> torch.Tensor:
    """Null sketch channels = (-1,) in normalized space."""
    if p <= 0:
        return sketch
    b = sketch.shape[0]
    blank = torch.full_like(sketch, -1.0)
    km = torch.rand(b, device=sketch.device) >= p
    m = km.view(-1, 1, 1, 1).to(dtype=sketch.dtype)
    return sketch * m + blank * (1.0 - m)


def csv_row(path: Path, cols: dict[str, str], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(cols)


def m11_as_float01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) * 0.5


@torch.no_grad()
def validation_pass(
    model: nn.Module, ddpm_module, dl: DataLoader, device: torch.device, min_snr: float | None
) -> float:
    model.eval()
    tot = 0.0
    nb = 0
    for batch in dl:
        ph = batch["photo"].to(device)
        sk = batch["sketch"].to(device)
        loss, _ = ddpm_module.training_losses(model, ph, sk, min_snr_gamma=min_snr)
        tot += float(loss)
        nb += 1
    model.train()
    return tot / max(nb, 1)


@torch.no_grad()
def save_triplets(
    ema_m: nn.Module,
    diffusion,
    batch: dict[str, torch.Tensor],
    path: Path,
    device: torch.device,
    *,
    gs: float,
    steps: int,
) -> None:
    sk = batch["sketch"].to(device)
    ph = batch["photo"].to(device)
    n = min(4, sk.shape[0])
    sk = sk[:n]
    fk = diffusion.ddim_sample_loop(
        ema_m, sk, guidance_scale=gs, steps=steps, eta=0.0
    )
    vis_sk = ((sk[:n].clamp(-1, 1) + 1) / 2).repeat(1, 3, 1, 1)
    vis_gt = ((ph[:n].clamp(-1, 1) + 1) / 2)
    vis_g = ((fk[:n].clamp(-1, 1) + 1) / 2)
    trip = torch.stack([vis_sk, vis_g, vis_gt], dim=1).flatten(0, 1).cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(trip, path, nrow=3)


def lpips_optional(device: torch.device, use: bool) -> nn.Module | None:
    if not use:
        return None
    try:
        import lpips as L
    except ImportError:
        raise SystemExit("LPIPS enabled: pip install lpips")
    m = L.LPIPS(net="alex").to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()
    return m


def lambda_phase2(step: int, phase_start: int, ramp_steps: float, lam_max: float) -> float:
    if phase_start < 0 or step < phase_start:
        return 0.0
    if ramp_steps <= 0:
        return lam_max
    u = step - phase_start
    return float(min(1.0, u / ramp_steps) * lam_max)


@torch.no_grad()
def run_fid_manifest(
    ema_m: nn.Module,
    diffusion,
    ds_subset,
    device: torch.device,
    *,
    guidance_scale: float,
    steps: int,
    batch_sz: int,
) -> float | None:
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        return None
    dl = DataLoader(
        ds_subset,
        batch_size=batch_sz,
        shuffle=False,
        num_workers=min(4, torch.get_num_threads() or 1),
        pin_memory=device.type == "cuda",
    )
    fid_m = FrechetInceptionDistance(normalize=True).to(device)
    ema_m.eval()
    for batch in dl:
        real = batch["photo"].to(device)
        sketch = batch["sketch"].to(device)
        fake = diffusion.ddim_sample_loop(
            ema_m, sketch, guidance_scale=guidance_scale, steps=steps, eta=0.0
        )
        fid_m.update(m11_as_float01(real), real=True)
        fid_m.update(m11_as_float01(fake), real=False)
    return float(fid_m.compute().item())


def main() -> None:
    from .celeba import PART_TRAIN, PART_VAL, CelebSketchDataset, sample_fixed_manifest
    from .ddpm import GaussianDDPM
    from .unet import SketchEpsilonUNet

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--save-dir", type=Path, default=ROOT / "checkpoints")
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--beta-start", type=float, default=1e-4)
    ap.add_argument("--beta-end", type=float, default=2e-2)
    ap.add_argument("--base-channels", type=int, default=96)
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--drop-sketch-prob", type=float, default=0.1)
    ap.add_argument("--min-snr-gamma", type=float, default=5.0)
    ap.add_argument("--guidance-scale", type=float, default=1.5)
    ap.add_argument("--sample-steps", type=int, default=80)
    ap.add_argument("--sample-every", type=int, default=2000)
    ap.add_argument("--fid-every", type=int, default=0)
    ap.add_argument("--fid-count", type=int, default=1024)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--manifest-seed", type=int, default=123)
    ap.add_argument(
        "--no-lpips",
        action="store_true",
        help="Skip perceptual loss (otherwise LPIPS starts halfway through planned steps).",
    )
    ap.add_argument("--max-train-images", type=int, default=None)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    min_snr = args.min_snr_gamma if args.min_snr_gamma > 0 else None

    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.csv"
    fid_log = save_dir / "fid_history.csv"

    manifest_path = args.manifest or (save_dir / "val_sketch_manifest.json")
    if manifest_path.is_file():
        id_list = json.loads(manifest_path.read_text())["image_ids"]
    else:
        id_list = sample_fixed_manifest(
            args.data_root,
            PART_VAL,
            image_root=None,
            max_count=min(args.fid_count, 50_000),
            seed=args.manifest_seed,
            out_json=manifest_path,
        )

    ds_tr = CelebSketchDataset(
        args.data_root, PART_TRAIN, args.image_size, max_images=args.max_train_images
    )
    ds_va = CelebSketchDataset(args.data_root, PART_VAL, args.image_size)
    ds_fid = CelebSketchDataset(args.data_root, PART_VAL, args.image_size, only_filenames=id_list)
    if len(ds_tr) == 0:
        raise SystemExit("No training images under data/ — check CSV and img_align_celeba.")

    pin = device.type == "cuda"
    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=pin,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=min(4, args.batch_size),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin,
    )

    steps_per_epoch = len(dl_tr)
    if steps_per_epoch == 0:
        raise SystemExit(
            "Train DataLoader has no batches (dataset smaller than --batch-size with drop_last?). "
            "Lower --batch-size."
        )
    epoch_budget = args.epochs * steps_per_epoch
    if args.max_steps and args.max_steps > 0:
        planned_steps = min(epoch_budget, args.max_steps)
    else:
        planned_steps = epoch_budget

    lpips_mid = planned_steps // 2
    use_lpips = (not args.no_lpips) and planned_steps > 0
    args.planned_train_steps = planned_steps
    args.lpips_start_step = lpips_mid if use_lpips else -1

    print(
        f"[train] train samples={len(ds_tr)} batch={args.batch_size} → "
        f"batches/epoch={steps_per_epoch} epochs={args.epochs} "
        f"max_steps={args.max_steps if args.max_steps > 0 else 'no-cap'} "
        f"→ planned_steps≈{planned_steps}",
        flush=True,
    )
    if use_lpips:
        print(
            f"[train] LPIPS from step≈{lpips_mid} "
            f"(ramp={LPIPS_RAMP_STEPS}, λ_max={LPIPS_LAMBDA_MAX})",
            flush=True,
        )
    elif args.no_lpips:
        print("[train] LPIPS off (--no-lpips)", flush=True)

    model = SketchEpsilonUNet(base=args.base_channels).to(device)
    ema = SketchEpsilonUNet(base=args.base_channels).to(device)
    ema.load_state_dict(model.state_dict())
    for q in ema.parameters():
        q.requires_grad_(False)

    ddpm = GaussianDDPM(args.timesteps, beta_start=args.beta_start, beta_end=args.beta_end).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    lip = lpips_optional(device, use_lpips)

    from torch.utils.tensorboard import SummaryWriter

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb = SummaryWriter(save_dir / "tb" / run_name)

    global_step = 0
    best_val = float("inf")
    fields_m = ["step", "epoch", "train_loss", "val_loss", "lr"]

    def train_one_batch(step_ix: int, batch: dict[str, torch.Tensor]) -> float:
        ph = batch["photo"].to(device, non_blocking=True)
        sk = batch["sketch"].to(device, non_blocking=True)
        bsz = ph.shape[0]
        noise = torch.randn_like(ph)
        t = torch.randint(0, ddpm.timesteps, (bsz,), device=device, dtype=torch.long)
        sk_in = sketch_cfg_dropout(sk, args.drop_sketch_prob)
        x_t = ddpm.q_sample(ph, t, noise)

        opt.zero_grad(set_to_none=True)
        ac = torch.amp.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
        with ac:
            eps_hat = model(x_t, sk_in, t)
            err = (eps_hat - noise).pow(2).flatten(1).mean(dim=-1)
            if min_snr is not None:
                acp = ddpm.alphas_cumprod[t]
                snr = acp / (1.0 - acp).clamp(min=1e-8)
                w = torch.clamp(snr, max=min_snr) / snr
                loss_eps = (err * w).mean()
            else:
                loss_eps = err.mean()
            pred_x0 = ddpm.predict_x0_from_eps(x_t, t, eps_hat)
        lam = lambda_phase2(step_ix, lpips_mid, float(LPIPS_RAMP_STEPS), LPIPS_LAMBDA_MAX)
        if lip is not None and lam > 0:
            loss_fin = loss_eps + lam * lip(pred_x0.float(), ph.float()).mean()
        else:
            loss_fin = loss_eps

        if use_amp:
            scaler.scale(loss_fin).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss_fin.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        ema_sync(ema, model, args.ema_decay)
        return float(loss_fin.item())

    def maybe_fid() -> None:
        if args.fid_every <= 0 or global_step % args.fid_every != 0:
            return
        fid_v = run_fid_manifest(
            ema,
            ddpm,
            ds_fid,
            device,
            guidance_scale=args.guidance_scale,
            steps=args.sample_steps,
            batch_sz=min(8, args.batch_size),
        )
        if fid_v is None:
            return
        csv_row(
            fid_log,
            {
                "step": str(global_step),
                "fid": f"{fid_v:.4f}",
                "guidance_scale": f"{args.guidance_scale:.3f}",
                "drop_sketch_prob": f"{args.drop_sketch_prob:.3f}",
                "cfg_used": str(args.guidance_scale > 1.0),
                "num_samples": str(len(ds_fid)),
            },
            ["step", "fid", "guidance_scale", "drop_sketch_prob", "cfg_used", "num_samples"],
        )

    for epoch in range(args.epochs):
        model.train()
        bar = tqdm(dl_tr, desc=f"epoch {epoch}")
        for batch in bar:
            if args.max_steps and global_step >= args.max_steps:
                break
            loss_f = train_one_batch(global_step, batch)
            global_step += 1
            if args.fid_every > 0 and global_step % args.fid_every == 0:
                maybe_fid()
            lr = opt.param_groups[0]["lr"]
            bar.set_postfix(loss=f"{loss_f:.4f}", step=global_step)
            tb.add_scalar("train/loss", loss_f, global_step)
            tb.add_scalar("train/lr", lr, global_step)
            if global_step % 50 == 0:
                csv_row(
                    metrics_path,
                    {
                        "step": str(global_step),
                        "epoch": str(epoch),
                        "train_loss": f"{loss_f:.6f}",
                        "val_loss": "",
                        "lr": f"{lr:.2e}",
                    },
                    fields_m,
                )
            if args.sample_every > 0 and global_step % args.sample_every == 0:
                vb = next(iter(dl_va))
                save_triplets(
                    ema,
                    ddpm,
                    vb,
                    save_dir / "samples" / f"step_{global_step:07d}.png",
                    device,
                    gs=args.guidance_scale,
                    steps=args.sample_steps,
                )

        vloss = validation_pass(model, ddpm, dl_va, device, min_snr)
        tb.add_scalar("val/loss", vloss, global_step)
        csv_row(
            metrics_path,
            {
                "step": str(global_step),
                "epoch": str(epoch),
                "train_loss": "",
                "val_loss": f"{vloss:.6f}",
                "lr": f"{opt.param_groups[0]['lr']:.2e}",
            },
            fields_m,
        )
        if vloss < best_val:
            best_val = vloss
            torch.save(
                {"model": model.state_dict(), "ema": ema.state_dict(), "step": global_step, "args": vars(args)},
                save_dir / "ckpt_best.pt",
            )
        torch.save(
            {"model": model.state_dict(), "ema": ema.state_dict(), "step": global_step, "args": vars(args)},
            save_dir / "ckpt_last.pt",
        )
        if args.max_steps and global_step >= args.max_steps:
            break

    vb = next(iter(dl_va))
    save_triplets(
        ema,
        ddpm,
        vb,
        save_dir / "results" / f"final_{global_step:07d}.png",
        device,
        gs=args.guidance_scale,
        steps=args.sample_steps,
    )
    tb.close()
    print("done.", global_step, "steps. best_val", best_val)


if __name__ == "__main__":
    main()