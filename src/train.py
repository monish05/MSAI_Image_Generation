#!/usr/bin/env python3
"""Train sketch-conditioned diffusion on CelebA (dodge sketch)."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

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


def _read_metric_series(
    metrics_csv: Path, train_col: str, val_col: str
) -> tuple[list[int], list[float], list[int], list[float]]:
    st_t: list[int] = []
    y_t: list[float] = []
    st_v: list[int] = []
    y_v: list[float] = []
    if not metrics_csv.is_file():
        return st_t, y_t, st_v, y_v
    with metrics_csv.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                st = int(row["step"])
            except (KeyError, ValueError):
                continue
            tv = (row.get(train_col) or "").strip()
            if tv:
                try:
                    st_t.append(st)
                    y_t.append(float(tv))
                except ValueError:
                    pass
            vv = (row.get(val_col) or "").strip()
            if vv:
                try:
                    st_v.append(st)
                    y_v.append(float(vv))
                except ValueError:
                    pass
    return st_t, y_t, st_v, y_v


def _plot_metric_png(
    metrics_csv: Path,
    out_png: Path,
    *,
    train_col: str,
    val_col: str,
    y_label: str,
    title: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not installed; skip {y_label} PNG", flush=True)
        return

    st_t, y_t, st_v, y_v = _read_metric_series(metrics_csv, train_col, val_col)
    if not st_t and not st_v:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    if st_t:
        ax.plot(st_t, y_t, lw=0.85, alpha=0.9, label="train (logged steps)")
    if st_v:
        ax.plot(st_v, y_v, marker="o", ms=3, lw=1.0, alpha=0.9, label="val (epoch end)")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if st_t or st_v:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def save_loss_plot_png(metrics_csv: Path, out_png: Path) -> None:
    _plot_metric_png(
        metrics_csv,
        out_png,
        train_col="train_loss",
        val_col="val_loss",
        y_label="loss",
        title="Sketch DDPM train / val loss",
    )


def save_psnr_plot_png(metrics_csv: Path, out_png: Path) -> None:
    _plot_metric_png(
        metrics_csv,
        out_png,
        train_col="train_psnr",
        val_col="val_psnr",
        y_label="PSNR (dB)",
        title="Sketch DDPM train / val PSNR (pred_x0 vs photo)",
    )


def m11_as_float01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) * 0.5


@torch.no_grad()
def psnr_pred_x0_db(pred_x0: torch.Tensor, photo: torch.Tensor) -> float:
    p = m11_as_float01(pred_x0).float()
    g = m11_as_float01(photo).float()
    mse = (p - g).pow(2).mean(dim=(1, 2, 3)).clamp(min=1e-12)
    return float((10.0 * torch.log10(1.0 / mse)).mean().item())


def _rgb01_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    rgb = rgb.clamp(0.0, 1.0)
    a = 0.055
    rgb_lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + a) / (1 + a)).pow(2.4))

    r, g, b = rgb_lin[:, 0:1], rgb_lin[:, 1:2], rgb_lin[:, 2:3]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    xn, yn, zn = 0.95047, 1.0, 1.08883
    x = x / xn
    y = y / yn
    z = z / zn

    d = 6 / 29
    d3 = d**3
    k = 1 / (3 * d * d)

    def f(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > d3, t.clamp_min(eps).pow(1 / 3), k * t + 4 / 29)

    fx, fy, fz = f(x), f(y), f(z)
    l = 116 * fy - 16
    aa = 500 * (fx - fy)
    bb = 200 * (fy - fz)
    return torch.cat([l, aa, bb], dim=1)


def color_ab_l1(pred_x0: torch.Tensor, photo: torch.Tensor) -> torch.Tensor:
    pred = _rgb01_to_lab(m11_as_float01(pred_x0).float())
    gt = _rgb01_to_lab(m11_as_float01(photo).float())
    return (pred[:, 1:] - gt[:, 1:]).abs().mean()


@torch.no_grad()
def validation_pass(
    model: nn.Module, ddpm_module, dl: DataLoader, device: torch.device, min_snr: float | None
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    tot_loss = 0.0
    tot_psnr = 0.0
    nb = 0
    try:
        for batch in dl:
            ph = batch["photo"].to(device)
            sk = batch["sketch"].to(device)
            loss, pred_x0 = ddpm_module.training_losses(model, ph, sk, min_snr_gamma=min_snr)
            tot_loss += float(loss)
            tot_psnr += psnr_pred_x0_db(pred_x0, ph)
            nb += 1
    finally:
        model.train(was_training)
    denom = max(nb, 1)
    return tot_loss / denom, tot_psnr / denom


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
    channel_swap_debug: bool = False,
) -> None:
    sk = batch["sketch"].to(device)
    ph = batch["photo"].to(device)
    n = min(4, sk.shape[0])
    sk = sk[:n]
    fk = diffusion.ddim_sample_loop(ema_m, sk, guidance_scale=gs, steps=steps, eta=0.0)
    vis_sk = ((sk[:n].clamp(-1, 1) + 1) / 2).repeat(1, 3, 1, 1)
    vis_gt = ((ph[:n].clamp(-1, 1) + 1) / 2)
    vis_g = ((fk[:n].clamp(-1, 1) + 1) / 2)
    trip = torch.stack([vis_sk, vis_g, vis_gt], dim=1).flatten(0, 1).cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(trip, path, nrow=3)
    if channel_swap_debug:
        vis_swap = vis_g[:, [2, 1, 0], :, :]
        trip4 = torch.stack([vis_sk, vis_g, vis_swap, vis_gt], dim=1).flatten(0, 1).cpu()
        swap_path = path.with_name(f"{path.stem}_chswap{path.suffix}")
        save_image(trip4, swap_path, nrow=4)


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
    try:
        fid_m = FrechetInceptionDistance(normalize=True).to(device)
    except ModuleNotFoundError:
        return None
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
    ap.add_argument("--image-root", type=Path, default=None)
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
    ap.add_argument("--beta-schedule", type=str, default="linear", choices=["linear", "cosine"])
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
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--lpips-start-frac", type=float, default=0.1)
    ap.add_argument("--max-train-images", type=int, default=None)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--early-stop-patience", type=int, default=20)
    ap.add_argument("--early-stop-min-delta", type=float, default=0.0)
    ap.add_argument("--color-loss-weight", type=float, default=0.0)
    ap.add_argument("--color-loss-start-frac", type=float, default=0.6)
    ap.add_argument("--color-loss-ramp-steps", type=float, default=5000.0)
    ap.add_argument("--no-loss-plot", action="store_true")
    ap.add_argument("--loss-plot-path", type=Path, default=None)
    ap.add_argument("--no-psnr-plot", action="store_true")
    ap.add_argument("--psnr-plot-path", type=Path, default=None)
    ap.add_argument("--triplet-channel-swap-debug", action="store_true")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="If save-dir/ckpt_last.pt exists, load weights, optimizer, scaler, step, and epoch and continue.",
    )
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
            image_root=args.image_root,
            max_count=min(args.fid_count, 50_000),
            seed=args.manifest_seed,
            out_json=manifest_path,
        )

    ds_tr = CelebSketchDataset(
        args.data_root,
        PART_TRAIN,
        args.image_size,
        max_images=args.max_train_images,
        image_root=args.image_root,
    )
    ds_va = CelebSketchDataset(args.data_root, PART_VAL, args.image_size, image_root=args.image_root)
    ds_fid = CelebSketchDataset(
        args.data_root, PART_VAL, args.image_size, image_root=args.image_root, only_filenames=id_list
    )
    if len(ds_tr) == 0:
        raise SystemExit(
            "No training images — check list_eval_partition.csv, --image-root, and that JPGs "
            "exist under that folder (often nested like img_align_celeba/img_align_celeba/)."
        )

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
        raise SystemExit("Train DataLoader has no batches. Lower --batch-size.")

    epoch_budget = args.epochs * steps_per_epoch
    planned_steps = min(epoch_budget, args.max_steps) if args.max_steps and args.max_steps > 0 else epoch_budget
    lpips_start_frac = max(0.0, min(1.0, float(args.lpips_start_frac)))
    lpips_mid = int(round(planned_steps * lpips_start_frac))
    use_lpips = (not args.no_lpips) and planned_steps > 0
    args.planned_train_steps = planned_steps
    args.lpips_start_step = lpips_mid if use_lpips else -1
    color_start_frac = max(0.0, min(1.0, float(args.color_loss_start_frac)))
    color_start_step = int(round(planned_steps * color_start_frac))

    model = SketchEpsilonUNet(base=args.base_channels).to(device)
    ema = SketchEpsilonUNet(base=args.base_channels).to(device)
    ema.load_state_dict(model.state_dict())
    for q in ema.parameters():
        q.requires_grad_(False)

    ddpm = GaussianDDPM(
        args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    lip = lpips_optional(device, use_lpips)

    from torch.utils.tensorboard import SummaryWriter

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb = SummaryWriter(save_dir / "tb" / run_name)

    global_step = 0
    best_val = float("inf")
    bad_epochs = 0
    start_epoch = 0

    resume_path = save_dir / "ckpt_last.pt"
    if args.resume and resume_path.is_file():
        try:
            ck = torch.load(resume_path, map_location=device, weights_only=False)
        except TypeError:
            ck = torch.load(resume_path, map_location=device)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        global_step = int(ck.get("step", 0))
        if "epoch" in ck:
            start_epoch = int(ck["epoch"]) + 1
        else:
            start_epoch = min(global_step // max(steps_per_epoch, 1), args.epochs)
            print(
                f"[train] resume: checkpoint has no 'epoch'; inferred start_epoch={start_epoch} from step",
                flush=True,
            )
        if ck.get("optimizer"):
            opt.load_state_dict(ck["optimizer"])
        if use_amp and ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])
        best_val = float(ck.get("best_val", float("inf")))
        bad_epochs = int(ck.get("bad_epochs", 0))
        print(
            f"[train] resumed from {resume_path}  step={global_step}  start_epoch={start_epoch}  "
            f"best_val={best_val}  bad_epochs={bad_epochs}",
            flush=True,
        )
    elif args.resume:
        print(f"[train] --resume set but no {resume_path}; training from scratch.", flush=True)

    if start_epoch >= args.epochs:
        print(
            f"[train] start_epoch={start_epoch} >= epochs={args.epochs}; skipping epoch loop.",
            flush=True,
        )
    fields_m = [
        "step",
        "epoch",
        "train_loss",
        "train_psnr",
        "val_loss",
        "val_psnr",
        "epoch_seconds",
        "lr",
    ]

    def train_one_batch(step_ix: int, batch: dict[str, torch.Tensor]) -> tuple[float, float, float]:
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

        lam_lpips = lambda_phase2(step_ix, lpips_mid, float(LPIPS_RAMP_STEPS), LPIPS_LAMBDA_MAX)
        loss_fin = loss_eps
        if lip is not None and lam_lpips > 0:
            loss_fin = loss_fin + lam_lpips * lip(pred_x0.float(), ph.float()).mean()

        color_aux = torch.tensor(0.0, device=device)
        if args.color_loss_weight > 0:
            lam_color = lambda_phase2(
                step_ix,
                color_start_step,
                float(args.color_loss_ramp_steps),
                float(args.color_loss_weight),
            )
            if lam_color > 0:
                color_aux = color_ab_l1(pred_x0, ph)
                loss_fin = loss_fin + lam_color * color_aux

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
        psnr_v = psnr_pred_x0_db(pred_x0.detach(), ph)
        return float(loss_fin.item()), psnr_v, float(color_aux.item())

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

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.perf_counter()
        last_loss_f = float("nan")
        last_psnr_v = float("nan")
        bar = tqdm(dl_tr, desc=f"epoch {epoch}")
        for batch in bar:
            if args.max_steps and global_step >= args.max_steps:
                break
            loss_f, psnr_v, color_v = train_one_batch(global_step, batch)
            last_loss_f, last_psnr_v = loss_f, psnr_v
            global_step += 1
            if args.fid_every > 0 and global_step % args.fid_every == 0:
                maybe_fid()
            lr = opt.param_groups[0]["lr"]
            bar.set_postfix(loss=f"{loss_f:.4f}", psnr=f"{psnr_v:.2f}", step=global_step)
            tb.add_scalar("train/loss", loss_f, global_step)
            tb.add_scalar("train/psnr", psnr_v, global_step)
            tb.add_scalar("train/color_ab_l1", color_v, global_step)
            tb.add_scalar("train/lr", lr, global_step)
            if global_step % 50 == 0:
                csv_row(
                    metrics_path,
                    {
                        "step": str(global_step),
                        "epoch": str(epoch),
                        "train_loss": f"{loss_f:.6f}",
                        "train_psnr": f"{psnr_v:.4f}",
                        "val_loss": "",
                        "val_psnr": "",
                        "epoch_seconds": "",
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
                    channel_swap_debug=args.triplet_channel_swap_debug,
                )

        vloss, vpsnr = validation_pass(ema, ddpm, dl_va, device, min_snr)
        tb.add_scalar("val/loss", vloss, global_step)
        tb.add_scalar("val/psnr", vpsnr, global_step)
        epoch_seconds = time.perf_counter() - epoch_start
        csv_row(
            metrics_path,
            {
                "step": str(global_step),
                "epoch": str(epoch),
                "train_loss": "",
                "train_psnr": "",
                "val_loss": f"{vloss:.6f}",
                "val_psnr": f"{vpsnr:.4f}",
                "epoch_seconds": f"{epoch_seconds:.1f}",
                "lr": f"{opt.param_groups[0]['lr']:.2e}",
            },
            fields_m,
        )
        improved = vloss < (best_val - args.early_stop_min_delta)
        if improved:
            best_val = vloss
            bad_epochs = 0
        else:
            bad_epochs += 1
        ck_common = {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "step": global_step,
            "epoch": epoch,
            "best_val": best_val,
            "bad_epochs": bad_epochs,
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "args": vars(args),
        }
        if improved:
            torch.save(ck_common, save_dir / "ckpt_best.pt")
        torch.save(ck_common, save_dir / "ckpt_last.pt")
        if args.max_steps and global_step >= args.max_steps:
            break
        if args.early_stop_patience > 0 and bad_epochs >= args.early_stop_patience:
            print(
                f"[train] early stop: no EMA val_loss improvement for {bad_epochs} epoch(s).",
                flush=True,
            )
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
        channel_swap_debug=args.triplet_channel_swap_debug,
    )
    tb.close()
    if not args.no_loss_plot:
        png = args.loss_plot_path if args.loss_plot_path is not None else (save_dir / "train_loss.png")
        save_loss_plot_png(metrics_path, png)
    if not args.no_psnr_plot:
        png = args.psnr_plot_path if args.psnr_plot_path is not None else (save_dir / "train_psnr.png")
        save_psnr_plot_png(metrics_path, png)
    print("done.", global_step, "steps. best_val", best_val)


if __name__ == "__main__":
    main()
