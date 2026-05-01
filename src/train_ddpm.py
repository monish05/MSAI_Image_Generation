#!/usr/bin/env python3
"""
Train sketch-conditioned DDPM on Sketchy CSV splits.
Single GPU:  python -m src.train_ddpm
Two GPUs:    torchrun --nproc_per_node=2 -m src.train_ddpm

Logs: TensorBoard (``checkpoints/tb/``) + ``metrics.csv`` (train + val loss). Resume with ``--resume``.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .ddpm import GaussianDDPM
from .sketchy_dataset import SketchyPairDataset
from .unet_conditional import ConditionalUNet

ROOT = Path(__file__).resolve().parent.parent


def _make_grad_scaler(enabled: bool):
    # torch.amp.GradScaler exists in newer PyTorch; torch.cuda.amp.GradScaler in older versions.
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_ctx(enabled: bool):
    # torch.amp.autocast exists in newer PyTorch; torch.cuda.amp.autocast in older versions.
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    return torch.cuda.amp.autocast(dtype=torch.float16, enabled=enabled)


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _migrate_legacy_metrics_csv(path: Path) -> None:
    """Upgrade step,epoch,loss files to step,epoch,train_loss,val_loss (in place)."""
    if not path.is_file():
        return
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header == ["step", "epoch", "train_loss", "val_loss"]:
            return
        if header != ["step", "epoch", "loss"]:
            return
        rows: list[list[str]] = []
        for row in reader:
            if len(row) >= 3:
                rows.append([row[0], row[1], row[2], ""])
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "epoch", "train_loss", "val_loss"])
        w.writerows(rows)


def plot_train_loss_png(metrics_csv: Path, out_png: Path) -> None:
    """Read checkpoints/metrics.csv and save train + val loss figure (matplotlib)."""
    if not metrics_csv.is_file():
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping --save-loss-plot.")
        return
    train_steps: list[int] = []
    train_losses: list[float] = []
    val_steps: list[int] = []
    val_losses: list[float] = []
    with metrics_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "step" not in reader.fieldnames:
            return
        fn = reader.fieldnames
        train_key = "train_loss" if "train_loss" in fn else "loss"
        has_val = "val_loss" in fn
        for row in reader:
            try:
                st = int(row["step"])
            except (KeyError, ValueError):
                continue
            tv = row.get(train_key, "").strip()
            if tv:
                try:
                    train_steps.append(st)
                    train_losses.append(float(tv))
                except ValueError:
                    pass
            if has_val:
                vv = row.get("val_loss", "").strip()
                if vv:
                    try:
                        val_steps.append(st)
                        val_losses.append(float(vv))
                    except ValueError:
                        pass
    if not train_steps and not val_steps:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4))
    if train_steps:
        plt.plot(train_steps, train_losses, lw=0.8, alpha=0.92, label="train")
    if val_steps:
        plt.plot(
            val_steps,
            val_losses,
            marker="o",
            linestyle="-",
            lw=1.0,
            markersize=4,
            alpha=0.9,
            label="val (epoch end)",
        )
    plt.xlabel("optimizer step")
    plt.ylabel("loss (noise MSE)")
    plt.title("Train and validation loss")
    plt.grid(True, alpha=0.3)
    if train_steps and val_steps:
        plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"wrote loss plot {out_png}")


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    msd = unwrap(model).state_dict()
    for k, ema_v in ema_model.state_dict().items():
        ema_v.copy_(ema_v * decay + msd[k] * (1.0 - decay))


def setup_distributed() -> tuple[bool, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return False, 0, 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    return True, local_rank, rank


def cleanup_distributed(ddp: bool) -> None:
    if ddp and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int = 0) -> None:
    s = seed + rank
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def log_distributed_status(ddp: bool, rank: int, local_rank: int, device: torch.device) -> None:
    world = dist.get_world_size() if ddp and dist.is_initialized() else 1
    host = os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME") or "unknown-host"
    print(
        f"[ddp] host={host} rank={rank}/{world} local_rank={local_rank} device={device}",
        flush=True,
    )
    if rank == 0:
        print(
            f"[ddp] enabled={ddp} backend={dist.get_backend() if ddp else 'none'} world_size={world}",
            flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument(
        "--train-csv",
        type=Path,
        default=ROOT / "metadata" / "sketchy_tx000" / "train.csv",
    )
    p.add_argument(
        "--val-csv",
        type=Path,
        default=ROOT / "metadata" / "sketchy_tx000" / "val.csv",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Train on H×W square crops (resized from disk). Default 64 for faster iteration.",
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="Batches each worker pre-loads (only used if num-workers > 0). Higher can hide I/O latency.",
    )
    p.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep worker processes alive between epochs (only if num-workers > 0). Cuts epoch-start stalls.",
    )
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=2e-2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", help="Use float16 autocast (CUDA).")
    p.add_argument("--save-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="If N>0, save ckpt_stepNNNNNNN.pt every N steps (rank 0). Default 0: only best/last checkpoints.",
    )
    p.add_argument("--sample-every", type=int, default=5000, help="Run val sampling every N steps (rank 0).")
    p.add_argument("--max-train-steps", type=int, default=0, help="Stop after this many optimizer steps (0 = no limit).")
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to .pt from this trainer (e.g. ckpt_last.pt) to continue training.",
    )
    p.add_argument(
        "--resume-best",
        action="store_true",
        help="Resume from save-dir/ckpt_best.pt if --resume is not provided.",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Log train loss to TensorBoard / metrics.csv every N steps (1 = every step).",
    )
    p.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard; metrics.csv is still written when log-every allows.",
    )
    p.add_argument(
        "--save-loss-plot",
        action="store_true",
        help="Save train loss PNG from metrics.csv when the run exits (rank 0).",
    )
    p.add_argument(
        "--loss-plot-path",
        type=Path,
        default=None,
        help="Output PNG for --save-loss-plot (default: save-dir/train_loss.png).",
    )
    p.add_argument(
        "--ema-decay",
        type=float,
        default=0.9999,
        help="Exponential moving average decay for model weights.",
    )
    p.add_argument(
        "--cond-drop-prob",
        type=float,
        default=0.1,
        help="Probability of dropping sketch condition for CFG-style training.",
    )
    p.add_argument(
        "--guidance-scale",
        type=float,
        default=2.0,
        help="Classifier-free guidance scale used during validation sampling.",
    )
    p.add_argument(
        "--sample-sampler",
        choices=["ddpm", "ddim"],
        default="ddim",
        help="Sampler for validation visualization images.",
    )
    p.add_argument(
        "--sample-steps",
        type=int,
        default=100,
        help="Number of denoising steps when sample-sampler=ddim.",
    )
    p.add_argument(
        "--sample-ddim-eta",
        type=float,
        default=0.0,
        help="DDIM stochasticity for validation sampling; 0 = deterministic.",
    )
    args = p.parse_args()

    ddp, local_rank, rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    log_distributed_status(ddp, rank, local_rank, device)
    if not torch.cuda.is_available():
        print("Warning: CUDA not available; training on CPU will be very slow.")
    set_seed(args.seed, rank)

    train_ds = SketchyPairDataset(
        args.train_csv, args.project_root, image_size=args.image_size
    )
    val_ds = SketchyPairDataset(
        args.val_csv, args.project_root, image_size=args.image_size
    )

    if ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    def _loader_kwargs() -> dict:
        kw: dict = {
            "num_workers": args.num_workers,
            "pin_memory": torch.cuda.is_available(),
        }
        if args.num_workers > 0:
            kw["persistent_workers"] = bool(args.persistent_workers)
            kw["prefetch_factor"] = max(2, int(args.prefetch_factor))
        return kw

    lk = _loader_kwargs()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        drop_last=True,
        **lk,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(4, args.batch_size),
        shuffle=False,
        **lk,
    )

    model = ConditionalUNet().to(device)
    ema_model = ConditionalUNet().to(device)
    ema_model.load_state_dict(model.state_dict(), strict=True)
    ema_model.eval()
    diffusion = GaussianDDPM(
        args.timesteps, beta_start=args.beta_start, beta_end=args.beta_end
    ).to(device)

    opt = torch.optim.AdamW(unwrap(model).parameters(), lr=args.lr)
    scaler = _make_grad_scaler(enabled=bool(args.amp and torch.cuda.is_available()))

    resume_path = args.resume
    if resume_path is None and args.resume_best:
        candidate = args.save_dir / "ckpt_best.pt"
        if candidate.is_file():
            resume_path = candidate
        elif rank == 0:
            print(f"--resume-best requested but not found: {candidate}")

    global_step = 0
    best_val_loss = float("inf")
    if resume_path is not None:
        if not resume_path.is_file():
            raise SystemExit(f"--resume not found: {resume_path.resolve()}")
        try:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(resume_path, map_location=device)
        unwrap(model).load_state_dict(ckpt["model"], strict=True)
        ema_state = ckpt.get("ema_model")
        if ema_state is not None:
            ema_model.load_state_dict(ema_state, strict=True)
        else:
            ema_model.load_state_dict(unwrap(model).state_dict(), strict=True)
        opt.load_state_dict(ckpt["opt"])
        s_state = ckpt.get("scaler")
        if s_state is not None:
            scaler.load_state_dict(s_state)
        global_step = int(ckpt.get("step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        if rank == 0:
            print(f"Resumed from {resume_path.resolve()} at global_step={global_step}")

    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    results_dir = args.save_dir / "results"
    tb_dir = args.save_dir / "tb"
    metrics_path = args.save_dir / "metrics.csv"
    metrics_header = ["step", "epoch", "train_loss", "val_loss"]

    writer: SummaryWriter | None = None
    if rank == 0 and not args.no_tensorboard:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(log_dir=str(tb_dir / run_name))
    if rank == 0:
        if resume_path is not None:
            _migrate_legacy_metrics_csv(metrics_path)
        if resume_path is None:
            with open(metrics_path, "w", newline="") as m:
                csv.writer(m).writerow(metrics_header)
        elif not metrics_path.is_file():
            with open(metrics_path, "w", newline="") as m:
                csv.writer(m).writerow(metrics_header)
            print(f"created new {metrics_path} (resume with no existing metrics)")
        else:
            print(f"Appending metrics rows to {metrics_path}")

    def log_loss(step: int, epoch: int, loss_val: float) -> None:
        if rank != 0:
            return
        if step % args.log_every != 0:
            return
        if writer is not None:
            writer.add_scalar("train/loss", loss_val, step)
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([step, epoch, f"{loss_val:.6f}", ""])

    def save_ckpt(tag: str) -> None:
        path = args.save_dir / f"ckpt_{tag}.pt"
        payload = {
            "model": unwrap(model).state_dict(),
            "ema_model": ema_model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": global_step,
            "best_val_loss": best_val_loss,
            "args": vars(args),
        }
        if rank == 0:
            torch.save(payload, path)
            print(f"saved {path}")

    @torch.no_grad()
    def evaluate_val_loss() -> float:
        unwrap_m = unwrap(model)
        unwrap_m.eval()
        losses: list[float] = []
        for batch in val_loader:
            photo = batch["photo"].to(device, non_blocking=True)
            sketch = batch["sketch"].to(device, non_blocking=True)
            with _autocast_ctx(enabled=bool(args.amp and torch.cuda.is_available())):
                l = diffusion.training_losses(unwrap_m, photo, sketch)
            losses.append(float(l.item()))
        unwrap_m.train()
        if not losses:
            return float("inf")
        return sum(losses) / len(losses)

    @torch.no_grad()
    def sample_to_dir(out_dir: Path, fname: str) -> None:
        if rank != 0:
            return
        from torchvision.utils import make_grid, save_image

        unwrap_m = unwrap(model)
        unwrap_m.eval()
        batch = next(iter(val_loader))
        n = min(4, batch["sketch"].shape[0])
        sk = batch["sketch"].to(device)[:n]
        photo = batch["photo"].to(device)[:n]
        if args.sample_sampler == "ddim":
            x_gen = diffusion.sample_ddim(
                ema_model,
                sk,
                guidance_scale=args.guidance_scale,
                sample_steps=args.sample_steps,
                eta=args.sample_ddim_eta,
            )
        else:
            x_gen = diffusion.sample(
                unwrap_m,
                sk,
                use_ema_model=ema_model,
                guidance_scale=args.guidance_scale,
                sampler="ddpm",
            )
        unwrap_m.train()
        out_dir.mkdir(parents=True, exist_ok=True)
        vis_gen = (x_gen.clamp(-1, 1) + 1) / 2
        vis_photo = (photo.clamp(-1, 1) + 1) / 2
        vis_sketch = ((sk.clamp(-1, 1) + 1) / 2).repeat(1, 3, 1, 1)

        # For each sample: [sketch | generated | ground-truth photo].
        triplets = torch.stack([vis_sketch, vis_gen, vis_photo], dim=1).flatten(0, 1)
        save_image(triplets, out_dir / fname, nrow=3)
        if writer is not None:
            writer.add_image("val/samples_triplet", make_grid(triplets, nrow=3), global_step)
            writer.add_image("val/samples_generated", make_grid(vis_gen, nrow=2), global_step)
        print(f"wrote {out_dir / fname}")

    @torch.no_grad()
    def sample_val() -> None:
        if rank == 0:
            sample_to_dir(args.save_dir / "samples", f"step_{global_step:07d}.png")

    @torch.no_grad()
    def export_final_results() -> None:
        sample_to_dir(results_dir, f"final_step_{global_step:07d}.png")

    try:
        for epoch in range(args.epochs):
            if args.max_train_steps and global_step >= args.max_train_steps:
                break
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            pbar = tqdm(train_loader, desc=f"epoch {epoch}", disable=rank != 0, initial=0)
            for batch in pbar:
                if args.max_train_steps and global_step >= args.max_train_steps:
                    break
                photo = batch["photo"].to(device, non_blocking=True)
                sketch = batch["sketch"].to(device, non_blocking=True)
                if args.cond_drop_prob > 0:
                    keep_mask = (
                        torch.rand((sketch.shape[0], 1, 1, 1), device=device)
                        >= args.cond_drop_prob
                    ).to(sketch.dtype)
                    sketch_cond = sketch * keep_mask
                else:
                    sketch_cond = sketch
                opt.zero_grad(set_to_none=True)
                with _autocast_ctx(enabled=bool(args.amp and torch.cuda.is_available())):
                    loss = diffusion.training_losses(model, photo, sketch_cond)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                update_ema(ema_model, model, args.ema_decay)
                global_step += 1
                loss_f = float(loss.item())
                if rank == 0:
                    pbar.set_postfix(loss=f"{loss_f:.4f}")
                log_loss(global_step, epoch, loss_f)
                if (
                    rank == 0
                    and args.save_every > 0
                    and global_step % args.save_every == 0
                ):
                    save_ckpt(f"step{global_step:07d}")
                if global_step % args.sample_every == 0:
                    if rank == 0:
                        sample_val()
                    if ddp:
                        dist.barrier()
                if args.max_train_steps and global_step >= args.max_train_steps:
                    break
            if args.max_train_steps and global_step >= args.max_train_steps:
                break
            if rank == 0:
                val_loss = evaluate_val_loss()
                print(f"epoch {epoch} val_loss={val_loss:.6f}")
                if writer is not None:
                    writer.add_scalar("val/loss", val_loss, global_step)
                with open(metrics_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [global_step, epoch, "", f"{val_loss:.6f}"]
                    )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_ckpt("best")
            if ddp:
                dist.barrier()
        if rank == 0:
            save_ckpt("last")
            export_final_results()
    finally:
        if writer is not None:
            writer.close()
        if rank == 0 and args.save_loss_plot:
            out_plot = args.loss_plot_path or (args.save_dir / "train_loss.png")
            plot_train_loss_png(args.save_dir / "metrics.csv", out_plot)
        cleanup_distributed(ddp)


if __name__ == "__main__":
    main()
