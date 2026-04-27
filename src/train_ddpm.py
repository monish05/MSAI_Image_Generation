#!/usr/bin/env python3
"""
Train sketch-conditioned DDPM on Sketchy CSV splits.
Single GPU:  python -m src.train_ddpm
Two GPUs:    torchrun --nproc_per_node=2 -m src.train_ddpm
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .ddpm import GaussianDDPM
from .sketchy_dataset import SketchyPairDataset
from .unet_conditional import ConditionalUNet

ROOT = Path(__file__).resolve().parent.parent


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


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
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=2e-2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", help="Use bfloat16 autocast (CUDA).")
    p.add_argument("--save-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--save-every", type=int, default=2000, help="Save checkpoint every N steps (rank 0).")
    p.add_argument("--sample-every", type=int, default=5000, help="Run val sampling every N steps (rank 0).")
    p.add_argument("--max-train-steps", type=int, default=0, help="Stop after this many optimizer steps (0 = no limit).")
    args = p.parse_args()

    ddp, local_rank, rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
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

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(4, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = ConditionalUNet().to(device)
    diffusion = GaussianDDPM(
        args.timesteps, beta_start=args.beta_start, beta_end=args.beta_end
    ).to(device)

    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    opt = torch.optim.AdamW(unwrap(model).parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and torch.cuda.is_available()))

    args.save_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0

    def save_ckpt(tag: str) -> None:
        path = args.save_dir / f"ckpt_{tag}.pt"
        torch.save(
            {
                "model": unwrap(model).state_dict(),
                "opt": opt.state_dict(),
                "step": global_step,
                "args": vars(args),
            },
            path,
        )
        if rank == 0:
            print(f"saved {path}")

    @torch.no_grad()
    def sample_val() -> None:
        unwrap_m = unwrap(model)
        unwrap_m.eval()
        batch = next(iter(val_loader))
        sk = batch["sketch"].to(device)[:4]
        x_gen = diffusion.sample(unwrap_m, sk)
        unwrap_m.train()
        out_dir = args.save_dir / "samples"
        out_dir.mkdir(exist_ok=True)
        from torchvision.utils import save_image

        save_image((x_gen.clamp(-1, 1) + 1) / 2, out_dir / f"step_{global_step:07d}.png", nrow=2)
        if rank == 0:
            print(f"wrote {out_dir / f'step_{global_step:07d}.png'}")

    try:
        for epoch in range(args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            pbar = tqdm(train_loader, desc=f"epoch {epoch}", disable=rank != 0)
            for batch in pbar:
                photo = batch["photo"].to(device, non_blocking=True)
                sketch = batch["sketch"].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=bool(args.amp and torch.cuda.is_available()),
                ):
                    loss = diffusion.training_losses(model, photo, sketch)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                global_step += 1
                if rank == 0:
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                if rank == 0 and global_step % args.save_every == 0:
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
            save_ckpt("last")
    finally:
        cleanup_distributed(ddp)


if __name__ == "__main__":
    main()
