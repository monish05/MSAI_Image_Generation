"""Conditional U-Net: noisy RGB + sketch -> predicted noise (epsilon)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(0, half, device=t.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _gn_groups(ch: int, gn: int = 8) -> int:
    g = min(gn, ch)
    while g > 1 and ch % g != 0:
        g -= 1
    return max(g, 1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_emb_dim: int) -> None:
        super().__init__()
        g1, g2 = _gn_groups(in_ch), _gn_groups(out_ch)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_ch))
        self.block1 = nn.Sequential(
            nn.GroupNorm(g1, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(g2, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, te: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.time_proj(te)[:, :, None, None]
        h = self.block2(h)
        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_emb_dim: int) -> None:
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, t_emb_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, te: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.res(x, te)
        return self.down(skip), skip


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, t_emb_dim: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res = ResBlock(in_ch + skip_ch, out_ch, t_emb_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, te: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return self.res(torch.cat([x, skip], dim=1), te)


class SelfAttention2d(nn.Module):
    """Lightweight spatial self-attention at bottleneck resolutions."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(_gn_groups(channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.norm(x).reshape(b, c, h * w)
        q, k, v = self.qkv(y).chunk(3, dim=1)
        head_dim = c // self.num_heads
        if head_dim == 0 or c % self.num_heads != 0:
            return x
        scale = head_dim ** -0.5
        q = q.reshape(b, self.num_heads, head_dim, h * w)
        k = k.reshape(b, self.num_heads, head_dim, h * w)
        v = v.reshape(b, self.num_heads, head_dim, h * w)
        attn = torch.softmax(
            torch.einsum("bhdn,bhdm->bhnm", q, k) * scale,
            dim=-1,
        )
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v).reshape(b, c, h * w)
        out = self.proj(out).reshape(b, c, h, w)
        return x + out


class ConditionalUNet(nn.Module):
    """
    4 downsamples: 256 -> 128 -> 64 -> 32 -> 16 (works for H=W divisible by 16).
    """

    def __init__(
        self,
        noise_channels: int = 3,
        sketch_channels: int = 1,
        out_channels: int = 3,
        time_dim: int = 256,
        t_emb_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.t_emb_dim = t_emb_dim
        in_ch = noise_channels + sketch_channels

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )

        self.in_conv = nn.Conv2d(in_ch, 64, 3, padding=1)
        self.r0 = ResBlock(64, 64, t_emb_dim)

        self.d1 = Down(64, 128, t_emb_dim)
        self.d2 = Down(128, 256, t_emb_dim)
        self.d3 = Down(256, 256, t_emb_dim)
        self.d4 = Down(256, 256, t_emb_dim)

        self.mid1 = ResBlock(256, 256, t_emb_dim)
        self.mid_attn = SelfAttention2d(256, num_heads=4)
        self.mid2 = ResBlock(256, 256, t_emb_dim)

        self.u1 = Up(256, 256, 256, t_emb_dim)
        self.u2 = Up(256, 256, 256, t_emb_dim)
        self.u3 = Up(256, 256, 128, t_emb_dim)
        self.u4 = Up(128, 128, 64, t_emb_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(_gn_groups(64), 64),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, 3, padding=1),
        )

    def forward(
        self, x_noisy: torch.Tensor, sketch: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        te = self.time_mlp(sinusoidal_time_embedding(t, self.time_dim))
        x = torch.cat([x_noisy, sketch], dim=1)
        h = self.in_conv(x)
        s0 = self.r0(h, te)
        h, sk1 = self.d1(s0, te)
        h, sk2 = self.d2(h, te)
        h, sk3 = self.d3(h, te)
        h, sk4 = self.d4(h, te)
        h = self.mid1(h, te)
        h = self.mid_attn(h)
        h = self.mid2(h, te)
        h = self.u1(h, sk4, te)
        h = self.u2(h, sk3, te)
        h = self.u3(h, sk2, te)
        h = self.u4(h, sk1, te)
        h = h + s0
        return self.out(h)
