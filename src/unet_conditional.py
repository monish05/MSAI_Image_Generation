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


class CrossAttention2d(nn.Module):
    """Spatial cross-attention: queries from features, keys/values from conditioner (sketch).

    Lets the bottleneck attend over **sketch pixels** aligned to the latent grid, separate from
    early fusion via channel concat. Typical order: ``SelfAttention2d`` then ``CrossAttention2d``.
    """

    def __init__(
        self, channels: int, cond_channels: int, num_heads: int = 4
    ) -> None:
        super().__init__()
        self.channels = channels
        self.cond_channels = cond_channels
        self.num_heads = num_heads
        self.norm_x = nn.GroupNorm(_gn_groups(channels), channels)
        self.norm_cond = nn.GroupNorm(_gn_groups(cond_channels), cond_channels)
        self.to_q = nn.Conv1d(channels, channels, kernel_size=1)
        self.to_k = nn.Conv1d(cond_channels, channels, kernel_size=1)
        self.to_v = nn.Conv1d(cond_channels, channels, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if cond.shape[-2:] != x.shape[-2:]:
            cond = F.interpolate(
                cond, size=(h, w), mode="bilinear", align_corners=False
            )
        head_dim = c // self.num_heads
        if head_dim == 0 or c % self.num_heads != 0:
            return x

        q = self.to_q(self.norm_x(x).reshape(b, c, h * w))
        ck = self.norm_cond(cond).reshape(b, self.cond_channels, h * w)
        k = self.to_k(ck)
        v = self.to_v(ck)

        scale = head_dim**-0.5
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
    4 downsamples: H -> H/2 -> ... (requires H=W divisible by 16, e.g. 64 or 256).

    ``base_channels`` scales width (default 64 = original; 96 is a stronger default for quality).
    Bottleneck width is ``base_channels * 4`` (must be divisible by ``attn_heads``).
    """

    def __init__(
        self,
        noise_channels: int = 3,
        sketch_channels: int = 1,
        out_channels: int = 3,
        base_channels: int = 64,
        time_dim: int = 256,
        t_emb_dim: int = 1024,
        attn_heads: int = 4,
        use_cross_attention: bool = True,
        num_semantic_classes: int = 0,
    ) -> None:
        super().__init__()
        self.base_channels = int(base_channels)
        self.sketch_channels = int(sketch_channels)
        self.use_cross_attention = bool(use_cross_attention)
        self.num_semantic_classes = int(num_semantic_classes)
        #: Index into ``class_emb`` for CFG / dropout (no category).
        self.null_class_index = int(num_semantic_classes)
        self.time_dim = time_dim
        self.t_emb_dim = t_emb_dim
        in_ch = noise_channels + sketch_channels

        c0 = self.base_channels
        c1 = self.base_channels * 2
        c2 = self.base_channels * 4
        if c2 % attn_heads != 0:
            raise ValueError(
                f"base_channels*4 ({c2}) must be divisible by attn_heads ({attn_heads})"
            )

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )
        self.class_emb: nn.Embedding | None
        if self.num_semantic_classes > 0:
            self.class_emb = nn.Embedding(self.num_semantic_classes + 1, t_emb_dim)
        else:
            self.class_emb = None

        self.in_conv = nn.Conv2d(in_ch, c0, 3, padding=1)
        self.r0 = ResBlock(c0, c0, t_emb_dim)

        self.d1 = Down(c0, c1, t_emb_dim)
        self.d2 = Down(c1, c2, t_emb_dim)
        self.d3 = Down(c2, c2, t_emb_dim)
        self.d4 = Down(c2, c2, t_emb_dim)

        self.mid1 = ResBlock(c2, c2, t_emb_dim)
        self.mid_attn = SelfAttention2d(c2, num_heads=attn_heads)
        self.mid_cross_attn = (
            CrossAttention2d(
                c2, cond_channels=sketch_channels, num_heads=attn_heads
            )
            if self.use_cross_attention
            else None
        )
        self.mid2 = ResBlock(c2, c2, t_emb_dim)

        self.u1 = Up(c2, c2, c2, t_emb_dim)
        self.u2 = Up(c2, c2, c2, t_emb_dim)
        self.u3 = Up(c2, c2, c1, t_emb_dim)
        self.u4 = Up(c1, c1, c0, t_emb_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(_gn_groups(c0), c0),
            nn.SiLU(),
            nn.Conv2d(c0, out_channels, 3, padding=1),
        )

    def forward(
        self,
        x_noisy: torch.Tensor,
        sketch: torch.Tensor,
        t: torch.Tensor,
        class_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = x_noisy.shape[0]
        device = x_noisy.device
        te = self.time_mlp(sinusoidal_time_embedding(t, self.time_dim))
        if self.class_emb is not None:
            if class_id is None:
                cid = torch.full(
                    (b,),
                    self.null_class_index,
                    device=device,
                    dtype=torch.long,
                )
            else:
                # Valid indices: 0 .. num_semantic_classes-1 (labels) or null_class_index (CFG/drop).
                cid = class_id.long().view(b).clamp(0, self.null_class_index)
            te = te + self.class_emb(cid)

        x = torch.cat([x_noisy, sketch], dim=1)
        h = self.in_conv(x)
        s0 = self.r0(h, te)
        h, sk1 = self.d1(s0, te)
        h, sk2 = self.d2(h, te)
        h, sk3 = self.d3(h, te)
        h, sk4 = self.d4(h, te)
        h = self.mid1(h, te)
        h = self.mid_attn(h)
        if self.mid_cross_attn is not None:
            h = self.mid_cross_attn(h, sketch)
        h = self.mid2(h, te)
        h = self.u1(h, sk4, te)
        h = self.u2(h, sk3, te)
        h = self.u3(h, sk2, te)
        h = self.u4(h, sk1, te)
        h = h + s0
        return self.out(h)
