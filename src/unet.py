"""Sketch-conditioned U-Net for epsilon prediction (pixel diffusion)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int) -> None:
        super().__init__()
        gn = lambda c: nn.GroupNorm(min(8, c), c)
        self.n1 = gn(in_ch)
        self.act = nn.SiLU()
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.n2 = gn(out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.c1(self.act(self.n1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]
        h = self.c2(self.act(self.n2(h)))
        return h + self.shortcut(x)


class SketchEpsilonUNet(nn.Module):
    """4 input channels (RGB noisy + sketch); 3-channel epsilon out."""

    def __init__(self, base: int = 96) -> None:
        super().__init__()
        self._t_sin = base
        self.t_emb_dim = base * 4
        self.t_mlp = nn.Sequential(
            nn.Linear(base, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
        )
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 4

        self.in_conv = nn.Conv2d(4, c1, 3, padding=1)
        self.d1_a = ResBlock(c1, c1, self.t_emb_dim)
        self.d1_b = ResBlock(c1, c1, self.t_emb_dim)
        self.down1 = nn.Conv2d(c1, c2, 4, stride=2, padding=1)

        self.d2_a = ResBlock(c2, c2, self.t_emb_dim)
        self.d2_b = ResBlock(c2, c2, self.t_emb_dim)
        self.down2 = nn.Conv2d(c2, c3, 4, stride=2, padding=1)

        self.d3_a = ResBlock(c3, c3, self.t_emb_dim)
        self.d3_b = ResBlock(c3, c3, self.t_emb_dim)
        self.down3 = nn.Conv2d(c3, c4, 4, stride=2, padding=1)

        self.mid_a = ResBlock(c4, c4, self.t_emb_dim)
        self.mid_b = ResBlock(c4, c4, self.t_emb_dim)

        self.up3 = nn.ConvTranspose2d(c4, c3, 4, stride=2, padding=1)
        self.u3_a = ResBlock(c3 + c3, c3, self.t_emb_dim)
        self.u3_b = ResBlock(c3, c3, self.t_emb_dim)

        self.up2 = nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1)
        self.u2_a = ResBlock(c2 + c2, c2, self.t_emb_dim)
        self.u2_b = ResBlock(c2, c2, self.t_emb_dim)

        self.up1 = nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1)
        self.u1_a = ResBlock(c1 + c1, c1, self.t_emb_dim)
        self.u1_b = ResBlock(c1, c1, self.t_emb_dim)

        gn0 = nn.GroupNorm(min(8, c1), c1)
        self.out_norm = gn0
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(c1, 3, 3, padding=1)

    def forward(self, x_t: torch.Tensor, sketch: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_mlp(timestep_embedding(t, self._t_sin))

        x = torch.cat([x_t, sketch], dim=1)

        h0 = self.in_conv(x)
        h1 = self.d1_b(self.d1_a(h0, t_emb), t_emb)
        h2 = self.d2_b(self.d2_a(self.down1(h1), t_emb), t_emb)
        h3 = self.d3_b(self.d3_a(self.down2(h2), t_emb), t_emb)
        h4 = self.mid_b(self.mid_a(self.down3(h3), t_emb), t_emb)

        u3 = self.u3_b(self.u3_a(torch.cat([self.up3(h4), h3], dim=1), t_emb), t_emb)
        u2 = self.u2_b(self.u2_a(torch.cat([self.up2(u3), h2], dim=1), t_emb), t_emb)
        u1 = self.u1_b(self.u1_a(torch.cat([self.up1(u2), h1], dim=1), t_emb), t_emb)
        return self.out_conv(self.out_act(self.out_norm(u1)))
