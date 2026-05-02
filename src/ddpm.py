"""Gaussian DDPM — epsilon loss, DDIM, optional CFG."""

from __future__ import annotations

import torch
import torch.nn as nn


class GaussianDDPM(nn.Module):
    def __init__(self, timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        self.timesteps = int(timesteps)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.alphas_cumprod[t][:, None, None, None]
        return torch.sqrt(a) * x0 + torch.sqrt(torch.clamp(1.0 - a, min=1e-8)) * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        a = self.alphas_cumprod[t][:, None, None, None]
        s2 = torch.sqrt(torch.clamp(1.0 - a, min=1e-8))
        return (x_t - s2 * eps) / torch.sqrt(a.clamp(min=1e-8))

    def training_losses(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        sketch: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        min_snr_gamma: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = x0.shape[0]
        device = x0.device
        if noise is None:
            noise = torch.randn_like(x0)
        t = torch.randint(0, self.timesteps, (b,), device=device, dtype=torch.long)
        x_t = self.q_sample(x0, t, noise)
        eps_hat = model(x_t, sketch, t)
        err = (eps_hat - noise).pow(2).flatten(1).mean(dim=-1)
        if min_snr_gamma is not None and min_snr_gamma > 0:
            acp_t = self.alphas_cumprod[t]
            snr = acp_t / (1.0 - acp_t).clamp(min=1e-8)
            w = torch.clamp(snr, max=min_snr_gamma) / snr
            loss = (err * w).mean()
        else:
            loss = err.mean()
        return loss, self.predict_x0_from_eps(x_t, t, eps_hat)

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        model: nn.Module,
        sketch: torch.Tensor,
        *,
        guidance_scale: float,
        eta: float = 0.0,
        steps: int = 50,
        generator: torch.Generator | None = None,
        null_sketch_val: float = -1.0,
    ) -> torch.Tensor:
        b, _, h, w = sketch.shape
        device = sketch.device
        alp = self.alphas_cumprod
        xt = torch.randn((b, 3, h, w), device=device, generator=generator)
        null_sk = torch.full_like(sketch, null_sketch_val)
        gs = float(guidance_scale)
        idxs = torch.linspace(self.timesteps - 1, 0, steps, device=device).long().tolist()

        for i, t_cur in enumerate(idxs):
            tb = torch.full((b,), int(t_cur), device=device, dtype=torch.long)
            eps_c = model(xt, sketch, tb)
            if gs <= 1.0:
                eps = eps_c
            else:
                eps_u = model(xt, null_sk, tb)
                eps = eps_u + gs * (eps_c - eps_u)

            a_t = alp[int(t_cur)].view(1, 1, 1, 1).clone()
            pred_x0 = self.predict_x0_from_eps(xt, tb, eps).clamp(-1.0, 1.0)
            if i == len(idxs) - 1:
                return pred_x0

            t_next = int(idxs[i + 1])
            a_nm1 = alp[t_next].view(1, 1, 1, 1)

            sig = (
                eta
                * torch.sqrt(
                    ((1 - a_nm1) / (1 - a_t).clamp(min=1e-8)).clamp(min=0)
                    * (1 - (a_t / a_nm1).clamp(max=1.0))
                )
            )
            c_dir = torch.sqrt(torch.clamp(1.0 - a_nm1 - sig**2, min=0.0))
            z = torch.randn_like(xt) if eta > 0 else torch.zeros_like(xt)
            xt = torch.sqrt(a_nm1) * pred_x0 + c_dir * eps + sig * z

        return xt
