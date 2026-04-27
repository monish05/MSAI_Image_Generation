"""DDPM linear schedule, q-sample, training loss, ancestral sampling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
    b = t.shape[0]
    out = a.gather(0, t.clamp(0, a.numel() - 1))
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class GaussianDDPM(nn.Module):
    def __init__(
        self,
        timesteps: int,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__()
        self.timesteps = int(timesteps)
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_mean_coef1", coef1)
        self.register_buffer("posterior_mean_coef2", coef2)

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            _extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    def training_losses(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        sketch: torch.Tensor,
        noise: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = x0.shape[0]
        if noise is None:
            noise = torch.randn_like(x0)
        if t is None:
            t = torch.randint(0, self.timesteps, (b,), device=x0.device, dtype=torch.long)
        x_t = self.q_sample(x0, t, noise)
        pred = model(x_t, sketch, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def p_sample_step(
        self, model: nn.Module, x: torch.Tensor, sketch: torch.Tensor, t: int
    ) -> torch.Tensor:
        b = x.shape[0]
        device = x.device
        t_b = torch.full((b,), t, device=device, dtype=torch.long)
        beta_t = _extract(self.betas, t_b, x.shape)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alphas_cumprod, t_b, x.shape)
        sqrt_recip = _extract(self.sqrt_recip_alphas, t_b, x.shape)
        eps = model(x, sketch, t_b)
        model_mean = sqrt_recip * (x - beta_t / sqrt_one_minus * eps)
        if t == 0:
            return model_mean
        noise = torch.randn_like(x)
        log_var = _extract(self.posterior_log_variance_clipped, t_b, x.shape)
        return model_mean + torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        sketch: torch.Tensor,
        use_ema_model: nn.Module | None = None,
    ) -> torch.Tensor:
        m = use_ema_model or model
        b, _, h, w = sketch.shape
        x = torch.randn(b, 3, h, w, device=sketch.device, dtype=sketch.dtype)
        for ti in reversed(range(self.timesteps)):
            x = self.p_sample_step(m, x, sketch, ti)
        return x
