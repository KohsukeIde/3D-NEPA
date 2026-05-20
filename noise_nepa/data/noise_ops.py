from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class NoiseSchedule:
    num_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    schedule: str = "linear"

    def __post_init__(self):
        if self.schedule != "linear":
            raise ValueError(f"Only linear beta schedule is implemented in smoke track, got {self.schedule}")
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_steps)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)

    def alpha(self, t: torch.Tensor, device=None) -> torch.Tensor:
        ab = self.alpha_bar.to(device or t.device)
        return ab[t.long().clamp(0, self.num_steps - 1)]


def add_gaussian_noise(
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """DDPM-style coordinate noise with point correspondence preserved.

    x0: B N 3 or N 3
    t: B or scalar tensor
    """
    if x0.dim() == 2:
        x0_in = x0.unsqueeze(0)
        squeeze = True
    else:
        x0_in = x0
        squeeze = False
    if t.dim() == 0:
        t = t.view(1).expand(x0_in.shape[0])
    if noise is None:
        noise = torch.randn(
            x0_in.shape,
            generator=generator,
            device=x0_in.device,
            dtype=x0_in.dtype,
        )
    a = schedule.alpha(t.to(x0_in.device), x0_in.device).view(-1, 1, 1)
    xt = torch.sqrt(a) * x0_in + torch.sqrt(1.0 - a) * noise
    return xt.squeeze(0) if squeeze else xt


def sample_tsr(batch_size: int, num_steps: int, device, min_gap: int = 50, high_min: float = 0.35) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample t > s > r for one-step and semigroup smoke.

    Uses integers in [0, num_steps). Higher t means noisier.
    """
    low_high = max(int(num_steps * high_min), min_gap * 2 + 1)
    t = torch.randint(low_high, num_steps, (batch_size,), device=device)
    s_min = torch.clamp(t - int(num_steps * 0.35), min=min_gap)
    s_max = torch.clamp(t - min_gap, min=min_gap)
    # Sample s by uniform fraction between s_min and s_max.
    u = torch.rand(batch_size, device=device)
    s = (s_min.float() + u * (s_max.float() - s_min.float()).clamp_min(1.0)).long()
    r = torch.clamp(s - min_gap, min=0)
    u2 = torch.rand(batch_size, device=device)
    r = (u2 * r.float()).long()
    return t.long(), s.long(), r.long()


def normalize_point_cloud(x: torch.Tensor) -> torch.Tensor:
    """Center and scale point cloud to roughly unit sphere."""
    c = x.mean(dim=-2, keepdim=True)
    x = x - c
    scale = torch.norm(x, dim=-1).amax(dim=-1, keepdim=True).clamp_min(1e-6)
    if x.dim() == 3:
        scale = scale.unsqueeze(-1)
    else:
        scale = scale.unsqueeze(-1)
    return x / scale
