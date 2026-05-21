from __future__ import annotations
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from coverage_nepa.models.level_encoder import LevelEncoder
from coverage_nepa.models.simple_point_encoder import SimplePointEncoder


class CoverageNEPA(nn.Module):
    def __init__(self, dim: int = 256, hidden: int = 512, max_levels: int = 12, ema_momentum: float = 0.996):
        super().__init__()
        self.online_encoder = SimplePointEncoder(dim=dim)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.level_encoder = LevelEncoder(max_levels=max_levels, dim=dim)
        self.predictor = nn.Sequential(
            nn.Linear(dim * 2, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.level_only_predictor = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim),
        )
        self.ema_momentum = float(ema_momentum)

    @torch.no_grad()
    def update_target(self):
        m = self.ema_momentum
        for pt, po in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            pt.data.mul_(m).add_(po.data, alpha=1.0 - m)

    def encode_online(self, x: torch.Tensor) -> torch.Tensor:
        return self.online_encoder(x)

    @torch.no_grad()
    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(x)

    def predict(self, z_k: torch.Tensor, level_k: torch.Tensor, level_m: torch.Tensor, variant: str = "conditioned") -> torch.Tensor:
        le = self.level_encoder(level_k, level_m)
        if variant == "no_level":
            le = torch.zeros_like(le)
            out = self.predictor(torch.cat([z_k, le], dim=-1))
        elif variant == "level_only":
            out = self.level_only_predictor(le)
        else:
            out = self.predictor(torch.cat([z_k, le], dim=-1))
        return F.normalize(out, dim=-1)

    def forward(self, batch: dict, variant: str = "conditioned") -> dict:
        x_k = batch["x_k"]
        x_m = batch["x_m"]
        k = batch["level_k"]
        m = batch["level_m"]
        z_k = self.encode_online(x_k)
        with torch.no_grad():
            z_m = self.encode_target(x_m)
        if variant == "z_shuffled":
            z_k = z_k[torch.randperm(z_k.shape[0], device=z_k.device)]
        pred = self.predict(z_k, k, m, variant="level_only" if variant == "level_only" else ("no_level" if variant == "no_level" else "conditioned"))
        loss = (1.0 - (pred * z_m).sum(dim=-1)).mean()
        z_var = z_k.float().var(dim=0, unbiased=False).mean()
        return {
            "loss": loss,
            "cos": (pred * z_m).sum(dim=-1).mean().detach(),
            "current_cos": (z_k * z_m).sum(dim=-1).mean().detach(),
            "z_var": z_var.detach(),
            "z_norm": z_k.norm(dim=-1).mean().detach(),
        }
