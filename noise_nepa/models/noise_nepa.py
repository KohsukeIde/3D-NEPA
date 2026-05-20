from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .simple_point_encoder import SimplePointEncoder
from .time_encoder import SinusoidalTimeEncoder
from .predictor import LatentTransitionPredictor


@dataclass
class NoiseNEPAOutput:
    loss: torch.Tensor
    loss_forward: torch.Tensor
    loss_semigroup: torch.Tensor
    cos_forward: torch.Tensor
    z_var: torch.Tensor


class NoiseNEPA(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,
        num_steps: int = 1000,
        encoder: nn.Module | None = None,
        ema_momentum: float = 0.996,
        residual_predictor: bool = True,
        semigroup_weight: float = 0.0,
        var_weight: float = 0.0,
    ):
        super().__init__()
        self.online_encoder = encoder if encoder is not None else SimplePointEncoder(embed_dim)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.time_encoder = SinusoidalTimeEncoder(num_steps=num_steps, out_dim=embed_dim)
        self.predictor = LatentTransitionPredictor(dim=embed_dim, residual=residual_predictor)
        self.ema_momentum = ema_momentum
        self.semigroup_weight = semigroup_weight
        self.var_weight = var_weight

    @torch.no_grad()
    def update_target(self):
        m = self.ema_momentum
        for pt, po in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            pt.data.mul_(m).add_(po.data, alpha=1.0 - m)

    def encode_online(self, x):
        return self.online_encoder(x)

    @torch.no_grad()
    def encode_target(self, x):
        return self.target_encoder(x)

    def predict(self, z, t, s):
        te = self.time_encoder(t, s)
        return self.predictor(z, te)

    def variance_regularizer(self, z: torch.Tensor) -> torch.Tensor:
        # VICReg-style anti-collapse regularizer, optional.
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        return torch.mean(F.relu(1.0 - std))

    def forward(self, batch: dict) -> NoiseNEPAOutput:
        x_t, x_s = batch["x_t"], batch["x_s"]
        t, s = batch["t"], batch["s"]
        z_t = self.encode_online(x_t)
        with torch.no_grad():
            z_s = self.encode_target(x_s)
        z_hat_s = self.predict(z_t, t, s)
        cos = F.cosine_similarity(z_hat_s, z_s, dim=-1)
        loss_forward = (1.0 - cos).mean()
        loss_semigroup = torch.zeros_like(loss_forward)
        if self.semigroup_weight > 0.0 and "x_r" in batch:
            x_r, r = batch["x_r"], batch["r"]
            with torch.no_grad():
                z_r = self.encode_target(x_r)
            z_hat_r_roll = self.predict(z_hat_s, s, r)
            z_hat_r_direct = self.predict(z_t, t, r)
            loss_roll_target = (1.0 - F.cosine_similarity(z_hat_r_roll, z_r, dim=-1)).mean()
            loss_direct_target = (1.0 - F.cosine_similarity(z_hat_r_direct, z_r, dim=-1)).mean()
            loss_cons = (1.0 - F.cosine_similarity(z_hat_r_roll, z_hat_r_direct.detach(), dim=-1)).mean()
            loss_semigroup = (loss_roll_target + loss_direct_target + loss_cons) / 3.0
        loss_var = self.variance_regularizer(z_t) if self.var_weight > 0.0 else torch.zeros_like(loss_forward)
        loss = loss_forward + self.semigroup_weight * loss_semigroup + self.var_weight * loss_var
        return NoiseNEPAOutput(
            loss=loss,
            loss_forward=loss_forward.detach(),
            loss_semigroup=loss_semigroup.detach(),
            cos_forward=cos.detach().mean(),
            z_var=z_t.detach().var(dim=0).mean(),
        )
