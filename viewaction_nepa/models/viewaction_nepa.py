from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from viewaction_nepa.models.action_encoder import ActionEncoder
from viewaction_nepa.models.losses import cosine_loss, variance_regularizer, batch_stats
from viewaction_nepa.models.simple_point_encoder import SimplePointEncoder


class ViewActionNEPA(nn.Module):
    def __init__(self, encoder: nn.Module | None = None, num_actions: int = 60,
                 action_vec_dim: int = 19, latent_dim: int = 384,
                 ema_momentum: float = 0.996, inverse_weight: float = 0.1,
                 use_action_vec: bool = True, variance_weight: float = 0.0,
                 contrast_weight: float = 0.0, contrast_temperature: float = 0.1,
                 hard_contrast_weight: float = 0.0):
        super().__init__()
        self.online_encoder = encoder if encoder is not None else SimplePointEncoder(latent_dim)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.latent_dim = getattr(self.online_encoder, "output_dim", latent_dim)
        self.ema_momentum = float(ema_momentum)
        self.inverse_weight = float(inverse_weight)
        self.variance_weight = float(variance_weight)
        self.contrast_weight = float(contrast_weight)
        self.contrast_temperature = float(contrast_temperature)
        self.hard_contrast_weight = float(hard_contrast_weight)
        self.action_encoder = ActionEncoder(num_actions, action_vec_dim, self.latent_dim, use_vec=use_action_vec)
        self.predictor = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim), nn.GELU(), nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim), nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.inverse_head = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim), nn.GELU(),
            nn.Linear(self.latent_dim, num_actions),
        )

    @torch.no_grad()
    def update_target(self):
        m = self.ema_momentum
        for p_t, p_o in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            p_t.data.mul_(m).add_(p_o.data, alpha=1.0 - m)

    def encode_online(self, points: torch.Tensor) -> torch.Tensor:
        return self.online_encoder(points)

    @torch.no_grad()
    def encode_target(self, points: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(points)

    def predict_next(self, z: torch.Tensor, action_id: torch.Tensor | None, action_vec: torch.Tensor | None,
                     variant: str = "action") -> torch.Tensor:
        if variant == "no_action":
            a = torch.zeros_like(z)
        else:
            if action_id is None:
                raise ValueError("action_id required for action-conditioned prediction")
            a = self.action_encoder(action_id, action_vec)
        if variant == "action_only":
            z = torch.zeros_like(z)
        return self.predictor(torch.cat([z, a], dim=-1))

    def forward_pair(self, batch: dict, variant: str = "action", device: torch.device | str | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        if device is None:
            device = next(self.parameters()).device
        pts0 = batch["points_t"].to(device, non_blocking=True)
        pts1 = batch["points_tp1"].to(device, non_blocking=True)
        action_id = batch["action_id"].to(device, non_blocking=True)
        action_vec = batch["action_vec"].to(device, non_blocking=True)
        if variant == "shuffled_action":
            if "wrong_action_id" in batch and "wrong_action_vec" in batch:
                action_id = batch["wrong_action_id"].to(device, non_blocking=True)
                action_vec = batch["wrong_action_vec"].to(device, non_blocking=True)
            else:
                perm = torch.randperm(action_id.shape[0], device=action_id.device)
                action_id = action_id[perm]
                action_vec = action_vec[perm]
            pred_variant = "action"
        else:
            pred_variant = variant
        z0 = self.encode_online(pts0)
        with torch.no_grad():
            z1 = self.encode_target(pts1)
        z1_pred = self.predict_next(z0, action_id, action_vec, variant=pred_variant)
        loss_fwd = cosine_loss(z1_pred, z1)
        loss_contrast = torch.tensor(0.0, device=z0.device)
        if self.contrast_weight > 0 and z1_pred.size(0) > 1:
            pred_n_for_ce = F.normalize(z1_pred, dim=-1)
            tgt_n_for_ce = F.normalize(z1, dim=-1)
            logits_ce = pred_n_for_ce @ tgt_n_for_ce.T
            logits_ce = logits_ce / max(self.contrast_temperature, 1e-6)
            labels_ce = torch.arange(logits_ce.size(0), device=logits_ce.device)
            loss_contrast = F.cross_entropy(logits_ce, labels_ce)
        loss_hard = torch.tensor(0.0, device=z0.device)
        hard_acc = torch.tensor(float("nan"), device=z0.device)
        if self.hard_contrast_weight > 0 and "candidate_points" in batch:
            cand = batch["candidate_points"].to(device, non_blocking=True)
            bsz, num_cand, num_pts, xyz_dim = cand.shape
            with torch.no_grad():
                z_cand = self.encode_target(cand.reshape(bsz * num_cand, num_pts, xyz_dim))
            z_cand = z_cand.reshape(bsz, num_cand, -1)
            hard_logits = torch.einsum(
                "bd,bkd->bk",
                F.normalize(z1_pred, dim=-1),
                F.normalize(z_cand, dim=-1),
            )
            hard_logits = hard_logits / max(self.contrast_temperature, 1e-6)
            hard_labels = batch["candidate_index"].to(device, non_blocking=True)
            loss_hard = F.cross_entropy(hard_logits, hard_labels)
            hard_acc = (hard_logits.argmax(dim=-1) == hard_labels).float().mean()
        logits = self.inverse_head(torch.cat([z0.detach(), z1.detach()], dim=-1))
        true_action_id = batch["action_id"].to(device, non_blocking=True)
        loss_inv = F.cross_entropy(logits, true_action_id)
        if self.variance_weight > 0:
            loss_var = 0.5 * (variance_regularizer(z0) + variance_regularizer(z1_pred))
        else:
            loss_var = torch.tensor(0.0, device=z0.device)
        loss = (
            loss_fwd
            + self.inverse_weight * loss_inv
            + self.variance_weight * loss_var
            + self.contrast_weight * loss_contrast
            + self.hard_contrast_weight * loss_hard
        )
        with torch.no_grad():
            pred_n = F.normalize(z1_pred, dim=-1)
            tgt_n = F.normalize(z1, dim=-1)
            cur_n = F.normalize(z0, dim=-1)
            cos = (pred_n * tgt_n).sum(dim=-1)
            cos_cur = (pred_n * cur_n).sum(dim=-1)
            inv_acc = (logits.argmax(dim=-1) == true_action_id).float().mean()
            stats = {
                "loss": float(loss.detach().cpu()),
                "loss_fwd": float(loss_fwd.detach().cpu()),
                "loss_inv": float(loss_inv.detach().cpu()),
                "loss_var": float(loss_var.detach().cpu()),
                "loss_contrast": float(loss_contrast.detach().cpu()),
                "loss_hard": float(loss_hard.detach().cpu()),
                "hard_acc": float(hard_acc.detach().cpu()),
                "forward_cos_mean": float(cos.mean().detach().cpu()),
                "forward_cos_std": float(cos.std().detach().cpu()),
                "current_cos_mean": float(cos_cur.mean().detach().cpu()),
                "pred_z_var_mean": float(z1_pred.var(dim=0).mean().detach().cpu()),
                "inverse_acc": float(inv_acc.detach().cpu()),
                **{f"online_{k}": v for k, v in batch_stats(z0).items()},
                **{f"target_{k}": v for k, v in batch_stats(z1).items()},
            }
        return loss, stats
