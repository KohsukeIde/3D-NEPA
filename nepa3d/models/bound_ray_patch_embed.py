from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class BoundRayPatchOutput:
    """Ray tokens aligned to existing point patches."""

    q_tok: torch.Tensor
    a_tok: torch.Tensor
    has_ray: torch.Tensor


def _ensure_3d(x: Optional[torch.Tensor], last_dim: int) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.dim() == 2 and last_dim == 1:
        return x.unsqueeze(-1)
    if x.dim() == 3 and x.shape[-1] == last_dim:
        return x
    raise ValueError(f"Unexpected shape {tuple(x.shape)} for last_dim={last_dim}")


def _sphere_proxy_anchor(
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    center: torch.Tensor,
    radius: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Query-only proxy anchor by intersecting rays with a coarse bounding sphere."""
    oc = ray_o - center[:, None, :]
    a = (ray_d * ray_d).sum(dim=-1)
    b = 2.0 * (oc * ray_d).sum(dim=-1)
    c = (oc * oc).sum(dim=-1) - (radius[:, None] ** 2)
    disc = b * b - 4.0 * a * c
    has_real = disc > 0.0
    sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))

    t1 = (-b - sqrt_disc) / (2.0 * a + eps)
    t2 = (-b + sqrt_disc) / (2.0 * a + eps)
    t_pos = torch.where(t1 > eps, t1, torch.where(t2 > eps, t2, torch.zeros_like(t1)))

    t_closest = (-b) / (2.0 * a + eps)
    t_closest = torch.clamp(t_closest, min=0.0)
    t = torch.where((t_pos > 0.0) & has_real, t_pos, t_closest)
    return ray_o + t[..., None] * ray_d


class BoundRayPatchEmbed(nn.Module):
    """Bind rays to nearest point-patch centers and pool per patch."""

    def __init__(
        self,
        d_model: int,
        *,
        include_ray_normal: bool = True,
        include_ray_unc: bool = False,
        use_ray_origin: bool = False,
        assign_mode: str = "proxy_sphere",  # proxy_sphere|x_anchor
        proxy_radius_scale: float = 1.05,
        pool: str = "amax",  # amax|mean
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.include_ray_normal = bool(include_ray_normal)
        self.include_ray_unc = bool(include_ray_unc)
        self.use_ray_origin = bool(use_ray_origin)
        self.assign_mode = str(assign_mode)
        self.proxy_radius_scale = float(proxy_radius_scale)
        self.pool = str(pool)

        if self.pool not in ("amax", "mean"):
            raise ValueError(f"Unsupported pool={pool}.")

        in_q = 6 + (3 if self.use_ray_origin else 0)  # rel_anchor(3)+ray_d(3)+optional rel_o(3)
        in_a = 8 + (3 if self.include_ray_normal else 0) + (1 if self.include_ray_unc else 0)
        if self.use_ray_origin:
            in_a += 3

        def mlp(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model),
            )

        self.mlp_q = mlp(in_q)
        self.mlp_a = mlp(in_a)

    @torch.no_grad()
    def _compute_proxy_center_radius(self, centers_xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        center = centers_xyz.mean(dim=1)
        r = torch.linalg.norm(centers_xyz - center[:, None, :], dim=-1).amax(dim=1)
        r = torch.clamp(r, min=1e-3) * self.proxy_radius_scale
        return center, r

    def forward(
        self,
        *,
        centers_xyz: torch.Tensor,  # (B,P,3)
        ray_o: torch.Tensor,  # (B,R,3)
        ray_d: torch.Tensor,  # (B,R,3)
        ray_t: Optional[torch.Tensor] = None,  # (B,R) or (B,R,1)
        ray_hit: Optional[torch.Tensor] = None,  # (B,R) or (B,R,1)
        ray_n: Optional[torch.Tensor] = None,  # (B,R,3)
        ray_unc: Optional[torch.Tensor] = None,  # (B,R) or (B,R,1)
        ray_available: Optional[torch.Tensor] = None,  # (B,)
    ) -> BoundRayPatchOutput:
        if centers_xyz.dim() != 3 or centers_xyz.shape[-1] != 3:
            raise ValueError(f"centers_xyz must be (B,P,3), got {tuple(centers_xyz.shape)}")
        if ray_o.dim() != 3 or ray_o.shape[-1] != 3:
            raise ValueError(f"ray_o must be (B,R,3), got {tuple(ray_o.shape)}")
        if ray_d.dim() != 3 or ray_d.shape[-1] != 3:
            raise ValueError(f"ray_d must be (B,R,3), got {tuple(ray_d.shape)}")

        device = centers_xyz.device
        dtype = centers_xyz.dtype
        B, P, _ = centers_xyz.shape
        _, R, _ = ray_o.shape

        ray_t_ = _ensure_3d(ray_t, 1)
        ray_hit_ = _ensure_3d(ray_hit, 1)
        ray_unc_ = _ensure_3d(ray_unc, 1)

        if ray_t_ is None:
            ray_t_ = torch.zeros((B, R, 1), device=device, dtype=dtype)
        if ray_hit_ is None:
            ray_hit_ = torch.zeros((B, R, 1), device=device, dtype=dtype)
        if self.include_ray_normal and ray_n is None:
            ray_n = torch.zeros((B, R, 3), device=device, dtype=dtype)
        if self.include_ray_unc and ray_unc_ is None:
            ray_unc_ = torch.zeros((B, R, 1), device=device, dtype=dtype)

        if self.use_ray_origin:
            pc_center, _ = self._compute_proxy_center_radius(centers_xyz)
            ray_o_rel = ray_o - pc_center[:, None, :]
        else:
            ray_o_rel = None

        if self.assign_mode == "proxy_sphere":
            pc_center, pc_radius = self._compute_proxy_center_radius(centers_xyz)
            anchor = _sphere_proxy_anchor(ray_o, ray_d, pc_center, pc_radius)
        elif self.assign_mode == "x_anchor":
            anchor = ray_o + ray_t_ * ray_d
        else:
            raise ValueError(f"Unknown assign_mode={self.assign_mode}")

        assign = torch.cdist(anchor, centers_xyz).argmin(dim=-1)  # (B,R)
        centers_assigned = torch.gather(centers_xyz, 1, assign[..., None].expand(-1, -1, 3))
        rel_anchor = anchor - centers_assigned

        q_parts = [rel_anchor, ray_d]
        if ray_o_rel is not None:
            q_parts.append(ray_o_rel)
        feat_q = torch.cat(q_parts, dim=-1)
        f_q = self.mlp_q(feat_q)

        a_parts = [rel_anchor, ray_d, ray_hit_, ray_t_]
        if self.include_ray_normal and ray_n is not None:
            a_parts.append(ray_n)
        if self.include_ray_unc and ray_unc_ is not None:
            a_parts.append(ray_unc_)
        if ray_o_rel is not None:
            a_parts.append(ray_o_rel)
        feat_a = torch.cat(a_parts, dim=-1)
        f_a = self.mlp_a(feat_a)

        if self.pool == "amax":
            neg = torch.finfo(f_q.dtype).min
            out_q = torch.full((B, P, self.d_model), neg, device=device, dtype=f_q.dtype)
            out_a = torch.full((B, P, self.d_model), neg, device=device, dtype=f_a.dtype)
            idx = assign[..., None].expand(-1, -1, self.d_model)
            out_q = out_q.scatter_reduce(1, idx, f_q, reduce="amax", include_self=True)
            out_a = out_a.scatter_reduce(1, idx, f_a, reduce="amax", include_self=True)
        else:
            out_q = torch.zeros((B, P, self.d_model), device=device, dtype=f_q.dtype)
            out_a = torch.zeros((B, P, self.d_model), device=device, dtype=f_a.dtype)
            idx = assign[..., None].expand(-1, -1, self.d_model)
            out_q = out_q.scatter_add(1, idx, f_q)
            out_a = out_a.scatter_add(1, idx, f_a)

        ones = torch.ones((B, R), device=device, dtype=torch.long)
        counts = torch.zeros((B, P), device=device, dtype=torch.long).scatter_add(1, assign, ones)
        has_ray = counts > 0
        if ray_available is not None:
            has_ray = has_ray & ray_available.to(device=device).bool().view(B, 1)

        # IMPORTANT (DDP stability): use multiplicative masking instead of torch.where.
        # When a full mini-batch has no valid rays on a rank, torch.where(all_false, x, 0)
        # can disconnect x-side parameters from the graph and trigger "unused parameter"
        # reduction errors. Multiplication keeps the graph connected while zeroing outputs.
        mask = has_ray[..., None].to(dtype=out_q.dtype)
        if self.pool == "amax":
            out_q = out_q * mask
            out_a = out_a * mask
        else:
            denom = torch.clamp(counts, min=1).to(out_q.dtype).unsqueeze(-1)
            out_q = (out_q / denom) * mask
            out_a = (out_a / denom) * mask

        return BoundRayPatchOutput(q_tok=out_q, a_tok=out_a, has_ray=has_ray)

