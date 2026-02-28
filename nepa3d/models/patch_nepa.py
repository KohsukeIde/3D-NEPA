"""Patch-token NEPA pretraining model.

Stage-2 target:
- Build point patches (serial or FPS+kNN),
- optionally fuse ray context into each point patch (Option A),
- train with next-embedding prediction on patch-token sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .causal_transformer import CausalTransformer
from .point_patch_embed import PointPatchEmbed
from .serial_patch_embed import SerialPatchEmbed


@dataclass
class PatchNepaOutput:
    z: torch.Tensor
    z_hat: torch.Tensor
    centers_xyz: torch.Tensor


class PatchTransformerNepa(nn.Module):
    def __init__(
        self,
        *,
        # patching
        patch_embed: str = "fps_knn",
        n_point: int = 1024,
        group_size: int = 32,
        num_groups: Optional[int] = 64,
        serial_order: str = "morton",
        serial_bits: int = 10,
        serial_shuffle_within_patch: int = 0,
        use_normals: bool = False,
        # transformer
        d_model: int = 384,
        n_layers: int = 12,
        n_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        qk_norm: int = 1,
        qk_norm_affine: int = 1,
        qk_norm_bias: int = 1,
        layerscale_value: float = 0.0,
        rope_theta: float = 10000.0,
        use_gated_mlp: int = 0,
        hidden_act: str = "gelu",
        backbone_mode: str = "nepa2d",
        # positional encoding (vanilla path only)
        pos_mode: str = "center_mlp_once",
        pos_mlp_hidden_mult: float = 2.0,
        pos_add_times: int = 1,
        # ray-patch binding
        use_ray_patch: bool = False,
        ray_hit_threshold: float = 0.5,
        ray_miss_t: float = 4.0,
        ray_pool_k_max: int = 32,
        ray_pool_mode: str = "mean",
        ray_fuse: str = "add",
        # sequence
        use_bos: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.backbone_mode = str(backbone_mode)
        self.pos_mode = str(pos_mode)
        self.pos_add_times = int(pos_add_times)
        self.use_normals = bool(use_normals)
        self.use_bos = bool(use_bos)

        self.use_ray_patch = bool(use_ray_patch)
        self.ray_hit_threshold = float(ray_hit_threshold)
        self.ray_miss_t = float(ray_miss_t)
        self.ray_pool_k_max = int(ray_pool_k_max)
        self.ray_pool_mode = str(ray_pool_mode)
        self.ray_fuse = str(ray_fuse)

        patch_embed = str(patch_embed)
        if num_groups is None:
            num_groups = max(1, int(round(float(n_point) / float(group_size))))

        if patch_embed == "serial":
            self.patch_embed = SerialPatchEmbed(
                embed_dim=self.d_model,
                group_size=int(group_size),
                order=str(serial_order),
                bits=int(serial_bits),
                shuffle_within_patch=bool(serial_shuffle_within_patch),
                use_normals=self.use_normals,
            )
        elif patch_embed in ("pointgpt", "fps_knn"):
            self.patch_embed = PointPatchEmbed(
                num_groups=int(num_groups),
                group_size=int(group_size),
                embed_dim=self.d_model,
                use_normals=self.use_normals,
            )
        else:
            raise ValueError(f"unknown patch_embed={patch_embed}")

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model)) if self.use_bos else None
        self.cls_pos = None

        if self.backbone_mode == "vanilla":
            if self.pos_mode == "center_mlp_once":
                hidden = int(self.d_model * float(pos_mlp_hidden_mult))
                self.pos_mlp = nn.Sequential(
                    nn.Linear(3, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, self.d_model),
                )
                self.pos_emb = None
            elif self.pos_mode == "center_linear":
                self.pos_emb = nn.Linear(3, self.d_model)
                self.pos_mlp = None
            elif self.pos_mode == "none":
                self.pos_emb = None
                self.pos_mlp = None
            else:
                raise ValueError(f"unknown pos_mode={self.pos_mode}")
            self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.d_model)) if self.use_bos else None
        else:
            self.pos_emb = None
            self.pos_mlp = None
            self.cls_pos = None

        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.cls_pos is not None:
            nn.init.trunc_normal_(self.cls_pos, std=0.02)

        rope_prefix_tokens = 1 if (self.backbone_mode == "nepa2d" and self.use_bos) else 0
        self.backbone = CausalTransformer(
            d_model=self.d_model,
            nhead=int(n_heads),
            num_layers=int(n_layers),
            mlp_ratio=float(mlp_ratio),
            dropout=float(dropout),
            drop_path=float(drop_path_rate),
            qk_norm=bool(qk_norm),
            qk_norm_affine=bool(qk_norm_affine),
            qk_norm_bias=bool(qk_norm_bias),
            layerscale_value=float(layerscale_value),
            rope_theta=float(rope_theta),
            rope_prefix_tokens=int(rope_prefix_tokens),
            use_gated_mlp=bool(use_gated_mlp),
            hidden_act=str(hidden_act),
            backbone_impl="legacy" if self.backbone_mode == "vanilla" else "nepa2d",
        )
        self.norm = nn.LayerNorm(self.d_model)

        if self.use_ray_patch:
            self.ray_encoder = nn.Sequential(
                nn.Linear(8, 128),
                nn.GELU(),
                nn.Linear(128, self.d_model),
            )
            if self.ray_fuse == "concat":
                self.ray_fuse_proj = nn.Linear(2 * self.d_model, self.d_model)
            else:
                self.ray_fuse_proj = None
        else:
            self.ray_encoder = None
            self.ray_fuse_proj = None

        self.pred_head = nn.Linear(self.d_model, self.d_model)

    def _apply_pos_embed(self, x: torch.Tensor, centers_xyz: torch.Tensor, *, has_bos: bool) -> torch.Tensor:
        if self.backbone_mode != "vanilla":
            return x

        if self.pos_mode == "none":
            if has_bos and self.cls_pos is not None:
                x[:, :1, :] = x[:, :1, :] + self.cls_pos
            return x

        if self.pos_mode == "center_linear":
            pos = self.pos_emb(centers_xyz)
        else:
            pos = self.pos_mlp(centers_xyz)

        if has_bos:
            if self.cls_pos is not None:
                x[:, :1, :] = x[:, :1, :] + self.cls_pos
            for _ in range(max(1, self.pos_add_times)):
                x[:, 1:, :] = x[:, 1:, :] + pos
            return x

        for _ in range(max(1, self.pos_add_times)):
            x = x + pos
        return x

    def _build_ray_anchor(
        self,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        ray_t: torch.Tensor,
        ray_hit: torch.Tensor,
    ) -> torch.Tensor:
        if ray_hit.dim() == 3 and ray_hit.size(-1) == 1:
            ray_hit = ray_hit.squeeze(-1)
        if ray_t.dim() == 3 and ray_t.size(-1) == 1:
            ray_t = ray_t.squeeze(-1)
        hit_mask = (ray_hit > float(self.ray_hit_threshold)).to(ray_o.dtype).unsqueeze(-1)
        x_hit = ray_o + ray_t.unsqueeze(-1) * ray_d
        x_miss = ray_o + float(self.ray_miss_t) * ray_d
        return hit_mask * x_hit + (1.0 - hit_mask) * x_miss

    def _pool_rays_to_patches(
        self,
        centers_xyz: torch.Tensor,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        ray_t: torch.Tensor,
        ray_hit: torch.Tensor,
        ray_available: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, n_group, _ = centers_xyz.shape
        out = centers_xyz.new_zeros((bsz, n_group, self.d_model))
        if (self.ray_encoder is None) or (ray_o is None) or (ray_o.numel() == 0):
            return out

        ray_anchor = self._build_ray_anchor(ray_o, ray_d, ray_t, ray_hit)

        for b in range(bsz):
            if (ray_available is not None) and (int(ray_available[b].item()) == 0):
                continue
            n_ray = int(ray_anchor[b].shape[0])
            if n_ray <= 0:
                continue
            dist = torch.cdist(ray_anchor[b].unsqueeze(0), centers_xyz[b].unsqueeze(0)).squeeze(0)
            assign = dist.argmin(dim=-1)

            ray_hit_b = ray_hit[b].squeeze(-1) if ray_hit[b].dim() == 2 else ray_hit[b]
            ray_t_b = ray_t[b].squeeze(-1) if ray_t[b].dim() == 2 else ray_t[b]
            ctr = centers_xyz[b][assign]
            feat_in = torch.cat(
                [
                    ray_anchor[b] - ctr,
                    ray_d[b],
                    ray_hit_b.unsqueeze(-1),
                    ray_t_b.unsqueeze(-1),
                ],
                dim=-1,
            )
            feat = self.ray_encoder(feat_in)

            for g in range(n_group):
                m = assign == g
                if not bool(m.any()):
                    continue
                v = feat[m]
                if self.ray_pool_k_max > 0 and v.shape[0] > self.ray_pool_k_max:
                    sel = torch.randperm(v.shape[0], device=v.device)[: self.ray_pool_k_max]
                    v = v[sel]
                if self.ray_pool_mode == "mean":
                    out[b, g] = v.mean(dim=0)
                else:
                    out[b, g] = v.max(dim=0).values
        return out

    def forward(
        self,
        *,
        pt_xyz: torch.Tensor,
        pt_n: Optional[torch.Tensor] = None,
        pt_dist: Optional[torch.Tensor] = None,
        ray_o: Optional[torch.Tensor] = None,
        ray_d: Optional[torch.Tensor] = None,
        ray_t: Optional[torch.Tensor] = None,
        ray_hit: Optional[torch.Tensor] = None,
        ray_available: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> PatchNepaOutput:
        del pt_dist  # reserved for future variants
        bsz = int(pt_xyz.shape[0])
        pt_feat = pt_n if (self.use_normals and pt_n is not None) else None

        patch_out = self.patch_embed(pt_xyz, pt_feat)
        x = patch_out.tokens
        centers_xyz = patch_out.centers_xyz

        if self.use_ray_patch and (ray_o is not None) and (ray_o.numel() > 0):
            if ray_hit is None:
                ray_hit = ray_o.new_zeros(ray_o.shape[:-1])
            if ray_t is None:
                ray_t = ray_o.new_zeros(ray_o.shape[:-1])
            ray_feat = self._pool_rays_to_patches(
                centers_xyz,
                ray_o,
                ray_d,
                ray_t,
                ray_hit,
                ray_available=ray_available,
            )
            if self.ray_fuse == "add":
                x = x + ray_feat
            else:
                x = self.ray_fuse_proj(torch.cat([x, ray_feat], dim=-1))

        if self.use_bos:
            bos = self.cls_token.expand(bsz, -1, -1)
            x = torch.cat([bos, x], dim=1)

        x = self._apply_pos_embed(x, centers_xyz, has_bos=self.use_bos)
        z = x
        h = self.backbone(x, is_causal=bool(is_causal))
        h = self.norm(h)
        z_hat = self.pred_head(h)
        return PatchNepaOutput(z=z, z_hat=z_hat, centers_xyz=centers_xyz)

    @staticmethod
    def nepa_loss(z: torch.Tensor, z_hat: torch.Tensor) -> torch.Tensor:
        if z.size(1) < 2:
            return z.new_zeros(())
        # Stop-gradient target branch to avoid trivial collapse.
        target = z[:, 1:, :].detach()
        pred = z_hat[:, :-1, :]
        sim = F.cosine_similarity(pred, target, dim=-1)
        return (1.0 - sim).mean()
