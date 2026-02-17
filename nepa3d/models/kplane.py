from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _parse_resolutions(values: Sequence[int] | str) -> List[int]:
    if isinstance(values, str):
        out = [int(v.strip()) for v in values.split(",") if v.strip()]
    else:
        out = [int(v) for v in values]
    out = [v for v in out if v > 1]
    if not out:
        raise ValueError("plane_resolutions must contain at least one value > 1")
    return out


def _coord_to_pixel(x: torch.Tensor, res: int) -> torch.Tensor:
    # Normalized cube [-1,1] -> pixel [0, res-1]
    return ((x.clamp(-1.0, 1.0) + 1.0) * 0.5) * float(res - 1)


def _splat_bilinear(
    uv: torch.Tensor,  # [B,N,2], in [-1,1]
    feat: torch.Tensor,  # [B,N,C]
    res: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    bsz, n, ch = feat.shape
    if n <= 0:
        return torch.zeros((bsz, ch, int(res), int(res)), device=feat.device, dtype=feat.dtype)

    x = _coord_to_pixel(uv[..., 0], int(res))
    y = _coord_to_pixel(uv[..., 1], int(res))

    x0 = torch.floor(x).long().clamp(0, int(res) - 1)
    y0 = torch.floor(y).long().clamp(0, int(res) - 1)
    x1 = (x0 + 1).clamp(0, int(res) - 1)
    y1 = (y0 + 1).clamp(0, int(res) - 1)

    wx = (x - x0.float()).clamp(0.0, 1.0)
    wy = (y - y0.float()).clamp(0.0, 1.0)

    w00 = (1.0 - wx) * (1.0 - wy)
    w10 = wx * (1.0 - wy)
    w01 = (1.0 - wx) * wy
    w11 = wx * wy

    flat = int(res) * int(res)
    plane = torch.zeros((bsz, ch, flat), device=feat.device, dtype=feat.dtype)
    wsum = torch.zeros((bsz, 1, flat), device=feat.device, dtype=feat.dtype)

    def _accumulate(ix: torch.Tensor, iy: torch.Tensor, w: torch.Tensor) -> None:
        idx = (iy * int(res) + ix).long()  # [B,N]
        w_feat = w.to(dtype=feat.dtype)
        contrib = feat * w_feat.unsqueeze(-1)  # [B,N,C]
        plane.scatter_add_(2, idx.unsqueeze(1).expand(-1, ch, -1), contrib.permute(0, 2, 1))
        wsum.scatter_add_(2, idx.unsqueeze(1), w_feat.unsqueeze(1))

    _accumulate(x0, y0, w00)
    _accumulate(x1, y0, w10)
    _accumulate(x0, y1, w01)
    _accumulate(x1, y1, w11)

    plane = plane / wsum.clamp_min(float(eps))
    return plane.view(bsz, ch, int(res), int(res))


def _sample_bilinear(plane: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    # plane: [B,C,H,W], uv: [B,N,2] in [-1,1]
    grid = uv.clamp(-1.0, 1.0).unsqueeze(2)  # [B,N,1,2]
    out = F.grid_sample(
        plane,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # [B,C,N,1]
    return out.squeeze(-1).permute(0, 2, 1).contiguous()  # [B,N,C]


@dataclass
class KPlaneConfig:
    plane_resolutions: Tuple[int, ...] = (64,)
    plane_channels: int = 64
    hidden_dim: int = 128
    # sum | product | rg_product
    fusion: str = "product"
    # Only used when fusion == "rg_product".
    # If <= 0, defaults to plane_channels (group_size = 1).
    product_rank_groups: int = 0
    # sum | mean (reduce across channels within each rank group)
    product_group_reduce: str = "sum"

    @classmethod
    def from_args(
        cls,
        plane_resolutions: Sequence[int] | str,
        plane_channels: int,
        hidden_dim: int,
        fusion: str,
        product_rank_groups: int = 0,
        product_group_reduce: str = "sum",
    ) -> "KPlaneConfig":
        return cls(
            plane_resolutions=tuple(_parse_resolutions(plane_resolutions)),
            plane_channels=int(plane_channels),
            hidden_dim=int(hidden_dim),
            fusion=str(fusion),
            product_rank_groups=int(product_rank_groups),
            product_group_reduce=str(product_group_reduce),
        )

    def fused_channels_per_plane(self) -> int:
        if self.fusion in ("sum", "product"):
            return int(self.plane_channels)
        if self.fusion == "rg_product":
            rg = int(self.product_rank_groups)
            return int(self.plane_channels) if rg <= 0 else rg
        raise ValueError(f"unknown fusion: {self.fusion}")


class KPlaneRegressor(nn.Module):
    """Tri-plane / K-plane style baseline for context-conditioned distance regression.

    - Context points (xyz, dist) are splatted into XY/XZ/YZ planes.
    - Query xyz pulls plane features via bilinear sampling.
    - Fusion:
      * sum     : tri-plane style
      * product : k-plane style (Hadamard)
    """

    def __init__(self, cfg: KPlaneConfig):
        super().__init__()
        if cfg.fusion not in ("sum", "product", "rg_product"):
            raise ValueError(f"unknown fusion: {cfg.fusion}")
        if cfg.fusion == "rg_product":
            ch = int(cfg.plane_channels)
            rg = int(cfg.product_rank_groups) if int(cfg.product_rank_groups) > 0 else ch
            if ch % max(1, rg) != 0:
                raise ValueError(
                    f"plane_channels ({ch}) must be divisible by product_rank_groups ({rg}) "
                    "for fusion=rg_product"
                )
            if str(cfg.product_group_reduce) not in ("sum", "mean"):
                raise ValueError(
                    f"unknown product_group_reduce: {cfg.product_group_reduce} "
                    "(expected sum|mean)"
                )
        self.cfg = cfg
        c = int(cfg.plane_channels)
        h = int(cfg.hidden_dim)
        fused_c = int(cfg.fused_channels_per_plane())
        in_q = fused_c * len(cfg.plane_resolutions)

        self.context_encoder = nn.Sequential(
            nn.Linear(4, h),
            nn.GELU(),
            nn.Linear(h, c),
        )
        self.query_proj = nn.Sequential(
            nn.Linear(in_q, h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
        )
        self.dist_head = nn.Linear(h, 1)

    @property
    def fusion(self) -> str:
        return self.cfg.fusion

    def _rank_grouped_product(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        # Inputs: [..., C]
        if a.shape != b.shape or a.shape != c.shape:
            raise ValueError(f"shape mismatch: {a.shape=} {b.shape=} {c.shape=}")
        ch = int(a.shape[-1])
        rg = int(self.cfg.product_rank_groups) if int(self.cfg.product_rank_groups) > 0 else ch
        if ch % max(1, rg) != 0:
            raise ValueError(
                f"plane_channels ({ch}) must be divisible by product_rank_groups ({rg})"
            )
        gsz = ch // max(1, rg)
        # [..., R, G]
        a_r = a.reshape(*a.shape[:-1], rg, gsz)
        b_r = b.reshape(*b.shape[:-1], rg, gsz)
        c_r = c.reshape(*c.shape[:-1], rg, gsz)
        prod = a_r * b_r * c_r
        red = str(self.cfg.product_group_reduce)
        if red == "sum":
            return prod.sum(dim=-1)
        if red == "mean":
            return prod.mean(dim=-1)
        raise ValueError(f"unknown product_group_reduce: {red}")

    def _fuse_planes(
        self,
        fxy: torch.Tensor,
        fxz: torch.Tensor,
        fyz: torch.Tensor,
    ) -> torch.Tensor:
        if self.cfg.fusion == "sum":
            return fxy + fxz + fyz
        if self.cfg.fusion == "product":
            return fxy * fxz * fyz
        if self.cfg.fusion == "rg_product":
            return self._rank_grouped_product(fxy, fxz, fyz)
        raise ValueError(f"unknown fusion: {self.cfg.fusion}")

    def _encode_context_features(
        self,
        ctx_xyz: torch.Tensor,
        ctx_dist: torch.Tensor,
        ablate_context_dist: bool = False,
    ) -> torch.Tensor:
        if ctx_dist.dim() == 2:
            d = ctx_dist.unsqueeze(-1)
        else:
            d = ctx_dist
        if bool(ablate_context_dist):
            d = torch.zeros_like(d)
        x = torch.cat([ctx_xyz, d], dim=-1)
        return self.context_encoder(x)

    def encode_planes(
        self,
        ctx_xyz: torch.Tensor,  # [B,Nc,3]
        ctx_dist: torch.Tensor,  # [B,Nc]
        ablate_context_dist: bool = False,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        ctx_feat = self._encode_context_features(
            ctx_xyz,
            ctx_dist,
            ablate_context_dist=bool(ablate_context_dist),
        )  # [B,Nc,C]

        planes = []
        xy = ctx_xyz[..., [0, 1]]
        xz = ctx_xyz[..., [0, 2]]
        yz = ctx_xyz[..., [1, 2]]
        for r in self.cfg.plane_resolutions:
            pxy = _splat_bilinear(xy, ctx_feat, int(r))
            pxz = _splat_bilinear(xz, ctx_feat, int(r))
            pyz = _splat_bilinear(yz, ctx_feat, int(r))
            planes.append((pxy, pxz, pyz))
        return planes

    def query_raw_features(
        self,
        planes: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        qry_xyz: torch.Tensor,  # [B,Nq,3]
        ablate_query_xyz: bool = False,
    ) -> torch.Tensor:
        q = qry_xyz
        if bool(ablate_query_xyz):
            q = torch.zeros_like(q)
        qxy = q[..., [0, 1]]
        qxz = q[..., [0, 2]]
        qyz = q[..., [1, 2]]

        per_scale = []
        for pxy, pxz, pyz in planes:
            fxy = _sample_bilinear(pxy, qxy)
            fxz = _sample_bilinear(pxz, qxz)
            fyz = _sample_bilinear(pyz, qyz)
            fs = self._fuse_planes(fxy, fxz, fyz)
            per_scale.append(fs)
        return torch.cat(per_scale, dim=-1)

    def query_features(
        self,
        planes: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        qry_xyz: torch.Tensor,
        ablate_query_xyz: bool = False,
    ) -> torch.Tensor:
        raw = self.query_raw_features(
            planes,
            qry_xyz,
            ablate_query_xyz=bool(ablate_query_xyz),
        )
        return self.query_proj(raw)

    def global_descriptor_from_planes(
        self,
        planes: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        desc = []
        for pxy, pxz, pyz in planes:
            gxy = pxy.mean(dim=(-1, -2))
            gxz = pxz.mean(dim=(-1, -2))
            gyz = pyz.mean(dim=(-1, -2))
            g = self._fuse_planes(gxy, gxz, gyz)
            desc.append(g)
        return torch.cat(desc, dim=-1)

    def forward(
        self,
        ctx_xyz: torch.Tensor,
        ctx_dist: torch.Tensor,
        qry_xyz: torch.Tensor,
        ablate_query_xyz: bool = False,
        ablate_context_dist: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        planes = self.encode_planes(
            ctx_xyz,
            ctx_dist,
            ablate_context_dist=bool(ablate_context_dist),
        )
        raw = self.query_raw_features(
            planes,
            qry_xyz,
            ablate_query_xyz=bool(ablate_query_xyz),
        )
        q_feat = self.query_proj(raw)
        pred = self.dist_head(q_feat).squeeze(-1)
        return pred, q_feat, raw, planes


def build_kplane_from_ckpt(ckpt_path: str, device: str | torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.get("kplane_cfg", {})
    cfg = KPlaneConfig.from_args(
        plane_resolutions=meta.get("plane_resolutions", (64,)),
        plane_channels=int(meta.get("plane_channels", 64)),
        hidden_dim=int(meta.get("hidden_dim", 128)),
        fusion=str(meta.get("fusion", "product")),
        product_rank_groups=int(meta.get("product_rank_groups", 0)),
        product_group_reduce=str(meta.get("product_group_reduce", "sum")),
    )
    model = KPlaneRegressor(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().to(device)
    return model, ckpt
