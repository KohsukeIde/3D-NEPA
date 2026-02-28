"""Patch-NEPA with QueryNEPA-parity Q/A tokenization.

This module keeps the Patch backbone (fps+knn or serial patchify) while
restoring QueryNEPA-side behaviors that were previously bypassed:
- qa_tokens (Q/A split vs fused token),
- qa_layout (interleave / split + optional SEP),
- type_id construction (for type-aware dual-mask),
- answer-only NEPA loss masking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .causal_transformer import CausalTransformer
from .encdec_transformer import EncoderDecoderTransformer
from .bound_ray_patch_embed import BoundRayPatchEmbed
from .point_patch_embed import PointPatchEmbed
from .serial_patch_embed import SerialPatchEmbed
from ..token.tokenizer import (
    TYPE_A_POINT,
    TYPE_A_RAY,
    TYPE_BOS,
    TYPE_EOS,
    TYPE_POINT,
    TYPE_RAY,
    TYPE_MISSING_RAY,
    TYPE_Q_POINT,
    TYPE_Q_RAY,
    TYPE_SEP,
    TYPE_VOCAB_SIZE,
)


@dataclass
class PatchNepaOutput:
    z: torch.Tensor
    h: torch.Tensor
    z_hat: torch.Tensor
    type_id: torch.Tensor
    centers_xyz: torch.Tensor
    group_idx: Optional[torch.Tensor] = None


class AnswerPatchEmbed(nn.Module):
    """Embed per-point answer features into a single patch answer token."""

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        hidden_dim: Optional[int] = None,
        n_layers: int = 2,
        pool: Literal["max", "mean"] = "max",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert in_dim > 0
        assert n_layers >= 1
        hidden = int(hidden_dim or d_model)

        layers: list[nn.Module] = []
        d = int(in_dim)
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.GELU())
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            d = hidden
        layers.append(nn.Linear(d, int(d_model)))
        self.mlp = nn.Sequential(*layers)
        self.pool = str(pool)

    def forward(self, ans_feat: torch.Tensor, group_idx: torch.Tensor) -> torch.Tensor:
        assert ans_feat.dim() == 3
        assert group_idx.dim() == 3
        B, _, C = ans_feat.shape
        Bg, P, K = group_idx.shape
        assert B == Bg

        ans4 = ans_feat.unsqueeze(1).expand(-1, P, -1, -1)  # (B,P,N,C)
        idx = group_idx.unsqueeze(-1).expand(-1, -1, -1, C)  # (B,P,K,C)
        grouped = torch.gather(ans4, dim=2, index=idx)  # (B,P,K,C)

        flat = grouped.reshape(B * P * K, C)
        flat = self.mlp(flat)
        grouped_h = flat.view(B, P, K, -1)  # (B,P,K,D)

        if self.pool == "max":
            return grouped_h.max(dim=2).values
        if self.pool == "mean":
            return grouped_h.mean(dim=2)
        raise ValueError(f"unknown answer pool={self.pool}")


class PatchTransformerNepa(nn.Module):
    """Patch-token NEPA model with QueryNEPA-parity Q/A options."""

    def __init__(
        self,
        *,
        # patchify
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
        qk_norm_affine: int = 0,
        qk_norm_bias: int = 0,
        layerscale_value: float = 1e-5,
        rope_theta: float = 100.0,
        use_gated_mlp: int = 0,
        hidden_act: str = "gelu",
        backbone_mode: str = "nepa2d",
        # QueryNEPA-parity Q/A
        qa_tokens: int = 1,
        qa_layout: str = "interleave",  # interleave|split|split_sep
        qa_sep_token: bool = True,
        qa_fuse: str = "add",  # add|concat
        use_pt_dist: bool = True,
        use_pt_grad: bool = False,
        answer_mlp_layers: int = 2,
        answer_pool: str = "max",
        # embeddings / arch
        max_len: int = 4096,
        nepa2d_pos: bool = True,
        type_specific_pos: bool = False,
        type_pos_max_len: int = 4096,
        pos_mode: str = "center_mlp",  # center_mlp|none
        encdec_arch: bool = False,
        # optional ray patch binding
        use_ray_patch: bool = False,
        include_ray_normal: bool = True,
        include_ray_unc: bool = False,
        use_ray_origin: bool = False,
        ray_assign_mode: str = "proxy_sphere",  # proxy_sphere|x_anchor
        ray_proxy_radius_scale: float = 1.05,
        ray_pool_mode: str = "amax",  # amax|mean
    ) -> None:
        super().__init__()

        if qa_layout == "split_sep":
            qa_layout = "split"
            qa_sep_token = True
        if int(qa_tokens) not in (0, 1):
            raise ValueError(f"qa_tokens must be 0/1, got {qa_tokens}")
        if str(qa_layout) not in ("interleave", "split"):
            raise ValueError(f"qa_layout must be interleave/split, got {qa_layout}")

        self.d_model = int(d_model)
        self.use_normals = bool(use_normals)
        self.qa_tokens = int(qa_tokens)
        self.qa_layout = str(qa_layout)
        self.qa_sep_token = bool(qa_sep_token)
        self.qa_fuse = str(qa_fuse)
        self.nepa2d_pos = bool(nepa2d_pos)
        self.type_specific_pos = bool(type_specific_pos)
        self.type_pos_max_len = int(type_pos_max_len)
        self.pos_mode = str(pos_mode)
        self.encdec_arch = bool(encdec_arch)
        self.use_pt_dist = bool(use_pt_dist)
        self.use_pt_grad = bool(use_pt_grad)
        self.backbone_mode = str(backbone_mode)
        self.use_ray_patch = bool(use_ray_patch)

        if num_groups is None:
            num_groups = max(1, int(round(float(n_point) / float(group_size))))

        patch_embed = str(patch_embed)
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

        # Patch answer embedding from {dist, grad}.
        ans_in = (1 if self.use_pt_dist else 0) + (3 if self.use_pt_grad else 0)
        self.answer_embed: Optional[AnswerPatchEmbed]
        if ans_in > 0:
            self.answer_embed = AnswerPatchEmbed(
                in_dim=ans_in,
                d_model=self.d_model,
                n_layers=int(answer_mlp_layers),
                pool=str(answer_pool),
                dropout=float(dropout),
            )
        else:
            self.answer_embed = None

        if self.qa_tokens == 0 and self.qa_fuse == "concat":
            self.qa_fuse_proj = nn.Linear(2 * self.d_model, self.d_model)
        else:
            self.qa_fuse_proj = None

        self.ray_patch_embed: Optional[BoundRayPatchEmbed]
        if self.use_ray_patch:
            self.ray_patch_embed = BoundRayPatchEmbed(
                d_model=self.d_model,
                include_ray_normal=bool(include_ray_normal),
                include_ray_unc=bool(include_ray_unc),
                use_ray_origin=bool(use_ray_origin),
                assign_mode=str(ray_assign_mode),
                proxy_radius_scale=float(ray_proxy_radius_scale),
                pool=str(ray_pool_mode),
            )
        else:
            self.ray_patch_embed = None

        # Special tokens
        self.bos_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.eos_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.sep_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.trunc_normal_(self.bos_token, std=0.02)
        nn.init.trunc_normal_(self.eos_token, std=0.02)
        nn.init.trunc_normal_(self.sep_token, std=0.02)

        self.type_emb = nn.Embedding(int(TYPE_VOCAB_SIZE), self.d_model)

        self.pos_emb = nn.Parameter(torch.zeros(1, int(max_len), self.d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        if self.nepa2d_pos:
            with torch.no_grad():
                self.pos_emb.zero_()

        if self.type_specific_pos:
            self.type_pos_emb = nn.Embedding(int(TYPE_VOCAB_SIZE) * int(self.type_pos_max_len), self.d_model)
            nn.init.trunc_normal_(self.type_pos_emb.weight, std=0.02)
        else:
            self.type_pos_emb = None

        if self.pos_mode == "center_mlp":
            self.center_mlp = nn.Sequential(
                nn.Linear(3, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model),
            )
        elif self.pos_mode == "none":
            self.center_mlp = None
        else:
            raise ValueError(f"unknown pos_mode={self.pos_mode}")

        if self.encdec_arch:
            self.backbone = EncoderDecoderTransformer(
                d_model=self.d_model,
                nhead=int(n_heads),
                num_encoder_layers=int(n_layers),
                num_decoder_layers=int(n_layers),
                dim_feedforward=int(self.d_model * float(mlp_ratio)),
                dropout=float(dropout),
                drop_path=float(drop_path_rate),
                src_causal=False,
            )
        else:
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
                use_gated_mlp=bool(use_gated_mlp),
                hidden_act=str(hidden_act),
                backbone_impl="legacy" if self.backbone_mode == "vanilla" else "nepa2d",
            )

        self.pred_head = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.d_model))

    def _apply_type_pos_emb(self, x: torch.Tensor, type_id: torch.Tensor) -> torch.Tensor:
        if self.type_pos_emb is None:
            return x
        B, T, _ = x.shape
        local = torch.zeros((B, T), device=x.device, dtype=torch.long)
        for ty in range(int(TYPE_VOCAB_SIZE)):
            m = type_id == ty
            if not bool(m.any()):
                continue
            local[m] = torch.cumsum(m.long(), dim=1)[m] - 1
        if int(local.max().item()) >= int(self.type_pos_max_len):
            raise ValueError(
                f"type_pos_max_len={self.type_pos_max_len} too small for local_max={int(local.max().item())}"
            )
        idx = type_id * int(self.type_pos_max_len) + local
        return x + self.type_pos_emb(idx)

    def _add_embeddings(
        self,
        tokens: torch.Tensor,
        type_id: torch.Tensor,
        centers_seq: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, T, _ = tokens.shape
        out = tokens + self.type_emb(type_id)
        if T > self.pos_emb.shape[1]:
            raise ValueError(f"T={T} exceeds max_len={self.pos_emb.shape[1]}")
        out = out + self.pos_emb[:, :T, :]
        out = self._apply_type_pos_emb(out, type_id)

        if self.center_mlp is not None and centers_seq is not None:
            cen = self.center_mlp(centers_seq)
            special = (type_id == TYPE_BOS) | (type_id == TYPE_EOS) | (type_id == TYPE_SEP)
            cen = cen.masked_fill(special.unsqueeze(-1), 0.0)
            out = out + cen
        return out

    def _build_seq_qa0(
        self,
        q_tok: torch.Tensor,
        a_tok: torch.Tensor,
        centers_xyz: torch.Tensor,
        q_ray_tok: Optional[torch.Tensor] = None,
        a_ray_tok: Optional[torch.Tensor] = None,
        ray_has: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, P, D = q_tok.shape
        if self.qa_fuse == "add":
            patch_tok = q_tok + a_tok
        elif self.qa_fuse == "concat":
            if self.qa_fuse_proj is None:
                raise RuntimeError("qa_fuse='concat' requires qa_fuse_proj")
            patch_tok = self.qa_fuse_proj(torch.cat([q_tok, a_tok], dim=-1))
        else:
            raise ValueError(f"unknown qa_fuse={self.qa_fuse}")

        parts: list[torch.Tensor] = [self.bos_token.expand(B, 1, D), patch_tok]
        types: list[torch.Tensor] = [
            torch.full((B, 1), int(TYPE_BOS), device=q_tok.device, dtype=torch.long),
            torch.full((B, P), int(TYPE_POINT), device=q_tok.device, dtype=torch.long),
        ]
        centers_parts: list[torch.Tensor] = [
            torch.zeros((B, 1, 3), device=q_tok.device, dtype=centers_xyz.dtype),
            centers_xyz,
        ]

        if q_ray_tok is not None or a_ray_tok is not None:
            if q_ray_tok is None or a_ray_tok is None:
                raise ValueError("q_ray_tok and a_ray_tok must both be provided for qa_tokens=0")
            if self.qa_fuse == "add":
                ray_tok = q_ray_tok + a_ray_tok
            elif self.qa_fuse == "concat":
                if self.qa_fuse_proj is None:
                    raise RuntimeError("qa_fuse='concat' requires qa_fuse_proj")
                ray_tok = self.qa_fuse_proj(torch.cat([q_ray_tok, a_ray_tok], dim=-1))
            else:
                raise ValueError(f"unknown qa_fuse={self.qa_fuse}")
            if ray_has is None:
                ray_has = torch.ones((B, P), device=q_tok.device, dtype=torch.bool)
            parts.append(ray_tok)
            types.append(
                torch.where(
                    ray_has,
                    torch.full((B, P), int(TYPE_RAY), device=q_tok.device, dtype=torch.long),
                    torch.full((B, P), int(TYPE_MISSING_RAY), device=q_tok.device, dtype=torch.long),
                )
            )
            centers_parts.append(centers_xyz)

        parts.append(self.eos_token.expand(B, 1, D))
        types.append(torch.full((B, 1), int(TYPE_EOS), device=q_tok.device, dtype=torch.long))
        centers_parts.append(torch.zeros((B, 1, 3), device=q_tok.device, dtype=centers_xyz.dtype))

        tokens = torch.cat(parts, dim=1)
        type_id = torch.cat(types, dim=1)
        centers_seq = torch.cat(centers_parts, dim=1)
        return tokens, type_id, centers_seq

    def _build_seq_qa1_interleave(
        self,
        q_tok: torch.Tensor,
        a_tok: torch.Tensor,
        centers_xyz: torch.Tensor,
        q_ray_tok: Optional[torch.Tensor] = None,
        a_ray_tok: Optional[torch.Tensor] = None,
        ray_has: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, P, D = q_tok.shape
        qa = torch.stack([q_tok, a_tok], dim=2).reshape(B, 2 * P, D)

        parts: list[torch.Tensor] = [self.bos_token.expand(B, 1, D), qa]
        types: list[torch.Tensor] = [
            torch.full((B, 1), int(TYPE_BOS), device=q_tok.device, dtype=torch.long),
            torch.stack(
                [
                    torch.full((B, P), int(TYPE_Q_POINT), device=q_tok.device, dtype=torch.long),
                    torch.full((B, P), int(TYPE_A_POINT), device=q_tok.device, dtype=torch.long),
                ],
                dim=2,
            ).reshape(B, 2 * P),
        ]
        centers_parts: list[torch.Tensor] = [
            torch.zeros((B, 1, 3), device=q_tok.device, dtype=centers_xyz.dtype),
            centers_xyz.repeat_interleave(2, dim=1),
        ]

        if q_ray_tok is not None or a_ray_tok is not None:
            if q_ray_tok is None or a_ray_tok is None:
                raise ValueError("q_ray_tok and a_ray_tok must both be provided for interleave ray mode")
            ray_qa = torch.stack([q_ray_tok, a_ray_tok], dim=2).reshape(B, 2 * P, D)
            if ray_has is None:
                ray_has = torch.ones((B, P), device=q_tok.device, dtype=torch.bool)
            q_type = torch.where(
                ray_has,
                torch.full((B, P), int(TYPE_Q_RAY), device=q_tok.device, dtype=torch.long),
                torch.full((B, P), int(TYPE_MISSING_RAY), device=q_tok.device, dtype=torch.long),
            )
            a_type = torch.where(
                ray_has,
                torch.full((B, P), int(TYPE_A_RAY), device=q_tok.device, dtype=torch.long),
                torch.full((B, P), int(TYPE_MISSING_RAY), device=q_tok.device, dtype=torch.long),
            )
            parts.append(ray_qa)
            types.append(torch.stack([q_type, a_type], dim=2).reshape(B, 2 * P))
            centers_parts.append(centers_xyz.repeat_interleave(2, dim=1))

        parts.append(self.eos_token.expand(B, 1, D))
        types.append(torch.full((B, 1), int(TYPE_EOS), device=q_tok.device, dtype=torch.long))
        centers_parts.append(torch.zeros((B, 1, 3), device=q_tok.device, dtype=centers_xyz.dtype))

        tokens = torch.cat(parts, dim=1)
        type_id = torch.cat(types, dim=1)
        centers_seq = torch.cat(centers_parts, dim=1)
        return tokens, type_id, centers_seq

    def _build_seq_qa1_split(
        self,
        q_tok: torch.Tensor,
        a_tok: torch.Tensor,
        centers_xyz: torch.Tensor,
        q_ray_tok: Optional[torch.Tensor] = None,
        a_ray_tok: Optional[torch.Tensor] = None,
        ray_has: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, P, D = q_tok.shape
        parts: list[torch.Tensor] = [self.bos_token.expand(B, 1, D), q_tok]
        types: list[torch.Tensor] = [
            torch.full((B, 1), int(TYPE_BOS), device=q_tok.device, dtype=torch.long),
            torch.full((B, P), int(TYPE_Q_POINT), device=q_tok.device, dtype=torch.long),
        ]
        centers_parts: list[torch.Tensor] = [
            torch.zeros((B, 1, 3), device=q_tok.device, dtype=centers_xyz.dtype),
            centers_xyz,
        ]

        if (q_ray_tok is None) ^ (a_ray_tok is None):
            raise ValueError("q_ray_tok and a_ray_tok must be both set or both None for split layout")

        if q_ray_tok is not None:
            if ray_has is None:
                ray_has = torch.ones((B, P), device=q_tok.device, dtype=torch.bool)
            q_types = torch.where(
                ray_has,
                torch.full((B, P), int(TYPE_Q_RAY), device=q_tok.device, dtype=torch.long),
                torch.full((B, P), int(TYPE_MISSING_RAY), device=q_tok.device, dtype=torch.long),
            )
            parts.append(q_ray_tok)
            types.append(q_types)
            centers_parts.append(centers_xyz)

        if self.qa_sep_token:
            parts.append(self.sep_token.expand(B, 1, D))
            types.append(torch.full((B, 1), int(TYPE_SEP), device=q_tok.device, dtype=torch.long))
            centers_parts.append(torch.zeros((B, 1, 3), device=q_tok.device, dtype=centers_xyz.dtype))

        parts.append(a_tok)
        types.append(torch.full((B, P), int(TYPE_A_POINT), device=q_tok.device, dtype=torch.long))
        centers_parts.append(centers_xyz)

        if a_ray_tok is not None:
            if ray_has is None:
                ray_has = torch.ones((B, P), device=q_tok.device, dtype=torch.bool)
            a_types = torch.where(
                ray_has,
                torch.full((B, P), int(TYPE_A_RAY), device=q_tok.device, dtype=torch.long),
                torch.full((B, P), int(TYPE_MISSING_RAY), device=q_tok.device, dtype=torch.long),
            )
            parts.append(a_ray_tok)
            types.append(a_types)
            centers_parts.append(centers_xyz)

        parts.append(self.eos_token.expand(B, 1, D))
        types.append(torch.full((B, 1), int(TYPE_EOS), device=q_tok.device, dtype=torch.long))
        centers_parts.append(torch.zeros((B, 1, 3), device=q_tok.device, dtype=centers_xyz.dtype))

        tokens = torch.cat(parts, dim=1)
        type_id = torch.cat(types, dim=1)
        centers_seq = torch.cat(centers_parts, dim=1)
        return tokens, type_id, centers_seq

    def forward(
        self,
        *,
        pt_xyz: Optional[torch.Tensor] = None,
        pt_n: Optional[torch.Tensor] = None,
        pt_dist: Optional[torch.Tensor] = None,
        pt_grad: Optional[torch.Tensor] = None,
        points_xyz: Optional[torch.Tensor] = None,
        points_dist: Optional[torch.Tensor] = None,
        points_grad: Optional[torch.Tensor] = None,
        ray_o: Optional[torch.Tensor] = None,
        ray_d: Optional[torch.Tensor] = None,
        ray_t: Optional[torch.Tensor] = None,
        ray_hit: Optional[torch.Tensor] = None,
        ray_n: Optional[torch.Tensor] = None,
        ray_unc: Optional[torch.Tensor] = None,
        ray_available: Optional[torch.Tensor] = None,
        is_causal: bool | int = True,
        dual_mask_near: float = 0.0,
        dual_mask_far: float = 0.0,
        dual_mask_window: int = 0,
        dual_mask_type_aware: int | bool = 0,
    ) -> PatchNepaOutput:
        if pt_xyz is None:
            pt_xyz = points_xyz
        if pt_dist is None:
            pt_dist = points_dist
        if pt_grad is None:
            pt_grad = points_grad
        if pt_xyz is None:
            raise ValueError("pt_xyz/points_xyz is required")

        patch_out = self.patch_embed(pt_xyz, pt_n if self.use_normals else None)
        q_tok = patch_out.tokens
        centers_xyz = patch_out.centers_xyz
        group_idx = patch_out.group_idx

        if self.answer_embed is None:
            a_tok = q_tok
        else:
            feats: list[torch.Tensor] = []
            if self.use_pt_dist:
                if pt_dist is None:
                    raise ValueError("use_pt_dist=True but pt_dist is None")
                if pt_dist.dim() == 2:
                    pt_dist = pt_dist.unsqueeze(-1)
                feats.append(pt_dist)
            if self.use_pt_grad:
                if pt_grad is None:
                    raise ValueError("use_pt_grad=True but pt_grad is None")
                if pt_grad.dim() != 3 or pt_grad.size(-1) != 3:
                    raise ValueError(f"pt_grad must be (B,N,3), got {tuple(pt_grad.shape)}")
                feats.append(pt_grad)
            ans_feat = torch.cat(feats, dim=-1) if len(feats) > 1 else feats[0]
            a_tok = self.answer_embed(ans_feat, group_idx)

        q_ray_tok: Optional[torch.Tensor] = None
        a_ray_tok: Optional[torch.Tensor] = None
        ray_has: Optional[torch.Tensor] = None
        if self.use_ray_patch:
            if self.ray_patch_embed is None:
                raise RuntimeError("use_ray_patch=True but ray_patch_embed is None")
            if ray_o is None or ray_d is None:
                raise ValueError("use_ray_patch=True requires ray_o and ray_d")
            ray_out = self.ray_patch_embed(
                centers_xyz=centers_xyz,
                ray_o=ray_o,
                ray_d=ray_d,
                ray_t=ray_t,
                ray_hit=ray_hit,
                ray_n=ray_n,
                ray_unc=ray_unc,
                ray_available=ray_available,
            )
            q_ray_tok = ray_out.q_tok
            a_ray_tok = ray_out.a_tok
            ray_has = ray_out.has_ray

        if self.qa_tokens == 0:
            tokens, type_id, centers_seq = self._build_seq_qa0(
                q_tok,
                a_tok,
                centers_xyz,
                q_ray_tok=q_ray_tok,
                a_ray_tok=a_ray_tok,
                ray_has=ray_has,
            )
        else:
            if self.qa_layout == "interleave":
                tokens, type_id, centers_seq = self._build_seq_qa1_interleave(
                    q_tok,
                    a_tok,
                    centers_xyz,
                    q_ray_tok=q_ray_tok,
                    a_ray_tok=a_ray_tok,
                    ray_has=ray_has,
                )
            else:
                tokens, type_id, centers_seq = self._build_seq_qa1_split(
                    q_tok,
                    a_tok,
                    centers_xyz,
                    q_ray_tok=q_ray_tok,
                    a_ray_tok=a_ray_tok,
                    ray_has=ray_has,
                )

        z = self._add_embeddings(tokens, type_id, centers_seq)

        if isinstance(self.backbone, EncoderDecoderTransformer):
            if self.qa_tokens != 1 or self.qa_layout != "split":
                raise ValueError("encdec_arch expects qa_tokens=1 and qa_layout='split'")
            if self.qa_sep_token:
                sep_pos = (type_id == int(TYPE_SEP)).int().argmax(dim=1)
                sep = int(sep_pos[0].item())
            else:
                sep = 1 + q_tok.shape[1]
            enc = z[:, :sep, :]
            dec = z[:, sep:, :]
            enc_out, dec_out = self.backbone(enc, dec, enc_xyz=None)
            h = torch.cat([enc_out, dec_out], dim=1)
        else:
            h = self.backbone(
                z,
                is_causal=bool(is_causal),
                type_id=type_id,
                dual_mask_near=float(dual_mask_near),
                dual_mask_far=float(dual_mask_far),
                dual_mask_window=int(dual_mask_window),
                dual_mask_type_aware=int(dual_mask_type_aware),
            )

        z_hat = self.pred_head(h)
        return PatchNepaOutput(
            z=z,
            h=h,
            z_hat=z_hat,
            type_id=type_id,
            centers_xyz=centers_xyz,
            group_idx=group_idx,
        )

    @staticmethod
    def nepa_loss(
        z: torch.Tensor,
        z_hat: torch.Tensor,
        type_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if z.size(1) < 2:
            return z.new_zeros(())
        pred = z_hat[:, :-1, :]
        tgt = z[:, 1:, :].detach()

        if type_id is None:
            mask = torch.ones(pred.shape[:2], device=z.device, dtype=torch.bool)
        else:
            tgt_ty = type_id[:, 1:]
            has_answer = bool((tgt_ty == int(TYPE_A_POINT)).any() or (tgt_ty == int(TYPE_A_RAY)).any())
            if has_answer:
                mask = (tgt_ty == int(TYPE_A_POINT)) | (tgt_ty == int(TYPE_A_RAY))
            else:
                mask = (
                    (tgt_ty != int(TYPE_BOS))
                    & (tgt_ty != int(TYPE_SEP))
                    & (tgt_ty != int(TYPE_EOS))
                    & (tgt_ty != int(TYPE_MISSING_RAY))
                )
        if not bool(mask.any()):
            return pred.new_zeros(())
        loss = 1.0 - F.cosine_similarity(pred, tgt, dim=-1)
        return loss[mask].mean()


class PatchTransformerNepaClassifier(nn.Module):
    """Classification-only wrapper over PatchTransformerNepa core.

    This keeps pretrain and finetune as separate classes while reusing the same
    PatchNEPA backbone weights via composition (`self.core`).
    """

    def __init__(
        self,
        num_classes: int,
        *,
        pooling: str = "cls_max",  # mean/mean_q | cls | cls_max
        head_mode: str = "pointmae_mlp",  # linear | pointmae_mlp
        head_hidden_dim: int = 256,
        head_dropout: float = 0.5,
        is_causal: bool = False,
        ft_sequence_mode: str = "qa_zeroa",  # qa_zeroa | q_only
        **nepa_kwargs,
    ) -> None:
        super().__init__()
        assert pooling in {"mean", "mean_q", "cls", "cls_max"}
        assert head_mode in {"linear", "pointmae_mlp"}
        assert ft_sequence_mode in {"qa_zeroa", "q_only"}

        self.core = PatchTransformerNepa(**nepa_kwargs)
        self.d_model = int(self.core.d_model)
        self.pooling = str(pooling)
        self.head_mode = str(head_mode)
        self.head_hidden_dim = int(head_hidden_dim)
        self.head_dropout = float(head_dropout)
        self.is_causal = bool(is_causal)
        self.ft_sequence_mode = str(ft_sequence_mode)

        if self.pooling == "cls_max":
            head_in_dim = 2 * self.d_model
        else:
            head_in_dim = self.d_model

        if self.head_mode == "linear":
            self.cls_head = nn.Linear(head_in_dim, int(num_classes))
        else:
            self.cls_head = nn.Sequential(
                nn.Linear(head_in_dim, self.head_hidden_dim),
                nn.BatchNorm1d(self.head_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.head_dropout),
                nn.Linear(self.head_hidden_dim, self.head_hidden_dim),
                nn.BatchNorm1d(self.head_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.head_dropout),
                nn.Linear(self.head_hidden_dim, int(num_classes)),
            )

    @staticmethod
    def _query_token_mask(type_id: torch.Tensor) -> torch.Tensor:
        # Primary query-token mask for both qa_tokens=1 and qa_tokens=0 cases.
        mask = (
            (type_id == int(TYPE_Q_POINT))
            | (type_id == int(TYPE_Q_RAY))
            | (type_id == int(TYPE_POINT))
            | (type_id == int(TYPE_RAY))
        )
        has_any = mask.any(dim=1, keepdim=True)
        # Fallback: remove specials and explicit answer-only/missing tokens.
        fallback = (
            (type_id != int(TYPE_BOS))
            & (type_id != int(TYPE_EOS))
            & (type_id != int(TYPE_SEP))
            & (type_id != int(TYPE_A_POINT))
            & (type_id != int(TYPE_A_RAY))
            & (type_id != int(TYPE_MISSING_RAY))
        )
        return torch.where(has_any, mask, fallback)

    def _pool_features(self, h: torch.Tensor, type_id: torch.Tensor) -> torch.Tensor:
        # BOS is always sequence index 0 in PatchNEPA builders.
        cls_feat = h[:, 0, :]
        q_mask = self._query_token_mask(type_id)

        if self.pooling == "cls":
            return cls_feat

        q_count = q_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=h.dtype)
        q_sum = (h * q_mask.unsqueeze(-1).to(dtype=h.dtype)).sum(dim=1)
        q_mean = q_sum / q_count

        if self.pooling in {"mean", "mean_q"}:
            return q_mean

        neg_inf = torch.finfo(h.dtype).min
        q_vals = h.masked_fill(~q_mask.unsqueeze(-1), neg_inf)
        q_max = q_vals.max(dim=1).values
        # If a row has no valid query token, fallback to cls token.
        valid = q_mask.any(dim=1, keepdim=True)
        q_max = torch.where(valid, q_max, cls_feat)
        return torch.cat([cls_feat, q_max], dim=-1)

    def _build_q_only_sequence(
        self,
        xyz: torch.Tensor,
        normals: Optional[torch.Tensor],
        ray_o: Optional[torch.Tensor],
        ray_d: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patch_out = self.core.patch_embed(xyz, normals if self.core.use_normals else None)
        q_tok = patch_out.tokens
        centers_xyz = patch_out.centers_xyz
        B, P, D = q_tok.shape
        dev = q_tok.device

        parts: list[torch.Tensor] = [self.core.bos_token.expand(B, 1, D), q_tok]
        types: list[torch.Tensor] = [
            torch.full((B, 1), int(TYPE_BOS), device=dev, dtype=torch.long),
            torch.full((B, P), int(TYPE_Q_POINT), device=dev, dtype=torch.long),
        ]
        centers_parts: list[torch.Tensor] = [
            torch.zeros((B, 1, 3), device=dev, dtype=centers_xyz.dtype),
            centers_xyz,
        ]

        if self.core.use_ray_patch:
            if self.core.ray_patch_embed is None:
                raise RuntimeError("use_ray_patch=True but ray_patch_embed is None")
            if ray_o is None or ray_d is None:
                raise ValueError("q_only mode with ray patch requires ray_o and ray_d")
            ray_out = self.core.ray_patch_embed(
                centers_xyz=centers_xyz,
                ray_o=ray_o,
                ray_d=ray_d,
                ray_t=None,
                ray_hit=None,
                ray_n=None,
                ray_unc=None,
                ray_available=None,
            )
            q_ray_type = torch.where(
                ray_out.has_ray,
                torch.full((B, P), int(TYPE_Q_RAY), device=dev, dtype=torch.long),
                torch.full((B, P), int(TYPE_MISSING_RAY), device=dev, dtype=torch.long),
            )
            parts.append(ray_out.q_tok)
            types.append(q_ray_type)
            centers_parts.append(centers_xyz)

        parts.append(self.core.eos_token.expand(B, 1, D))
        types.append(torch.full((B, 1), int(TYPE_EOS), device=dev, dtype=torch.long))
        centers_parts.append(torch.zeros((B, 1, 3), device=dev, dtype=centers_xyz.dtype))
        return torch.cat(parts, dim=1), torch.cat(types, dim=1), torch.cat(centers_parts, dim=1)

    def forward_features(
        self,
        xyz: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        ray_o: Optional[torch.Tensor] = None,
        ray_d: Optional[torch.Tensor] = None,
        ray_t: Optional[torch.Tensor] = None,
        ray_hit: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Query-only classification protocol:
        # - qa_zeroa: keep QA layout but feed zero-valued point answers
        # - q_only: remove all A tokens at finetune and use only query sequence
        # In both cases, ray answer values are intentionally ignored.
        _ = ray_t
        _ = ray_hit

        if self.ft_sequence_mode == "q_only":
            tokens, type_id, centers_seq = self._build_q_only_sequence(
                xyz=xyz,
                normals=normals,
                ray_o=ray_o,
                ray_d=ray_d,
            )
            z = self.core._add_embeddings(tokens, type_id, centers_seq)
            if isinstance(self.core.backbone, EncoderDecoderTransformer):
                # Encoder-only inference in q_only mode; keep EOS as a 1-token decoder stub.
                enc = z[:, :-1, :]
                dec = z[:, -1:, :]
                enc_out, _ = self.core.backbone(enc, dec, enc_xyz=None)
                h = enc_out
                type_for_pool = type_id[:, :-1]
            else:
                h = self.core.backbone(
                    z,
                    is_causal=self.is_causal,
                    type_id=type_id,
                    dual_mask_near=0.0,
                    dual_mask_far=0.0,
                    dual_mask_window=0,
                    dual_mask_type_aware=0,
                )
                type_for_pool = type_id
            h = self.core.pred_head[0](h)
            return self._pool_features(h, type_for_pool)

        pt_dist = torch.zeros((xyz.shape[0], xyz.shape[1], 1), dtype=xyz.dtype, device=xyz.device)
        out = self.core(
            pt_xyz=xyz,
            pt_n=normals,
            pt_dist=pt_dist,
            ray_o=ray_o,
            ray_d=ray_d,
            ray_t=None,
            ray_hit=None,
            is_causal=self.is_causal,
            dual_mask_near=0.0,
            dual_mask_far=0.0,
            dual_mask_window=0,
            dual_mask_type_aware=0,
        )
        h = self.core.pred_head[0](out.h)
        return self._pool_features(h, out.type_id)

    def forward(
        self,
        xyz: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        ray_o: Optional[torch.Tensor] = None,
        ray_d: Optional[torch.Tensor] = None,
        ray_t: Optional[torch.Tensor] = None,
        ray_hit: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat = self.forward_features(
            xyz,
            normals,
            ray_o=ray_o,
            ray_d=ray_d,
            ray_t=ray_t,
            ray_hit=ray_hit,
        )
        return self.cls_head(feat)


# Backward-compatible alias
PatchNepa = PatchTransformerNepa
