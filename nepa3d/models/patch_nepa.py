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
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .causal_transformer import CausalTransformer
from .encdec_transformer import EncoderDecoderTransformer
from .bound_ray_patch_embed import BoundRayPatchEmbed
from .point_patch_embed import PointPatchEmbed
from .serial_patch_embed import SerialPatchEmbed, morton3d_codes, serialize_indices
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
    TYPE_SEP_CTX,
    TYPE_SEP_QA,
    TYPE_VOCAB_SIZE,
)


@dataclass
class PatchNepaOutput:
    tokens: torch.Tensor
    z: torch.Tensor
    h: torch.Tensor
    z_hat: torch.Tensor
    type_id: torch.Tensor
    centers_xyz: torch.Tensor
    group_idx: Optional[torch.Tensor] = None
    q_keep_ratio: Optional[torch.Tensor] = None


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
        self.in_dim = int(in_dim)
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
        patch_local_encoder: str = "mlp",  # mlp | pointmae_conv
        patch_fps_random_start: bool = False,
        n_point: int = 1024,
        group_size: int = 32,
        num_groups: Optional[int] = 64,
        serial_order: str = "morton",
        serial_bits: int = 10,
        serial_shuffle_within_patch: int = 0,
        patch_order_mode: str = "none",  # none|...|sample:<m1,m2,...> (rev_/_rev wrappers supported)
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
        answer_in_dim: Optional[int] = None,  # explicit answer feature dim override
        q_mask_mode: str = "mask_token",  # mask_token|zero
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
        ray_assign_mode: str = "proxy_sphere",  # proxy_sphere|x_anchor|independent_fps_knn
        ray_proxy_radius_scale: float = 1.05,
        ray_pool_mode: str = "amax",  # amax|mean
        ray_num_groups: int = 32,
        ray_group_size: int = 32,
    ) -> None:
        super().__init__()

        if qa_layout == "split_sep":
            qa_layout = "split"
            qa_sep_token = True
        if int(qa_tokens) not in (0, 1):
            raise ValueError(f"qa_tokens must be 0/1, got {qa_tokens}")
        if str(qa_layout) not in ("interleave", "split"):
            raise ValueError(f"qa_layout must be interleave/split, got {qa_layout}")
        if str(q_mask_mode) not in ("mask_token", "zero"):
            raise ValueError(f"q_mask_mode must be mask_token/zero, got {q_mask_mode}")
        if str(backbone_mode) not in ("nepa2d", "vanilla", "pointmae"):
            raise ValueError(f"backbone_mode must be nepa2d/vanilla/pointmae, got {backbone_mode}")
        self.d_model = int(d_model)
        self.use_normals = bool(use_normals)
        self.qa_tokens = int(qa_tokens)
        self.qa_layout = str(qa_layout)
        self.qa_sep_token = bool(qa_sep_token)
        self.qa_fuse = str(qa_fuse)
        self.q_mask_mode = str(q_mask_mode)
        self.patch_order_mode = str(patch_order_mode)
        self.nepa2d_pos = bool(nepa2d_pos)
        self.type_specific_pos = bool(type_specific_pos)
        self.type_pos_max_len = int(type_pos_max_len)
        self.pos_mode = str(pos_mode)
        self.encdec_arch = bool(encdec_arch)
        self.use_pt_dist = bool(use_pt_dist)
        self.use_pt_grad = bool(use_pt_grad)
        self.answer_in_dim_override = None if answer_in_dim is None else int(answer_in_dim)
        self.backbone_mode = str(backbone_mode)
        self.pointmae_backbone = self.backbone_mode == "pointmae"
        self.use_ray_patch = bool(use_ray_patch)
        if self.encdec_arch and self.pointmae_backbone:
            raise ValueError("backbone_mode='pointmae' requires encdec_arch=0")

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
                local_encoder=str(patch_local_encoder),
                fps_random_start=bool(patch_fps_random_start),
            )
        else:
            raise ValueError(f"unknown patch_embed={patch_embed}")

        # Patch answer embedding from {dist, grad}.
        ans_in_auto = (1 if self.use_pt_dist else 0) + (3 if self.use_pt_grad else 0)
        ans_in = int(ans_in_auto if self.answer_in_dim_override is None else self.answer_in_dim_override)
        if ans_in < 0:
            raise ValueError(f"answer_in_dim must be >=0, got {ans_in}")
        self.answer_in_dim = int(ans_in)
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
                ray_num_groups=int(ray_num_groups),
                ray_group_size=int(ray_group_size),
            )
        else:
            self.ray_patch_embed = None

        # Special tokens
        self.bos_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.eos_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        # SEP roles: context/query boundary vs query/answer boundary.
        self.sep_ctx_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.sep_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.q_mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.trunc_normal_(self.bos_token, std=0.02)
        nn.init.trunc_normal_(self.eos_token, std=0.02)
        nn.init.trunc_normal_(self.sep_ctx_token, std=0.02)
        nn.init.trunc_normal_(self.sep_token, std=0.02)
        nn.init.trunc_normal_(self.q_mask_token, std=0.02)

        self.type_emb: Optional[nn.Embedding]
        if self.pointmae_backbone:
            self.type_emb = None
        else:
            self.type_emb = nn.Embedding(int(TYPE_VOCAB_SIZE), self.d_model)

        self.pos_emb = nn.Parameter(torch.zeros(1, int(max_len), self.d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        if self.nepa2d_pos and (not self.pointmae_backbone):
            with torch.no_grad():
                self.pos_emb.zero_()

        if self.type_specific_pos and (not self.pointmae_backbone):
            self.type_pos_emb = nn.Embedding(int(TYPE_VOCAB_SIZE) * int(self.type_pos_max_len), self.d_model)
            nn.init.trunc_normal_(self.type_pos_emb.weight, std=0.02)
        else:
            self.type_pos_emb = None

        if self.pointmae_backbone:
            # Point-MAE parity path: absolute positional embedding is injected
            # per block via CausalTransformer(pos=...), not center-MLP.
            self.center_mlp = None
        elif self.pos_mode == "center_mlp":
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
            if self.backbone_mode == "vanilla":
                backbone_impl = "legacy"
            elif self.backbone_mode == "pointmae":
                backbone_impl = "pointmae"
            else:
                backbone_impl = "nepa2d"
            self.backbone = CausalTransformer(
                d_model=self.d_model,
                nhead=int(n_heads),
                num_layers=int(n_layers),
                mlp_ratio=float(mlp_ratio),
                dropout=float(dropout),
                drop_path=float(drop_path_rate),
                qkv_bias=(not self.pointmae_backbone),
                qk_norm=bool(qk_norm),
                qk_norm_affine=bool(qk_norm_affine),
                qk_norm_bias=bool(qk_norm_bias),
                layerscale_value=float(layerscale_value),
                rope_theta=float(rope_theta),
                use_gated_mlp=bool(use_gated_mlp),
                hidden_act=str(hidden_act),
                backbone_impl=backbone_impl,
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
        if self.pointmae_backbone:
            return tokens + self._pointmae_pos(tokens.size(1))
        B, T, _ = tokens.shape
        if self.type_emb is None:
            raise RuntimeError("type_emb is None in non-pointmae embedding path")
        out = tokens + self.type_emb(type_id)
        if T > self.pos_emb.shape[1]:
            raise ValueError(f"T={T} exceeds max_len={self.pos_emb.shape[1]}")
        out = out + self.pos_emb[:, :T, :]
        out = self._apply_type_pos_emb(out, type_id)

        if self.center_mlp is not None and centers_seq is not None:
            cen = self.center_mlp(centers_seq)
            special = (
                (type_id == TYPE_BOS)
                | (type_id == TYPE_EOS)
                | (type_id == TYPE_SEP_CTX)
                | (type_id == TYPE_SEP_QA)
            )
            cen = cen.masked_fill(special.unsqueeze(-1), 0.0)
            out = out + cen
        return out

    def _pointmae_pos(self, t: int) -> torch.Tensor:
        if int(t) > int(self.pos_emb.shape[1]):
            raise ValueError(f"T={t} exceeds max_len={self.pos_emb.shape[1]}")
        return self.pos_emb[:, : int(t), :]

    def _prepare_backbone_inputs(
        self,
        tokens: torch.Tensor,
        type_id: torch.Tensor,
        centers_seq: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Build backbone input / optional per-block positional tensor / NEPA target `z`."""
        if self.pointmae_backbone:
            pos = self._pointmae_pos(tokens.size(1))
            z = tokens + pos
            return tokens, pos, z
        z = self._add_embeddings(tokens, type_id, centers_seq)
        return z, None, z

    def _apply_query_mask(
        self,
        tokens: torch.Tensor,
        type_id: torch.Tensor,
        q_mask_prob: float,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        p = float(q_mask_prob)
        if p <= 0.0 or not self.training:
            return tokens, None
        p = min(max(p, 0.0), 1.0)

        q_mask = (
            (type_id == int(TYPE_Q_POINT))
            | (type_id == int(TYPE_Q_RAY))
            | (type_id == int(TYPE_POINT))
            | (type_id == int(TYPE_RAY))
        )
        n_q = q_mask.sum()
        if int(n_q.item()) <= 0:
            keep_ratio = torch.tensor(1.0, device=tokens.device, dtype=tokens.dtype)
            return tokens, keep_ratio

        drop_mask = (torch.rand_like(tokens[:, :, 0]) < p) & q_mask
        kept = ((~drop_mask) & q_mask).sum().to(dtype=tokens.dtype)
        keep_ratio = kept / n_q.to(dtype=tokens.dtype)
        if not bool(drop_mask.any()):
            return tokens, keep_ratio

        out = tokens.clone()
        if self.q_mask_mode == "zero":
            out = out.masked_fill(drop_mask.unsqueeze(-1), 0.0)
        elif self.q_mask_mode == "mask_token":
            mask_tok = self.q_mask_token.expand(out.shape[0], out.shape[1], out.shape[2])
            out = torch.where(drop_mask.unsqueeze(-1), mask_tok, out)
        else:
            raise RuntimeError(f"invalid q_mask_mode={self.q_mask_mode}")
        return out, keep_ratio

    @staticmethod
    def _gather_by_perm(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
        """Gather x along dim=1 using a per-batch patch permutation perm=(B,P)."""
        if x.dim() == 2:
            return x.gather(1, perm)
        if x.dim() == 3:
            return x.gather(1, perm.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        raise ValueError(f"unsupported tensor rank for patch gather: {tuple(x.shape)}")

    def set_patch_order_mode(self, mode: str) -> None:
        """Update patch-order mode at runtime (for epoch/batch schedule switching)."""
        self.patch_order_mode = str(mode)

    @staticmethod
    def _normalize_patch_mode_token(token: str) -> tuple[str, bool]:
        mode = str(token).strip().lower().replace("-", "_")
        rev = False
        while mode.startswith("rev_"):
            rev = not rev
            mode = mode[len("rev_") :]
        while mode.endswith("_rev"):
            rev = not rev
            mode = mode[: -len("_rev")]
        return mode, rev

    def _compute_patch_order_perm_base(self, centers_xyz: torch.Tensor, mode: str) -> torch.Tensor:
        if centers_xyz.dim() != 3 or centers_xyz.size(-1) != 3:
            raise ValueError(f"centers_xyz must be (B,P,3), got {tuple(centers_xyz.shape)}")

        B, P, _ = centers_xyz.shape
        device = centers_xyz.device
        bits = 10

        if mode in {"none", "original", "as_is", "fps", "identity", "native"}:
            return torch.arange(P, device=device).view(1, P).expand(B, P)
        if mode in {"reverse", "rev"}:
            return torch.arange(P - 1, -1, -1, device=device).view(1, P).expand(B, P)
        if mode in {"random", "shuffle", "rfps"}:
            return torch.argsort(torch.rand(B, P, device=device), dim=1)
        if mode == "random_sweep":
            d = torch.randn((B, 3), device=device, dtype=torch.float32)
            d = F.normalize(d, dim=-1, eps=1e-6).to(dtype=centers_xyz.dtype)
            score = (centers_xyz * d[:, None, :]).sum(dim=-1)
            return torch.argsort(score, dim=1)
        if mode in {"morton", "z", "morton_xyz"}:
            return serialize_indices(centers_xyz, order="morton", bits=bits)
        if mode in {"morton_trans", "z_trans", "morton_xy_swap"}:
            return serialize_indices(centers_xyz, order="morton_trans", bits=bits)

        # morton axis permutation aliases: morton_yzx, morton_zxy, morton_xzy, ...
        if mode.startswith("morton_"):
            perm_str = mode[len("morton_") :]
            if len(perm_str) == 3 and len(set(perm_str)) == 3 and set(perm_str) == {"x", "y", "z"}:
                axis_map = {"x": 0, "y": 1, "z": 2}
                xyz = centers_xyz[..., [axis_map[c] for c in perm_str]]
                codes = morton3d_codes(xyz, bits=bits)
                return torch.argsort(codes, dim=1)

        raise ValueError(
            f"unknown patch_order_mode token={mode} "
            "(supported: none|reverse|random|random_sweep|morton|morton_trans|morton_<perm>)"
        )

    def _compute_patch_order_perm(self, centers_xyz: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Return per-sample patch permutation (B,P), or None to keep original order."""
        if centers_xyz is None:
            return None
        if centers_xyz.dim() != 3 or centers_xyz.size(-1) != 3:
            raise ValueError(f"centers_xyz must be (B,P,3), got {tuple(centers_xyz.shape)}")
        if centers_xyz.size(1) <= 1:
            return None

        raw_mode = str(self.patch_order_mode or "none")
        mode0, rev0 = self._normalize_patch_mode_token(raw_mode)
        if mode0 in {"none", "original", "as_is", "fps", "identity", "native"} and (not rev0):
            return None

        B, P, _ = centers_xyz.shape
        if mode0.startswith("sample:"):
            pool_txt = mode0[len("sample:") :]
            pool_raw = [m.strip() for m in pool_txt.split(",") if m.strip()]
            if not pool_raw:
                raise ValueError(f"sample patch_order_mode requires non-empty pool: {self.patch_order_mode}")

            parsed: list[tuple[str, bool]] = []
            for m in pool_raw:
                m_i, rev_i = self._normalize_patch_mode_token(m)
                parsed.append((m_i, bool(rev0) ^ bool(rev_i)))

            choice = torch.randint(0, len(parsed), (B,), device=centers_xyz.device)
            rows: list[torch.Tensor] = []
            for bi in range(B):
                m_i, rev_i = parsed[int(choice[bi].item())]
                perm_b = self._compute_patch_order_perm_base(centers_xyz[bi : bi + 1], m_i)
                if rev_i:
                    perm_b = torch.flip(perm_b, dims=[1])
                rows.append(perm_b[0])
            return torch.stack(rows, dim=0)

        perm = self._compute_patch_order_perm_base(centers_xyz, mode0)
        if rev0:
            perm = torch.flip(perm, dims=[1])
        return perm

    def _maybe_reorder_patch_embed_output(self, patch_out):
        """Reorder PatchEmbedOutput by patch_order_mode; preserve aligned tensors."""
        perm = self._compute_patch_order_perm(getattr(patch_out, "centers_xyz", None))
        if perm is None:
            return patch_out
        tokens = self._gather_by_perm(patch_out.tokens, perm)
        centers_xyz = self._gather_by_perm(patch_out.centers_xyz, perm)
        group_idx = self._gather_by_perm(patch_out.group_idx, perm)
        return patch_out.__class__(tokens=tokens, centers_xyz=centers_xyz, group_idx=group_idx)

    def _build_seq_qa0(
        self,
        q_tok: torch.Tensor,
        a_tok: torch.Tensor,
        centers_xyz: torch.Tensor,
        q_ray_tok: Optional[torch.Tensor] = None,
        a_ray_tok: Optional[torch.Tensor] = None,
        ray_has: Optional[torch.Tensor] = None,
        ray_centers_xyz: Optional[torch.Tensor] = None,
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
            _, Pr, _ = q_ray_tok.shape
            if a_ray_tok.shape[:2] != (B, Pr):
                raise ValueError("q_ray_tok/a_ray_tok shape mismatch")
            if self.qa_fuse == "add":
                ray_tok = q_ray_tok + a_ray_tok
            elif self.qa_fuse == "concat":
                if self.qa_fuse_proj is None:
                    raise RuntimeError("qa_fuse='concat' requires qa_fuse_proj")
                ray_tok = self.qa_fuse_proj(torch.cat([q_ray_tok, a_ray_tok], dim=-1))
            else:
                raise ValueError(f"unknown qa_fuse={self.qa_fuse}")
            if ray_has is None:
                ray_has = torch.ones((B, Pr), device=q_tok.device, dtype=torch.bool)
            if ray_has.shape != (B, Pr):
                raise ValueError("ray_has shape mismatch with ray tokens")
            if ray_centers_xyz is None:
                ray_centers_xyz = torch.zeros((B, Pr, 3), device=q_tok.device, dtype=centers_xyz.dtype)
            parts.append(ray_tok)
            types.append(
                torch.where(
                    ray_has,
                    torch.full((B, Pr), int(TYPE_RAY), device=q_tok.device, dtype=torch.long),
                    torch.full((B, Pr), int(TYPE_MISSING_RAY), device=q_tok.device, dtype=torch.long),
                )
            )
            centers_parts.append(ray_centers_xyz)

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
        ray_centers_xyz: Optional[torch.Tensor] = None,
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
            _, Pr, _ = q_ray_tok.shape
            if a_ray_tok.shape[:2] != (B, Pr):
                raise ValueError("q_ray_tok/a_ray_tok shape mismatch")
            ray_qa = torch.stack([q_ray_tok, a_ray_tok], dim=2).reshape(B, 2 * Pr, D)
            if ray_has is None:
                ray_has = torch.ones((B, Pr), device=q_tok.device, dtype=torch.bool)
            if ray_has.shape != (B, Pr):
                raise ValueError("ray_has shape mismatch with ray tokens")
            if ray_centers_xyz is None:
                ray_centers_xyz = torch.zeros((B, Pr, 3), device=q_tok.device, dtype=centers_xyz.dtype)
            q_type = torch.where(
                ray_has,
                torch.full((B, Pr), int(TYPE_Q_RAY), device=q_tok.device, dtype=torch.long),
                torch.full((B, Pr), int(TYPE_MISSING_RAY), device=q_tok.device, dtype=torch.long),
            )
            a_type = torch.where(
                ray_has,
                torch.full((B, Pr), int(TYPE_A_RAY), device=q_tok.device, dtype=torch.long),
                torch.full((B, Pr), int(TYPE_MISSING_RAY), device=q_tok.device, dtype=torch.long),
            )
            parts.append(ray_qa)
            types.append(torch.stack([q_type, a_type], dim=2).reshape(B, 2 * Pr))
            centers_parts.append(ray_centers_xyz.repeat_interleave(2, dim=1))

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
        ray_centers_xyz: Optional[torch.Tensor] = None,
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
            _, Pr, _ = q_ray_tok.shape
            if a_ray_tok is None or a_ray_tok.shape[:2] != (B, Pr):
                raise ValueError("q_ray_tok/a_ray_tok shape mismatch")
            if ray_has is None:
                ray_has = torch.ones((B, Pr), device=q_tok.device, dtype=torch.bool)
            if ray_has.shape != (B, Pr):
                raise ValueError("ray_has shape mismatch with ray tokens")
            if ray_centers_xyz is None:
                ray_centers_xyz = torch.zeros((B, Pr, 3), device=q_tok.device, dtype=centers_xyz.dtype)
            q_types = torch.where(
                ray_has,
                torch.full((B, Pr), int(TYPE_Q_RAY), device=q_tok.device, dtype=torch.long),
                torch.full((B, Pr), int(TYPE_MISSING_RAY), device=q_tok.device, dtype=torch.long),
            )
            parts.append(q_ray_tok)
            types.append(q_types)
            centers_parts.append(ray_centers_xyz)

        if self.qa_sep_token:
            parts.append(self.sep_token.expand(B, 1, D))
            types.append(torch.full((B, 1), int(TYPE_SEP_QA), device=q_tok.device, dtype=torch.long))
            centers_parts.append(torch.zeros((B, 1, 3), device=q_tok.device, dtype=centers_xyz.dtype))

        parts.append(a_tok)
        types.append(torch.full((B, P), int(TYPE_A_POINT), device=q_tok.device, dtype=torch.long))
        centers_parts.append(centers_xyz)

        if a_ray_tok is not None:
            _, Pr, _ = a_ray_tok.shape
            if ray_has is None:
                ray_has = torch.ones((B, Pr), device=q_tok.device, dtype=torch.bool)
            if ray_has.shape != (B, Pr):
                raise ValueError("ray_has shape mismatch with ray tokens")
            if ray_centers_xyz is None:
                ray_centers_xyz = torch.zeros((B, Pr, 3), device=q_tok.device, dtype=centers_xyz.dtype)
            a_types = torch.where(
                ray_has,
                torch.full((B, Pr), int(TYPE_A_RAY), device=q_tok.device, dtype=torch.long),
                torch.full((B, Pr), int(TYPE_MISSING_RAY), device=q_tok.device, dtype=torch.long),
            )
            parts.append(a_ray_tok)
            types.append(a_types)
            centers_parts.append(ray_centers_xyz)

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
        q_mask_prob: float = 0.0,
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
        patch_out = self._maybe_reorder_patch_embed_output(patch_out)
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
            if len(feats) == 0:
                ans_feat = torch.zeros(
                    (pt_xyz.shape[0], pt_xyz.shape[1], int(self.answer_in_dim)),
                    device=pt_xyz.device,
                    dtype=pt_xyz.dtype,
                )
            else:
                ans_feat = torch.cat(feats, dim=-1) if len(feats) > 1 else feats[0]
            a_tok = self.answer_embed(ans_feat, group_idx)

        q_ray_tok: Optional[torch.Tensor] = None
        a_ray_tok: Optional[torch.Tensor] = None
        ray_has: Optional[torch.Tensor] = None
        ray_centers_xyz: Optional[torch.Tensor] = None
        if self.use_ray_patch:
            if self.ray_patch_embed is None:
                raise RuntimeError("use_ray_patch=True but ray_patch_embed is None")
            if ray_o is None or ray_d is None:
                # Mixed datasets may include samples with no ray modality (e.g., UDF).
                # Keep training stable by providing dummy rays and disabling them via ray_available.
                B = int(pt_xyz.shape[0])
                dev = pt_xyz.device
                dt = pt_xyz.dtype
                cand_r = []
                for t in (ray_t, ray_hit, ray_n, ray_unc):
                    if t is None or t.dim() < 2:
                        continue
                    cand_r.append(int(t.shape[1]))
                R = max(cand_r) if cand_r else 1
                if ray_o is None:
                    ray_o = torch.zeros((B, R, 3), device=dev, dtype=dt)
                if ray_d is None:
                    ray_d = torch.zeros((B, R, 3), device=dev, dtype=dt)
                if ray_available is None:
                    ray_available = torch.zeros((B,), device=dev, dtype=torch.long)
                else:
                    ray_available = torch.zeros_like(ray_available.to(device=dev)).view(B)
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
            ray_centers_xyz = ray_out.centers_xyz

        if self.qa_tokens == 0:
            tokens, type_id, centers_seq = self._build_seq_qa0(
                q_tok,
                a_tok,
                centers_xyz,
                q_ray_tok=q_ray_tok,
                a_ray_tok=a_ray_tok,
                ray_has=ray_has,
                ray_centers_xyz=ray_centers_xyz,
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
                    ray_centers_xyz=ray_centers_xyz,
                )
            else:
                tokens, type_id, centers_seq = self._build_seq_qa1_split(
                    q_tok,
                    a_tok,
                    centers_xyz,
                    q_ray_tok=q_ray_tok,
                    a_ray_tok=a_ray_tok,
                    ray_has=ray_has,
                    ray_centers_xyz=ray_centers_xyz,
                )

        # Keep sep_token connected in all layouts (including interleave without SEP)
        # to avoid DDP unused-parameter errors.
        tokens = tokens + (self.sep_ctx_token.sum() * 0.0)
        tokens = tokens + (self.sep_token.sum() * 0.0)
        # Keep q_mask_token connected even when q_mask_prob=0 to avoid DDP unused-parameter errors.
        tokens = tokens + (self.q_mask_token.sum() * 0.0)
        tokens, q_keep_ratio = self._apply_query_mask(tokens, type_id, float(q_mask_prob))
        backbone_in, backbone_pos, z = self._prepare_backbone_inputs(tokens, type_id, centers_seq)

        if isinstance(self.backbone, EncoderDecoderTransformer):
            if self.qa_tokens != 1 or self.qa_layout != "split":
                raise ValueError("encdec_arch expects qa_tokens=1 and qa_layout='split'")
            if self.qa_sep_token:
                sep_pos = (type_id == int(TYPE_SEP_QA)).int().argmax(dim=1)
                sep = int(sep_pos[0].item())
            else:
                sep = 1 + q_tok.shape[1]
            enc = backbone_in[:, :sep, :]
            dec = backbone_in[:, sep:, :]
            enc_out, dec_out = self.backbone(enc, dec, enc_xyz=None)
            h = torch.cat([enc_out, dec_out], dim=1)
        else:
            h = self.backbone(
                backbone_in,
                is_causal=bool(is_causal),
                type_id=type_id,
                pos=backbone_pos,
                dual_mask_near=float(dual_mask_near),
                dual_mask_far=float(dual_mask_far),
                dual_mask_window=int(dual_mask_window),
                dual_mask_type_aware=int(dual_mask_type_aware),
            )

        z_hat = self.pred_head(h)
        return PatchNepaOutput(
            tokens=tokens,
            z=z,
            h=h,
            z_hat=z_hat,
            type_id=type_id,
            centers_xyz=centers_xyz,
            group_idx=group_idx,
            q_keep_ratio=q_keep_ratio,
        )

    # ---------------------------------------------------------------------
    # Token-level API (v2 / CPAC support)
    # ---------------------------------------------------------------------
    def encode_patches(
        self,
        pt_xyz: torch.Tensor,
        pt_n: Optional[torch.Tensor] = None,
        *,
        patch_order_mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode surface points into patch tokens.

        Returns:
            tokens: [B, P, D]
            centers_xyz: [B, P, 3]
            group_idx: [B, P, K]
        """
        if (pt_n is not None) and (not self.use_normals):
            pt_n = None
        patch_out = self.patch_embed(pt_xyz, pt_n)
        if patch_order_mode is None:
            patch_out = self._maybe_reorder_patch_embed_output(patch_out)
        else:
            prev = str(self.patch_order_mode)
            try:
                if str(patch_order_mode) != prev:
                    self.set_patch_order_mode(str(patch_order_mode))
                patch_out = self._maybe_reorder_patch_embed_output(patch_out)
            finally:
                if str(self.patch_order_mode) != prev:
                    self.set_patch_order_mode(prev)
        return patch_out.tokens, patch_out.centers_xyz, patch_out.group_idx

    @staticmethod
    def _identity_group_idx(batch_size: int, n: int, device: torch.device) -> torch.Tensor:
        """Build [B,N,1] identity groups (one token per point)."""
        base = torch.arange(int(n), device=device, dtype=torch.long).view(1, int(n), 1)
        return base.repeat(int(batch_size), 1, 1)

    def encode_point_queries(
        self,
        qry_xyz: torch.Tensor,
        *,
        token_value: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode arbitrary 3D coordinates as query tokens."""
        if qry_xyz.dim() != 3 or qry_xyz.size(-1) != 3:
            raise ValueError(f"qry_xyz must be (B,N,3), got {tuple(qry_xyz.shape)}")
        b, n, _ = qry_xyz.shape
        tokens = torch.full(
            (b, n, int(self.d_model)),
            float(token_value),
            device=qry_xyz.device,
            dtype=qry_xyz.dtype,
        )
        return tokens, qry_xyz

    def encode_point_answers(
        self,
        ans_feat: torch.Tensor,
        ans_xyz: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode per-point answer features (1 point -> 1 token)."""
        if self.answer_embed is None:
            raise RuntimeError(
                "encode_point_answers() requires answer_embed. "
                "Enable use_pt_dist and/or use_pt_grad."
            )
        if ans_xyz.dim() != 3 or ans_xyz.size(-1) != 3:
            raise ValueError(f"ans_xyz must be (B,N,3), got {tuple(ans_xyz.shape)}")
        if ans_feat.dim() != 3 or ans_feat.size(0) != ans_xyz.size(0) or ans_feat.size(1) != ans_xyz.size(1):
            raise ValueError(
                f"ans_feat must be (B,N,F) aligned with ans_xyz; got feat={tuple(ans_feat.shape)} xyz={tuple(ans_xyz.shape)}"
            )
        if ans_feat.size(-1) != int(self.answer_embed.in_dim):
            raise ValueError(
                f"ans_feat last-dim must match answer_embed.in_dim={int(self.answer_embed.in_dim)}, got {ans_feat.size(-1)}"
            )
        b, n, _ = ans_xyz.shape
        group_idx = self._identity_group_idx(b, n, ans_xyz.device)
        tokens = self.answer_embed(ans_feat, group_idx)
        return tokens, ans_xyz

    def encode_rays(
        self,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        ray_t: torch.Tensor,
        ray_hit: torch.Tensor,
        ray_n: Optional[torch.Tensor] = None,
        ray_unc: Optional[torch.Tensor] = None,
        ray_available: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode rays as 1-ray/1-token query+answer streams.

        Returns:
            q_tok: [B, R, D]
            a_tok: [B, R, D]
            centers_xyz: [B, R, 3] (x_anchor)
        """
        if self.ray_patch_embed is None:
            raise RuntimeError("encode_rays() requires use_ray_patch=True")

        if ray_o.dim() != 3 or ray_o.size(-1) != 3:
            raise ValueError(f"ray_o must be (B,R,3), got {tuple(ray_o.shape)}")
        if ray_d.dim() != 3 or ray_d.size(-1) != 3:
            raise ValueError(f"ray_d must be (B,R,3), got {tuple(ray_d.shape)}")
        if ray_t.dim() == 2:
            ray_t = ray_t.unsqueeze(-1)
        if ray_hit.dim() == 2:
            ray_hit = ray_hit.unsqueeze(-1)
        if ray_t.dim() != 3 or ray_t.size(-1) != 1:
            raise ValueError(f"ray_t must be (B,R,1), got {tuple(ray_t.shape)}")
        if ray_hit.dim() != 3 or ray_hit.size(-1) != 1:
            raise ValueError(f"ray_hit must be (B,R,1), got {tuple(ray_hit.shape)}")
        if ray_o.shape[:2] != ray_d.shape[:2] or ray_o.shape[:2] != ray_t.shape[:2] or ray_o.shape[:2] != ray_hit.shape[:2]:
            raise ValueError(
                "ray tensors first two dims must match: "
                f"o={tuple(ray_o.shape)} d={tuple(ray_d.shape)} t={tuple(ray_t.shape)} hit={tuple(ray_hit.shape)}"
            )
        b, r, _ = ray_o.shape
        x_anchor = ray_o + ray_t * ray_d
        rel_anchor = torch.zeros_like(ray_d)

        rel_o = None
        if self.ray_patch_embed.use_ray_origin:
            rel_o = ray_o - x_anchor

        q_parts = [rel_anchor, ray_d]
        if rel_o is not None:
            q_parts.append(rel_o)
        q_tok = self.ray_patch_embed.mlp_q(torch.cat(q_parts, dim=-1))

        a_parts = [rel_anchor, ray_d, ray_hit, ray_t]
        if self.ray_patch_embed.include_ray_normal:
            if ray_n is None:
                ray_n = torch.zeros_like(ray_d)
            elif ray_n.dim() != 3 or ray_n.shape != ray_d.shape:
                raise ValueError(f"ray_n must be (B,R,3), got {tuple(ray_n.shape)}")
            a_parts.append(ray_n)
        if self.ray_patch_embed.include_ray_unc:
            if ray_unc is None:
                ray_unc = torch.zeros((b, r, 1), device=ray_o.device, dtype=ray_o.dtype)
            elif ray_unc.dim() == 2:
                ray_unc = ray_unc.unsqueeze(-1)
            if ray_unc.dim() != 3 or ray_unc.size(-1) != 1 or ray_unc.shape[:2] != (b, r):
                raise ValueError(f"ray_unc must be (B,R,1), got {tuple(ray_unc.shape)}")
            a_parts.append(ray_unc)
        if rel_o is not None:
            a_parts.append(rel_o)
        a_tok = self.ray_patch_embed.mlp_a(torch.cat(a_parts, dim=-1))

        if ray_available is not None:
            if ray_available.dim() == 2:
                ray_available = ray_available.unsqueeze(-1)
            if ray_available.dim() != 3 or ray_available.shape[:2] != (b, r) or ray_available.size(-1) != 1:
                raise ValueError(f"ray_available must be (B,R,1) or (B,R), got {tuple(ray_available.shape)}")
            mask = ray_available.to(device=ray_o.device, dtype=q_tok.dtype)
            q_tok = q_tok * mask
            a_tok = a_tok * mask

        return q_tok, a_tok, x_anchor

    def forward_tokens(
        self,
        tokens: torch.Tensor,
        type_id: torch.Tensor,
        centers_xyz: Optional[torch.Tensor] = None,
        *,
        is_causal: bool = True,
        q_mask_prob: float = 0.0,
        dual_mask_near: float = 0.0,
        dual_mask_far: float = 0.0,
        dual_mask_window: int = 0,
        dual_mask_seed: Optional[int] = None,
        dual_mask_type_aware: bool = False,
    ) -> PatchNepaOutput:
        """Forward pass from pre-tokenized streams.

        Args:
            tokens: [B,L,D] content tokens before type/pos/center embeddings.
            type_id: [B,L] TYPE_* ids.
            centers_xyz: [B,L,3] per-token centers for center_mlp.
        """
        if tokens.dim() != 3:
            raise ValueError(f"forward_tokens(): expected tokens [B,L,D], got {tuple(tokens.shape)}")
        if type_id.dim() != 2:
            raise ValueError(f"forward_tokens(): expected type_id [B,L], got {tuple(type_id.shape)}")

        b, l, d = tokens.shape
        if d != int(self.d_model):
            raise ValueError(f"forward_tokens(): tokens dim mismatch model d_model={int(self.d_model)} got {d}")
        if type_id.shape != (b, l):
            raise ValueError(f"forward_tokens(): type_id shape must be {(b, l)}, got {tuple(type_id.shape)}")

        if centers_xyz is None:
            centers_xyz = torch.zeros((b, l, 3), device=tokens.device, dtype=tokens.dtype)
        elif centers_xyz.shape != (b, l, 3):
            raise ValueError(f"forward_tokens(): centers_xyz must be {(b, l, 3)}, got {tuple(centers_xyz.shape)}")

        # Keep special parameters connected to avoid DDP unused-parameter issues.
        tokens = tokens + (self.sep_ctx_token.sum() * 0.0)
        tokens = tokens + (self.sep_token.sum() * 0.0)
        tokens = tokens + (self.q_mask_token.sum() * 0.0)
        tokens, q_keep_ratio = self._apply_query_mask(tokens, type_id, float(q_mask_prob))
        backbone_in, backbone_pos, z = self._prepare_backbone_inputs(tokens, type_id, centers_xyz)

        if isinstance(self.backbone, EncoderDecoderTransformer):
            sep_mask = type_id == int(TYPE_SEP_QA)
            if bool(sep_mask.any()):
                sep_pos = sep_mask.int().argmax(dim=1)
                if bool((sep_pos != sep_pos[0]).any()):
                    raise ValueError("forward_tokens(): encdec mode requires a consistent SEP index across batch")
                sep = int(sep_pos[0].item())
                sep = min(max(sep, 1), l - 1)
                enc = backbone_in[:, :sep, :]
                dec = backbone_in[:, sep:, :]
            else:
                enc = backbone_in[:, :-1, :]
                dec = backbone_in[:, -1:, :]
            enc_out, dec_out = self.backbone(enc, dec, enc_xyz=None)
            h = torch.cat([enc_out, dec_out], dim=1)
        else:
            h = self.backbone(
                backbone_in,
                is_causal=bool(is_causal),
                type_id=type_id,
                pos=backbone_pos,
                dual_mask_near=float(dual_mask_near),
                dual_mask_far=float(dual_mask_far),
                dual_mask_window=int(dual_mask_window),
                dual_mask_seed=(int(dual_mask_seed) if dual_mask_seed is not None else None),
                dual_mask_type_aware=int(bool(dual_mask_type_aware)),
            )

        z_hat = self.pred_head(h)
        return PatchNepaOutput(
            tokens=tokens,
            z=z,
            h=h,
            z_hat=z_hat,
            type_id=type_id,
            centers_xyz=centers_xyz,
            group_idx=None,
            q_keep_ratio=q_keep_ratio,
        )

    @staticmethod
    def nepa_loss(
        z: torch.Tensor,
        z_hat: torch.Tensor,
        type_id: Optional[torch.Tensor] = None,
        *,
        skip_k: int = 1,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        k = int(skip_k)
        if k < 1:
            raise ValueError(f"skip_k must be >=1, got {skip_k}")
        target_seq = z if target is None else target
        if target_seq.size(1) <= k:
            return z_hat.sum() * 0.0
        if target_seq.shape != z.shape:
            raise ValueError(
                f"target shape mismatch: z={tuple(z.shape)} target={tuple(target_seq.shape)}"
            )

        pred = z_hat[:, :-k, :]
        tgt = target_seq[:, k:, :].detach()

        if type_id is None:
            mask = torch.ones(pred.shape[:2], device=z.device, dtype=torch.bool)
        else:
            tgt_ty = type_id[:, k:]
            has_answer = bool((tgt_ty == int(TYPE_A_POINT)).any() or (tgt_ty == int(TYPE_A_RAY)).any())
            if has_answer:
                mask = (tgt_ty == int(TYPE_A_POINT)) | (tgt_ty == int(TYPE_A_RAY))
            else:
                mask = (
                    (tgt_ty != int(TYPE_BOS))
                    & (tgt_ty != int(TYPE_SEP_CTX))
                    & (tgt_ty != int(TYPE_SEP_QA))
                    & (tgt_ty != int(TYPE_EOS))
                    & (tgt_ty != int(TYPE_MISSING_RAY))
                )
        if not bool(mask.any()):
            return pred.sum() * 0.0
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
        pool_mode: Optional[str] = None,  # alias of pooling
        cls_token_source: str = "last_q",  # bos | last_q | eos
        head_mode: str = "pointmae_mlp",  # linear | pointmae_mlp
        head_hidden_dim: int = 256,
        head_dropout: float = 0.5,
        is_causal: bool = False,
        ft_sequence_mode: str = "qa_zeroa",  # qa_zeroa | q_only
        **nepa_kwargs,
    ) -> None:
        super().__init__()
        if pool_mode is not None:
            pooling = str(pool_mode)
        assert pooling in {"mean", "mean_q", "cls", "cls_max", "sep"}
        assert cls_token_source in {"bos", "last_q", "eos"}
        assert head_mode in {"linear", "pointmae_mlp"}
        assert ft_sequence_mode in {"qa_zeroa", "q_only"}

        self.core = PatchTransformerNepa(**nepa_kwargs)
        self.d_model = int(self.core.d_model)
        self.pooling = str(pooling)
        self.cls_token_source = str(cls_token_source)
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

    def _select_cls_feat(self, h: torch.Tensor, type_id: torch.Tensor) -> torch.Tensor:
        b, t, _ = h.shape
        dev = h.device
        src = self.cls_token_source
        if src == "bos":
            return h[:, 0, :]

        if src == "eos":
            eos_mask = type_id == int(TYPE_EOS)
            has_eos = eos_mask.any(dim=1)
            eos_idx = eos_mask.to(dtype=torch.float32).argmax(dim=1)
            eos_idx = torch.where(has_eos, eos_idx, torch.full_like(eos_idx, t - 1))
            return h[torch.arange(b, device=dev), eos_idx]

        # last_q: last valid query-like token in sequence.
        q_mask = self._query_token_mask(type_id)
        has_q = q_mask.any(dim=1)
        rev_idx = q_mask.flip(dims=[1]).to(dtype=torch.float32).argmax(dim=1)
        last_q_idx = (t - 1) - rev_idx
        last_q_idx = torch.where(has_q, last_q_idx, torch.zeros_like(last_q_idx))
        return h[torch.arange(b, device=dev), last_q_idx]

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
            & (type_id != int(TYPE_SEP_CTX))
            & (type_id != int(TYPE_SEP_QA))
            & (type_id != int(TYPE_A_POINT))
            & (type_id != int(TYPE_A_RAY))
            & (type_id != int(TYPE_MISSING_RAY))
        )
        return torch.where(has_any, mask, fallback)

    @staticmethod
    def _select_sep_feat(h: torch.Tensor, type_id: torch.Tensor) -> torch.Tensor:
        b, _, _ = h.shape
        dev = h.device
        sep_mask = (type_id == int(TYPE_SEP_QA)) | (type_id == int(TYPE_SEP_CTX))
        if not bool(torch.all(sep_mask.any(dim=1))):
            raise ValueError(
                "pooling='sep' requires SEP token in sequence (set qa_layout='split_sep' and ft_sequence_mode='qa_zeroa')."
            )
        sep_idx = sep_mask.to(dtype=torch.float32).argmax(dim=1)
        return h[torch.arange(b, device=dev), sep_idx]

    def _pool_features(self, h: torch.Tensor, type_id: torch.Tensor) -> torch.Tensor:
        if self.pooling == "sep":
            return self._select_sep_feat(h, type_id)
        cls_feat = self._select_cls_feat(h, type_id)
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
        patch_out = self.core._maybe_reorder_patch_embed_output(patch_out)
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
            Pr = int(ray_out.q_tok.shape[1])
            q_ray_type = torch.where(
                ray_out.has_ray,
                torch.full((B, Pr), int(TYPE_Q_RAY), device=dev, dtype=torch.long),
                torch.full((B, Pr), int(TYPE_MISSING_RAY), device=dev, dtype=torch.long),
            )
            parts.append(ray_out.q_tok)
            types.append(q_ray_type)
            centers_parts.append(ray_out.centers_xyz)

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
            backbone_in, backbone_pos, z = self.core._prepare_backbone_inputs(tokens, type_id, centers_seq)
            if isinstance(self.core.backbone, EncoderDecoderTransformer):
                # Encoder-only inference in q_only mode; keep EOS as a 1-token decoder stub.
                enc = backbone_in[:, :-1, :]
                dec = backbone_in[:, -1:, :]
                enc_out, _ = self.core.backbone(enc, dec, enc_xyz=None)
                h = enc_out
                type_for_pool = type_id[:, :-1]
            else:
                h = self.core.backbone(
                    backbone_in,
                    is_causal=self.is_causal,
                    type_id=type_id,
                    pos=backbone_pos,
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
