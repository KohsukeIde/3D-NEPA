from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .causal_transformer import CausalTransformer
from .encdec_transformer import EncoderDecoderTransformer
from ..token.tokenizer import (
    TYPE_MISSING_RAY,
    TYPE_Q_POINT,
    TYPE_A_POINT,
    TYPE_Q_RAY,
    TYPE_A_RAY,
    TYPE_VOCAB_SIZE,
)


class QueryNepa(nn.Module):
    def __init__(
        self,
        feat_dim=15,
        d_model=384,
        n_types=9,
        nhead=6,
        num_layers=8,
        mlp_ratio=4,
        dropout=0.0,
        drop_path=0.0,
        backbone_impl: str = "nepa2d",
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_affine: bool = False,
        qk_norm_bias: bool = False,
        layerscale_value: float = 1e-5,
        rope_theta: float = 100.0,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        use_gated_mlp: bool = False,
        hidden_act: str = "gelu",
        final_layernorm: bool = True,
        max_len=2048,
        type_specific_pos: bool = False,
        arch="causal",
        topo_k=0,
        topo_include_bos=True,
        topo_ray_coord="origin",  # origin | proj | bbox
        topo_ray_bbox=0.5,
        encdec_src_causal=0,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.d_model = int(d_model)
        self.arch = str(arch)
        self.backbone_impl = str(backbone_impl).lower()
        self.topo_k = int(topo_k)
        self.topo_include_bos = bool(topo_include_bos)
        self.topo_ray_coord = str(topo_ray_coord)
        self.topo_ray_bbox = float(topo_ray_bbox)
        self.encdec_src_causal = bool(encdec_src_causal)
        self.type_specific_pos = bool(type_specific_pos)

        self.type_emb = nn.Embedding(int(n_types), int(d_model))
        self.token_mlp = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Type-specific positional embedding: each token type gets its own local
        # position index (0..count(type)-1). This avoids the single shared
        # positional index having to represent heterogeneous token semantics
        # (point vs ray, query vs answer, etc.).
        if self.type_specific_pos:
            self.type_pos_max_len = int(max_len)
            self.type_pos_emb = nn.Embedding(int(n_types) * self.type_pos_max_len, int(d_model))
            nn.init.trunc_normal_(self.type_pos_emb.weight, std=0.02)

        if self.arch == "causal":
            self.backbone = CausalTransformer(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=drop_path,
                backbone_impl=str(backbone_impl),
                qkv_bias=bool(qkv_bias),
                qk_norm=bool(qk_norm),
                qk_norm_affine=bool(qk_norm_affine),
                qk_norm_bias=bool(qk_norm_bias),
                layerscale_value=float(layerscale_value),
                rope_theta=float(rope_theta),
                layer_norm_eps=float(layer_norm_eps),
                hidden_dropout_prob=float(hidden_dropout_prob),
                attention_probs_dropout_prob=float(attention_probs_dropout_prob),
                use_gated_mlp=bool(use_gated_mlp),
                hidden_act=str(hidden_act),
                final_layernorm=bool(final_layernorm),
            )
        elif self.arch == "encdec":
            self.backbone = EncoderDecoderTransformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_layers,
                num_decoder_layers=num_layers,
                dim_feedforward=int(d_model * mlp_ratio),
                dropout=dropout,
                drop_path=drop_path,
                topo_k=self.topo_k,
                topo_include_bos=self.topo_include_bos,
                src_causal=bool(self.encdec_src_causal),
            )
        else:
            raise ValueError(f"Unknown arch={self.arch}")

        self.pred_head = nn.Linear(d_model, d_model)
        self._init_weights_nepa2d_style()

    def _init_weights_nepa2d_style(self) -> None:
        init_std = 0.02

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=init_std)

        # Keep explicit control for learned absolute/type-specific positional tables.
        nn.init.trunc_normal_(self.pos_emb, std=init_std)
        if self.type_specific_pos:
            nn.init.trunc_normal_(self.type_pos_emb.weight, std=init_std)

    def embed_tokens(self, feat, type_id):
        b, t, _ = feat.shape
        x = self.token_mlp(feat) + self.type_emb(type_id)
        # NEPA2D-style backbone uses RoPE (no additive absolute position embedding).
        # Keep pos_emb tensor for checkpoint compatibility / max_len handling.
        if self.arch == "causal" and self.backbone_impl == "nepa2d":
            x = x + (self.pos_emb[:, :t, :] * 0.0)
        else:
            x = x + self.pos_emb[:, :t, :]

        if self.type_specific_pos:
            # Local position within each token type, computed by per-type cumulative sums.
            n_types = int(self.type_emb.num_embeddings)
            local = torch.zeros((b, t), dtype=torch.long, device=type_id.device)
            # (small) loop over types is fine; n_types is ~5 or ~10.
            for tt in range(n_types):
                m = type_id == tt
                if bool(m.any()):
                    c = torch.cumsum(m.long(), dim=1) - 1
                    local[m] = c[m]

            # Safety clamp in case a single type exceeds configured max_len.
            local = torch.clamp(local, 0, self.type_pos_max_len - 1)
            type_id_safe = torch.clamp(type_id, 0, n_types - 1)
            idx = type_id_safe * self.type_pos_max_len + local
            x = x + self.type_pos_emb(idx)

        return x

    def forward(
        self,
        feat,
        type_id,
        is_causal: bool = True,
        dual_mask_near: float = 0.0,
        dual_mask_far: float = 0.0,
        dual_mask_window: int = 0,
        dual_mask_seed: int | None = None,
        dual_mask_type_aware: int | bool = 0,
    ):
        """Forward.

        dual_mask_* are only used during training (see CausalTransformer).
        """
        z = self.embed_tokens(feat, type_id)

        if self.arch == "causal":
            h = self.backbone(
                z,
                is_causal=bool(is_causal),
                type_id=type_id,
                dual_mask_near=float(dual_mask_near),
                dual_mask_far=float(dual_mask_far),
                dual_mask_window=int(dual_mask_window),
                dual_mask_seed=None if dual_mask_seed is None else int(dual_mask_seed),
                dual_mask_type_aware=int(dual_mask_type_aware),
            )
        else:
            # encdec expects split layout:
            # [BOS][Q...][A...][EOS]
            b, t, _ = z.shape
            is_a = (type_id == TYPE_A_POINT) | (type_id == TYPE_A_RAY)
            if not bool((is_a.sum(dim=1) > 0).all()):
                raise ValueError("encdec arch requires answer tokens (qa_tokens=1)")

            a0 = torch.argmax(is_a.int(), dim=1)
            if not bool(torch.all(a0 == a0[0])):
                raise ValueError("encdec arch expects consistent qa_layout across batch")
            a0 = int(a0[0].item())

            eos = t - 1
            is_answer_like = is_a | (type_id == TYPE_MISSING_RAY)
            if not bool(is_answer_like[:, a0:eos].all()):
                raise ValueError(
                    "encdec arch requires answers to be contiguous (qa_layout='split' or 'split_sep')"
                )

            enc_in = z[:, :a0, :]
            dec_in = z[:, a0:eos, :]

            # Build encoder xyz for optional topology mask.
            enc_feat = feat[:, :a0, :]
            enc_type = type_id[:, :a0]
            enc_xyz = torch.zeros((b, a0, 3), device=feat.device, dtype=feat.dtype)

            m_qp = enc_type == TYPE_Q_POINT
            m_qr = enc_type == TYPE_Q_RAY
            if bool(m_qp.any()):
                enc_xyz[m_qp] = enc_feat[m_qp][:, 0:3]

            if bool(m_qr.any()):
                mode = self.topo_ray_coord
                if mode == "origin":
                    enc_xyz[m_qr] = enc_feat[m_qr][:, 3:6]
                elif mode == "proj":
                    # Project each query-ray to the nearest query-point.
                    for bi in range(b):
                        idx_r = (enc_type[bi] == TYPE_Q_RAY).nonzero(as_tuple=False).squeeze(1)
                        if idx_r.numel() == 0:
                            continue
                        idx_p = (enc_type[bi] == TYPE_Q_POINT).nonzero(as_tuple=False).squeeze(1)
                        if idx_p.numel() == 0:
                            enc_xyz[bi, idx_r] = enc_feat[bi, idx_r, 3:6]
                            continue

                        p = enc_feat[bi, idx_p, 0:3]  # (Np, 3)
                        o = enc_feat[bi, idx_r, 3:6]  # (Nr, 3)
                        d = enc_feat[bi, idx_r, 6:9]  # (Nr, 3)
                        denom = (d * d).sum(dim=-1, keepdim=True).clamp_min(1e-6)

                        v = p.unsqueeze(0) - o.unsqueeze(1)  # (Nr, Np, 3)
                        tproj = (v * d.unsqueeze(1)).sum(dim=-1) / denom  # (Nr, Np)
                        tproj = tproj.clamp_min(0.0)
                        proj = o.unsqueeze(1) + tproj.unsqueeze(-1) * d.unsqueeze(1)  # (Nr, Np, 3)
                        dist2 = ((p.unsqueeze(0) - proj) ** 2).sum(dim=-1)  # (Nr, Np)
                        min_idx = dist2.argmin(dim=1)
                        enc_xyz[bi, idx_r] = proj[torch.arange(o.shape[0], device=feat.device), min_idx]
                elif mode == "bbox":
                    bbox = float(self.topo_ray_bbox)
                    for bi in range(b):
                        idx_r = (enc_type[bi] == TYPE_Q_RAY).nonzero(as_tuple=False).squeeze(1)
                        if idx_r.numel() == 0:
                            continue
                        o = enc_feat[bi, idx_r, 3:6]
                        d = enc_feat[bi, idx_r, 6:9]
                        d_safe = torch.where(d.abs() < 1e-6, torch.full_like(d, 1e-6), d)
                        inv_d = 1.0 / d_safe
                        t1 = (-bbox - o) * inv_d
                        t2 = (bbox - o) * inv_d
                        tmin = torch.minimum(t1, t2).amax(dim=-1)
                        tmax = torch.maximum(t1, t2).amin(dim=-1)
                        zero = torch.zeros_like(tmin)
                        valid = tmax >= torch.maximum(tmin, zero)
                        t_enter = torch.where(valid, torch.maximum(tmin, zero), zero)
                        xyz = o + t_enter.unsqueeze(-1) * d
                        xyz = torch.where(valid.unsqueeze(-1), xyz, o)
                        enc_xyz[bi, idx_r] = xyz
                else:
                    raise ValueError(f"Unknown topo_ray_coord: {mode}")

            enc_out, dec_out = self.backbone(enc_in, dec_in, enc_xyz=enc_xyz)
            eos_h = torch.zeros((b, 1, self.d_model), device=feat.device, dtype=feat.dtype)
            h = torch.cat([enc_out, dec_out, eos_h], dim=1)

        z_hat = self.pred_head(h)
        return z, z_hat, h

    def nepa_loss(self, z, z_hat, type_id=None):
        """NEPA next-embedding prediction loss.

        - Legacy sequences: valid targets are everything except TYPE_MISSING_RAY.
        - Q/A sequences: **answer-only** by default (targets are TYPE_A_POINT / TYPE_A_RAY),
          and TYPE_MISSING_RAY is always excluded.
        """
        pred = z_hat[:, :-1, :]
        target = z[:, 1:, :].detach()
        loss = 1.0 - F.cosine_similarity(pred, target, dim=-1, eps=1e-8)
        if type_id is None:
            return loss.mean()

        target_type = type_id[:, 1:]
        valid = target_type != TYPE_MISSING_RAY

        # Auto-detect Q/A tokenization: if answer types exist, compute loss only on them.
        has_answer_types = (target_type == TYPE_A_POINT).any() or (target_type == TYPE_A_RAY).any()
        if bool(has_answer_types):
            valid = valid & ((target_type == TYPE_A_POINT) | (target_type == TYPE_A_RAY))

        if valid.any():
            return loss[valid].mean()
        return loss.new_tensor(0.0)

    def mae_loss(self, z_hat, z_target, mask):
        if mask is None:
            return F.mse_loss(z_hat, z_target)
        if mask.any():
            pred = z_hat[mask]
            tgt = z_target[mask].detach()
            return F.mse_loss(pred, tgt)
        return z_hat.new_tensor(0.0)
