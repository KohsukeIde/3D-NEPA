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
        max_len=2048,
        arch="causal",
        topo_k=0,
        topo_include_bos=True,
        topo_ray_coord="origin",  # origin | proj | bbox
        topo_ray_bbox=0.5,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.d_model = int(d_model)
        self.arch = str(arch)
        self.topo_k = int(topo_k)
        self.topo_include_bos = bool(topo_include_bos)
        self.topo_ray_coord = str(topo_ray_coord)
        self.topo_ray_bbox = float(topo_ray_bbox)

        self.type_emb = nn.Embedding(int(n_types), int(d_model))
        self.token_mlp = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        if self.arch == "causal":
            self.backbone = CausalTransformer(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
        elif self.arch == "encdec":
            self.backbone = EncoderDecoderTransformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_layers,
                num_decoder_layers=num_layers,
                dim_feedforward=int(d_model * mlp_ratio),
                dropout=dropout,
                topo_k=self.topo_k,
                topo_include_bos=self.topo_include_bos,
            )
        else:
            raise ValueError(f"Unknown arch={self.arch}")

        self.pred_head = nn.Linear(d_model, d_model)

    def embed_tokens(self, feat, type_id):
        b, t, _ = feat.shape
        return self.token_mlp(feat) + self.type_emb(type_id) + self.pos_emb[:, :t, :]

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
                raise ValueError("encdec arch requires qa_layout='split' (answers contiguous)")

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
