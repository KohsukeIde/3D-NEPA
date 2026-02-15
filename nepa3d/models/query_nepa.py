import torch
import torch.nn as nn
import torch.nn.functional as F

from .causal_transformer import CausalTransformer
from ..token.tokenizer import (
    TYPE_MISSING_RAY,
    TYPE_A_POINT,
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
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.type_emb = nn.Embedding(n_types, d_model)
        self.token_mlp = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.backbone = CausalTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.pred_head = nn.Linear(d_model, d_model)

    def embed_tokens(self, feat, type_id):
        b, t, _ = feat.shape
        return self.token_mlp(feat) + self.type_emb(type_id) + self.pos_emb[:, :t, :]

    def forward(
        self,
        feat,
        type_id,
        dual_mask_near: float = 0.0,
        dual_mask_far: float = 0.0,
        dual_mask_window: int = 0,
        dual_mask_seed: int | None = None,
    ):
        """Forward.

        dual_mask_* are only used during training (see CausalTransformer).
        """
        z = self.embed_tokens(feat, type_id)
        h = self.backbone(
            z,
            dual_mask_near=float(dual_mask_near),
            dual_mask_far=float(dual_mask_far),
            dual_mask_window=int(dual_mask_window),
            dual_mask_seed=None if dual_mask_seed is None else int(dual_mask_seed),
        )
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
