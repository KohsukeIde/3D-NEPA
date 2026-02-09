import torch
import torch.nn as nn
import torch.nn.functional as F

from .causal_transformer import CausalTransformer
from ..token.tokenizer import TYPE_MISSING_RAY


class QueryNepa(nn.Module):
    def __init__(
        self,
        feat_dim=15,
        d_model=384,
        n_types=5,
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

    def forward(self, feat, type_id):
        z = self.embed_tokens(feat, type_id)
        h = self.backbone(z)
        z_hat = self.pred_head(h)
        return z, z_hat, h

    def nepa_loss(self, z, z_hat, type_id=None):
        pred = z_hat[:, :-1, :]
        target = z[:, 1:, :].detach()
        loss = 1.0 - F.cosine_similarity(pred, target, dim=-1, eps=1e-8)
        if type_id is None:
            return loss.mean()
        target_type = type_id[:, 1:]
        valid = target_type != TYPE_MISSING_RAY
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
