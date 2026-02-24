from __future__ import annotations

import torch
import torch.nn as nn

from ..token.tokenizer import TYPE_POINT, TYPE_Q_POINT, TYPE_Q_RAY, TYPE_RAY


class CausalTransformer(nn.Module):
    def __init__(
        self,
        d_model=384,
        nhead=6,
        num_layers=8,
        mlp_ratio=4,
        dropout=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * mlp_ratio,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                    activation="gelu",
                )
                for _ in range(int(num_layers))
            ]
        )
        drop_path = self._clamp01(drop_path)
        if int(num_layers) <= 1:
            rates = [drop_path]
        else:
            rates = torch.linspace(0.0, float(drop_path), int(num_layers)).tolist()
        self.drop_path_rates = [float(r) for r in rates]

    @staticmethod
    def _clamp01(x: float) -> float:
        x = float(x)
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    @staticmethod
    def _apply_drop_path(x_prev: torch.Tensor, x_next: torch.Tensor, drop_prob: float) -> torch.Tensor:
        """Apply block-level stochastic depth to residual (x_next - x_prev)."""
        p = float(drop_prob)
        if p <= 0.0 or (not x_prev.requires_grad) or (not x_prev.is_floating_point()):
            return x_next
        if p >= 1.0:
            return x_prev
        if not x_prev.is_cuda and x_prev.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            return x_next
        keep_prob = 1.0 - p
        shape = (x_prev.shape[0],) + (1,) * (x_prev.ndim - 1)
        rnd = torch.rand(shape, device=x_prev.device, dtype=x_prev.dtype)
        mask = (rnd < keep_prob).to(x_prev.dtype)
        residual = (x_next - x_prev) * mask / keep_prob
        return x_prev + residual

    def forward(
        self,
        x,
        is_causal: bool = True,
        type_id=None,
        dual_mask_near: float = 0.0,
        dual_mask_far: float = 0.0,
        dual_mask_window: int = 0,
        dual_mask_seed: int | None = None,
        dual_mask_type_aware: int | bool = 0,
    ):
        """Causal transformer with optional *dual masking* (PointGPT-style).

        Base mask: standard causal (future tokens are masked) when `is_causal=True`.
        Dual mask: additionally masks *some past tokens* stochastically, which reduces
        short-range redundancy / geometric shortcut incentives in AR settings.

        Args:
            dual_mask_near: probability to mask past tokens within `dual_mask_window`.
            dual_mask_far: probability to mask past tokens outside the window.
            dual_mask_window: window size in token steps (<=0 disables near/far split).
            dual_mask_seed: optional seed for deterministic dual masking.
        """
        t = x.size(1)
        attn_mask = None
        if bool(is_causal):
            # Base causal mask (True = blocked)
            attn_mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)

        # Dual masking only during training with causal attention.
        if bool(is_causal) and self.training:
            p_near = self._clamp01(dual_mask_near)
            p_far = self._clamp01(dual_mask_far)
            if (p_near > 0.0) or (p_far > 0.0):
                i = torch.arange(t, device=x.device)[:, None]
                j = torch.arange(t, device=x.device)[None, :]
                rel = i - j  # >0 means "past" token
                past = rel > 0

                # Optional type-aware mode: only drop Q/Q edges so query-to-answer
                # information remains visible while suppressing local geometry copy.
                eligible = past
                if bool(dual_mask_type_aware) and type_id is not None:
                    tid = type_id
                    if tid.dim() == 1:
                        tid = tid.unsqueeze(0)
                    is_query_like = (
                        (tid == TYPE_Q_POINT)
                        | (tid == TYPE_Q_RAY)
                        | (tid == TYPE_POINT)
                        | (tid == TYPE_RAY)
                    )
                    is_query_pos = is_query_like.any(dim=0)
                    qq = is_query_pos[:, None] & is_query_pos[None, :]
                    eligible = past & qq

                if int(dual_mask_window) > 0:
                    win = int(dual_mask_window)
                    prob = torch.zeros((t, t), device=x.device, dtype=torch.float32)
                    prob = prob.masked_fill(eligible & (rel <= win), p_near)
                    prob = prob.masked_fill(eligible & (rel > win), p_far)
                else:
                    prob = torch.zeros((t, t), device=x.device, dtype=torch.float32)
                    prob = prob.masked_fill(eligible, p_far)

                gen = None
                if dual_mask_seed is not None:
                    gen = torch.Generator(device=x.device)
                    gen.manual_seed(int(dual_mask_seed) & 0xFFFFFFFF)

                u = torch.rand((t, t), device=x.device, generator=gen)
                extra = eligible & (u < prob)
                attn_mask = attn_mask | extra

        for li, layer in enumerate(self.layers):
            x_next = layer(x, src_mask=attn_mask)
            if self.training:
                x = self._apply_drop_path(x, x_next, self.drop_path_rates[li])
            else:
                x = x_next
        return x
