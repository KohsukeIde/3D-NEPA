import torch
import torch.nn as nn


class CausalTransformer(nn.Module):
    def __init__(self, d_model=384, nhead=6, num_layers=8, mlp_ratio=4, dropout=0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)

    @staticmethod
    def _clamp01(x: float) -> float:
        x = float(x)
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    def forward(
        self,
        x,
        dual_mask_near: float = 0.0,
        dual_mask_far: float = 0.0,
        dual_mask_window: int = 0,
        dual_mask_seed: int | None = None,
    ):
        """Causal transformer with optional *dual masking* (PointGPT-style).

        Base mask: standard causal (future tokens are masked).
        Dual mask: additionally masks *some past tokens* stochastically, which reduces
        short-range redundancy / geometric shortcut incentives in AR settings.

        Args:
            dual_mask_near: probability to mask past tokens within `dual_mask_window`.
            dual_mask_far: probability to mask past tokens outside the window.
            dual_mask_window: window size in token steps (<=0 disables near/far split).
            dual_mask_seed: optional seed for deterministic dual masking.
        """
        t = x.size(1)
        # Base causal mask (True = blocked)
        attn_mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)

        # Dual masking only during training.
        if self.training:
            p_near = self._clamp01(dual_mask_near)
            p_far = self._clamp01(dual_mask_far)
            if (p_near > 0.0) or (p_far > 0.0):
                i = torch.arange(t, device=x.device)[:, None]
                j = torch.arange(t, device=x.device)[None, :]
                rel = i - j  # >0 means "past" token
                past = rel > 0

                if int(dual_mask_window) > 0:
                    win = int(dual_mask_window)
                    prob = torch.full((t, t), p_far, device=x.device, dtype=torch.float32)
                    prob = prob.masked_fill(past & (rel <= win), p_near)
                else:
                    prob = torch.full((t, t), p_far, device=x.device, dtype=torch.float32)

                gen = None
                if dual_mask_seed is not None:
                    gen = torch.Generator(device=x.device)
                    gen.manual_seed(int(dual_mask_seed) & 0xFFFFFFFF)

                u = torch.rand((t, t), device=x.device, generator=gen)
                extra = past & (u < prob)
                attn_mask = attn_mask | extra

        return self.enc(x, mask=attn_mask)
