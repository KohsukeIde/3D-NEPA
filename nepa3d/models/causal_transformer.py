from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..token.tokenizer import (
    TYPE_MISSING_RAY,
    TYPE_POINT,
    TYPE_Q_POINT,
    TYPE_Q_POINT_MESH,
    TYPE_Q_POINT_UDF,
    TYPE_Q_POINT_PC,
    TYPE_Q_RAY,
    TYPE_RAY,
)


def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    p = float(drop_prob)
    if p <= 0.0 or (not training):
        return x
    keep_prob = 1.0 - p
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    rnd = keep_prob + torch.rand(shape, device=x.device, dtype=x.dtype)
    rnd.floor_()
    return x.div(keep_prob) * rnd


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


class _PointMAEMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        drop: float,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(int(in_features), int(hidden_features))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(hidden_features), int(out_features))
        self.drop = nn.Dropout(float(drop))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _PointMAESelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qkv_bias: bool,
        attn_drop: float,
        proj_drop: float,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.dim = int(dim)
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim={self.dim} must be divisible by num_heads={self.num_heads}")
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=bool(qkv_bias))
        self.attn_drop = nn.Dropout(float(attn_drop))
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(float(proj_drop))

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        b, t, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, t, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                min_val = torch.finfo(scores.dtype).min
                if attn_mask.dim() == 2:
                    scores = scores.masked_fill(attn_mask[None, None, :, :], min_val)
                elif attn_mask.dim() == 3:
                    scores = scores.masked_fill(attn_mask[:, None, :, :], min_val)
                else:
                    raise ValueError(f"attn_mask(bool) must be 2D/3D, got {tuple(attn_mask.shape)}")
            else:
                if attn_mask.dim() == 2:
                    scores = scores + attn_mask[None, None, :, :]
                elif attn_mask.dim() == 3:
                    scores = scores + attn_mask[:, None, :, :]
                else:
                    raise ValueError(f"attn_mask(additive) must be 2D/3D, got {tuple(attn_mask.shape)}")

        attn = scores.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, t, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _PointMAEBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(int(dim))
        self.attn = _PointMAESelfAttention(
            int(dim),
            int(num_heads),
            qkv_bias=bool(qkv_bias),
            attn_drop=float(attn_drop),
            proj_drop=float(drop),
        )
        self.drop_path = DropPath(float(drop_path)) if float(drop_path) > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(int(dim))
        mlp_hidden_dim = int(int(dim) * float(mlp_ratio))
        self.mlp = _PointMAEMlp(
            in_features=int(dim),
            hidden_features=mlp_hidden_dim,
            out_features=int(dim),
            drop=float(drop),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), attn_mask=attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class _QKNorm(nn.Module):
    def __init__(
        self,
        head_dim: int,
        *,
        eps: float,
        affine: bool,
        bias: bool,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.head_dim = int(head_dim)
        self.weight = nn.Parameter(torch.ones(self.head_dim)) if bool(affine) else None
        self.bias = nn.Parameter(torch.zeros(self.head_dim)) if bool(bias) else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.head_dim,), self.weight, self.bias, self.eps)


class _LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float | None) -> None:
        super().__init__()
        if init_value is None:
            self.gamma = None
        else:
            v = float(init_value)
            self.gamma = nn.Parameter(v * torch.ones(int(dim))) if v > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gamma is None:
            return x
        return x * self.gamma


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    num_prefix_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    t = int(q.shape[-2])
    prefix = max(0, int(num_prefix_tokens))
    if prefix >= t:
        return q, k

    if prefix <= 0:
        q_main = q
        k_main = k
        q_prefix = None
        k_prefix = None
    else:
        q_prefix, q_main = q.split((prefix, t - prefix), dim=-2)
        k_prefix, k_main = k.split((prefix, t - prefix), dim=-2)

    q_main = (q_main * cos) + (_rotate_half(q_main) * sin)
    k_main = (k_main * cos) + (_rotate_half(k_main) * sin)

    if prefix <= 0:
        return q_main, k_main
    return torch.cat((q_prefix, q_main), dim=-2), torch.cat((k_prefix, k_main), dim=-2)


class _NepaSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        *,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        qkv_bias: bool,
        qk_norm: bool,
        qk_norm_affine: bool,
        qk_norm_bias: bool,
        rope_theta: float,
        rope_prefix_tokens: int,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        if self.d_model % self.nhead != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by nhead={self.nhead}")

        self.head_dim = self.d_model // self.nhead
        self.scaling = self.head_dim ** -0.5

        self.query = nn.Linear(self.d_model, self.d_model, bias=bool(qkv_bias))
        self.key = nn.Linear(self.d_model, self.d_model, bias=bool(qkv_bias))
        self.value = nn.Linear(self.d_model, self.d_model, bias=bool(qkv_bias))
        self.proj = nn.Linear(self.d_model, self.d_model)

        self.attn_dropout = float(attention_probs_dropout_prob)
        self.proj_dropout = float(hidden_dropout_prob)

        if bool(qk_norm):
            self.q_norm = _QKNorm(
                self.head_dim,
                eps=float(layer_norm_eps),
                affine=bool(qk_norm_affine),
                bias=bool(qk_norm_bias),
            )
            self.k_norm = _QKNorm(
                self.head_dim,
                eps=float(layer_norm_eps),
                affine=bool(qk_norm_affine),
                bias=bool(qk_norm_bias),
            )
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.rope_prefix_tokens = max(0, int(rope_prefix_tokens))
        # Match ViT-NEPA-style RoPE parameterization (head_dim must be divisible by 4).
        self.use_rope = bool(float(rope_theta) > 0.0 and (self.head_dim % 4 == 0))
        if self.use_rope:
            inv_freq = 1.0 / (
                float(rope_theta)
                ** torch.arange(0, 1, 4.0 / float(self.head_dim), dtype=torch.float32)
            )
            self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

    def _rope_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        # 2D NEPA uses normalized patch-center coordinates in [-1, 1].
        # For 3D sequence tokens, we map sequence centers to [-1, 1] and duplicate
        # them into 2 channels so the angle construction is structurally identical.
        pos = (torch.arange(int(seq_len), dtype=torch.float32, device=device) + 0.5) / float(max(1, int(seq_len)))
        coord = 2.0 * pos - 1.0  # [-1,1]
        coords = torch.stack([coord, coord], dim=-1)  # (T,2)

        angles = 2.0 * math.pi * coords[:, :, None] * self.rope_inv_freq[None, None, :]
        angles = angles.flatten(1, 2).tile(1, 2)  # (T, head_dim)
        cos = angles.cos()[None, None, :, :].to(dtype=dtype)
        sin = angles.sin()[None, None, :, :].to(dtype=dtype)
        return cos, sin

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        b, t, _ = x.shape

        q = self.query(x).view(b, t, self.nhead, self.head_dim).transpose(1, 2)
        k = self.key(x).view(b, t, self.nhead, self.head_dim).transpose(1, 2)
        v = self.value(x).view(b, t, self.nhead, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.use_rope and (t > self.rope_prefix_tokens):
            cos, sin = self._rope_cos_sin(t - self.rope_prefix_tokens, device=x.device, dtype=q.dtype)
            q, k = _apply_rope(q, k, cos, sin, num_prefix_tokens=self.rope_prefix_tokens)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                if attn_mask.dim() == 2:
                    scores = scores.masked_fill(attn_mask[None, None, :, :], torch.finfo(scores.dtype).min)
                elif attn_mask.dim() == 3:
                    scores = scores.masked_fill(attn_mask[:, None, :, :], torch.finfo(scores.dtype).min)
                else:
                    raise ValueError(f"attn_mask(bool) must be 2D/3D, got {tuple(attn_mask.shape)}")
            else:
                if attn_mask.dim() == 2:
                    scores = scores + attn_mask[None, None, :, :]
                elif attn_mask.dim() == 3:
                    scores = scores + attn_mask[:, None, :, :]
                else:
                    raise ValueError(f"attn_mask(additive) must be 2D/3D, got {tuple(attn_mask.shape)}")

        probs = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        if self.attn_dropout > 0.0 and self.training:
            probs = F.dropout(probs, p=self.attn_dropout, training=True)

        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        out = self.proj(out)
        if self.proj_dropout > 0.0 and self.training:
            out = F.dropout(out, p=self.proj_dropout, training=True)
        return out


class _NepaMLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
        *,
        hidden_dropout_prob: float,
        use_gated_mlp: bool,
        hidden_act: str = "gelu",
    ) -> None:
        super().__init__()
        self.use_gated_mlp = bool(use_gated_mlp)
        self.up_proj = nn.Linear(int(d_model), int(intermediate_size))
        self.gate_proj = nn.Linear(int(d_model), int(intermediate_size)) if self.use_gated_mlp else None
        self.fc2 = nn.Linear(int(intermediate_size), int(d_model))
        self.dropout = float(hidden_dropout_prob)

        act_name = str(hidden_act).lower()
        if act_name == "gelu":
            self.act = nn.GELU()
        elif act_name in {"silu", "swish"}:
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported hidden_act={hidden_act} (supported: gelu|silu)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.up_proj(x)
        if self.use_gated_mlp:
            gate = self.act(self.gate_proj(x))
            x = gate * up
        else:
            x = self.act(up)
        x = self.fc2(x)
        if self.dropout > 0.0 and self.training:
            x = F.dropout(x, p=self.dropout, training=True)
        return x


class _NepaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        intermediate_size: int,
        *,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        drop_path: float,
        qkv_bias: bool,
        qk_norm: bool,
        qk_norm_affine: bool,
        qk_norm_bias: bool,
        layerscale_value: float,
        rope_theta: float,
        rope_prefix_tokens: int,
        layer_norm_eps: float,
        use_gated_mlp: bool,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.layernorm_before = nn.LayerNorm(int(d_model), eps=float(layer_norm_eps))
        self.attn = _NepaSelfAttention(
            d_model=int(d_model),
            nhead=int(nhead),
            hidden_dropout_prob=float(hidden_dropout_prob),
            attention_probs_dropout_prob=float(attention_probs_dropout_prob),
            qkv_bias=bool(qkv_bias),
            qk_norm=bool(qk_norm),
            qk_norm_affine=bool(qk_norm_affine),
            qk_norm_bias=bool(qk_norm_bias),
            rope_theta=float(rope_theta),
            rope_prefix_tokens=int(rope_prefix_tokens),
            layer_norm_eps=float(layer_norm_eps),
        )
        self.layer_scale1 = _LayerScale(int(d_model), float(layerscale_value))

        self.layernorm_after = nn.LayerNorm(int(d_model), eps=float(layer_norm_eps))
        self.mlp = _NepaMLP(
            d_model=int(d_model),
            intermediate_size=int(intermediate_size),
            hidden_dropout_prob=float(hidden_dropout_prob),
            use_gated_mlp=bool(use_gated_mlp),
            hidden_act=str(hidden_act),
        )
        self.layer_scale2 = _LayerScale(int(d_model), float(layerscale_value))

        self.drop_path_rate = float(drop_path)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + _drop_path(self.layer_scale1(self.attn(self.layernorm_before(x), attn_mask=attn_mask)), self.drop_path_rate, self.training)
        x = x + _drop_path(self.layer_scale2(self.mlp(self.layernorm_after(x))), self.drop_path_rate, self.training)
        return x


class CausalTransformer(nn.Module):
    def __init__(
        self,
        d_model=384,
        nhead=6,
        num_layers=8,
        mlp_ratio=4,
        dropout=0.0,
        drop_path=0.0,
        *,
        backbone_impl: str = "nepa2d",
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_affine: bool = False,
        qk_norm_bias: bool = False,
        layerscale_value: float = 1e-5,
        rope_theta: float = 100.0,
        rope_prefix_tokens: int = 1,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        use_gated_mlp: bool = False,
        hidden_act: str = "gelu",
        final_layernorm: bool = True,
    ):
        super().__init__()
        self.backbone_impl = str(backbone_impl).lower()

        drop_path = self._clamp01(drop_path)
        if int(num_layers) <= 1:
            rates = [drop_path]
        else:
            rates = torch.linspace(0.0, float(drop_path), int(num_layers)).tolist()
        self.drop_path_rates = [float(r) for r in rates]

        if self.backbone_impl == "legacy":
            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=int(d_model),
                        nhead=int(nhead),
                        dim_feedforward=int(d_model * mlp_ratio),
                        dropout=float(dropout),
                        batch_first=True,
                        norm_first=True,
                        activation="gelu",
                    )
                    for _ in range(int(num_layers))
                ]
            )
        elif self.backbone_impl == "pointmae":
            self.layers = nn.ModuleList(
                [
                    _PointMAEBlock(
                        dim=int(d_model),
                        num_heads=int(nhead),
                        mlp_ratio=float(mlp_ratio),
                        qkv_bias=bool(qkv_bias),
                        drop=float(dropout),
                        attn_drop=float(dropout),
                        drop_path=float(self.drop_path_rates[i]),
                    )
                    for i in range(int(num_layers))
                ]
            )
        elif self.backbone_impl == "nepa2d":
            self.blocks = nn.ModuleList(
                [
                    _NepaBlock(
                        d_model=int(d_model),
                        nhead=int(nhead),
                        intermediate_size=int(d_model * mlp_ratio),
                        hidden_dropout_prob=float(hidden_dropout_prob),
                        attention_probs_dropout_prob=float(attention_probs_dropout_prob),
                        drop_path=float(self.drop_path_rates[i]),
                        qkv_bias=bool(qkv_bias),
                        qk_norm=bool(qk_norm),
                        qk_norm_affine=bool(qk_norm_affine),
                        qk_norm_bias=bool(qk_norm_bias),
                        layerscale_value=float(layerscale_value),
                        rope_theta=float(rope_theta),
                        rope_prefix_tokens=int(rope_prefix_tokens),
                        layer_norm_eps=float(layer_norm_eps),
                        use_gated_mlp=bool(use_gated_mlp),
                        hidden_act=str(hidden_act),
                    )
                    for i in range(int(num_layers))
                ]
            )
            self.final_ln = nn.LayerNorm(int(d_model), eps=float(layer_norm_eps)) if bool(final_layernorm) else nn.Identity()
        else:
            raise ValueError(f"Unknown backbone_impl={self.backbone_impl}; use legacy|pointmae|nepa2d")

    @staticmethod
    def _clamp01(x: float) -> float:
        x = float(x)
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    @staticmethod
    def _apply_drop_path(x_prev: torch.Tensor, x_next: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
        if (not training) or float(drop_prob) <= 0.0:
            return x_next
        return x_prev + _drop_path(x_next - x_prev, float(drop_prob), training)

    def forward(
        self,
        x,
        is_causal: bool = True,
        type_id=None,
        pos: Optional[torch.Tensor] = None,
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
                        | (tid == TYPE_Q_POINT_MESH)
                        | (tid == TYPE_Q_POINT_UDF)
                        | (tid == TYPE_Q_POINT_PC)
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

        # Missing-ray tokens should not be used as K/V for non-missing queries.
        # We keep at least self-attention for missing query rows to avoid all-masked rows.
        if bool(dual_mask_type_aware) and type_id is not None:
            tid = type_id
            if tid.dim() == 1:
                tid = tid.unsqueeze(0)
            if tid.dim() != 2 or tid.shape[1] != t:
                raise ValueError(f"type_id must be (B,T) with T={t}, got {tuple(tid.shape)}")

            is_missing = tid == int(TYPE_MISSING_RAY)  # (B,T)
            if bool(is_missing.any()):
                miss_kv = is_missing[:, None, :].expand(-1, t, -1).clone()  # (B,T,T)
                eye = torch.eye(t, device=x.device, dtype=torch.bool).unsqueeze(0)  # (1,T,T)
                self_keep = is_missing.unsqueeze(-1) & eye
                miss_kv = miss_kv & (~self_keep)

                if attn_mask is None:
                    attn_mask = miss_kv
                elif attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0).expand(tid.shape[0], -1, -1) | miss_kv
                else:
                    attn_mask = attn_mask | miss_kv

        if self.backbone_impl == "legacy":
            for li, layer in enumerate(self.layers):
                # Point-MAE-style positional injection for vanilla encoder path:
                # add positional embedding at every block input.
                x_in = x + pos if pos is not None else x
                x_next = layer(x_in, src_mask=attn_mask)
                x = self._apply_drop_path(x, x_next, self.drop_path_rates[li], self.training)
            return x

        if self.backbone_impl == "pointmae":
            for layer in self.layers:
                # Point-MAE convention: inject absolute positional embedding per block.
                x_in = x + pos if pos is not None else x
                x = layer(x_in, attn_mask=attn_mask)
            return x

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        x = self.final_ln(x)
        return x
