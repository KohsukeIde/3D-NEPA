"""Checkpoint utilities.

This repo often evolves model hyper-parameters (e.g., sequence length / positional
embeddings). For long-sequence scaling we want to:

1) Build a model with a *new* max_len.
2) Load an older checkpoint by resizing the learned 1D positional embedding.

We intentionally keep this logic in a small helper so that train / eval scripts
can share the exact same behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F


def resize_pos_emb_1d(pos_emb: torch.Tensor, new_len: int) -> torch.Tensor:
    """Resize a learned 1D positional embedding.

    Args:
        pos_emb: Tensor shaped [1, L, D] (as used by QueryNepa.pos_emb).
        new_len: Target length.

    Returns:
        Resized positional embedding of shape [1, new_len, D].

    Notes:
        - Uses linear interpolation along sequence length.
        - Works for both upsampling and downsampling.
    """

    if pos_emb.ndim != 3 or pos_emb.shape[0] != 1:
        raise ValueError(
            f"pos_emb must have shape [1, L, D], got {tuple(pos_emb.shape)}"
        )
    old_len = int(pos_emb.shape[1])
    if int(new_len) == old_len:
        return pos_emb
    if int(new_len) <= 0:
        raise ValueError(f"new_len must be > 0, got {new_len}")

    # [1, L, D] -> [1, D, L]
    x = pos_emb.permute(0, 2, 1)
    x = F.interpolate(x, size=int(new_len), mode="linear", align_corners=False)
    # [1, D, L] -> [1, L, D]
    return x.permute(0, 2, 1).contiguous()


def maybe_resize_pos_emb_in_state_dict(
    state_dict: Dict[str, torch.Tensor],
    new_len: int,
    *,
    key: str = "pos_emb",
) -> Dict[str, torch.Tensor]:
    """Return a (shallow) copied state_dict with resized pos_emb if needed."""

    if key not in state_dict:
        return state_dict

    pos = state_dict[key]
    if not torch.is_tensor(pos):
        return state_dict

    if pos.ndim != 3:
        return state_dict
    old_len = int(pos.shape[1])
    if old_len == int(new_len):
        return state_dict

    new_state = dict(state_dict)
    new_state[key] = resize_pos_emb_1d(pos, int(new_len))
    return new_state


def maybe_resize_type_emb_in_state_dict(
    state_dict: Dict[str, torch.Tensor],
    new_n_types: int,
    *,
    key: str = "type_emb.weight",
    init: str = "copy_bos",
) -> Dict[str, torch.Tensor]:
    """Return a (shallow) copied state_dict with resized type embedding if needed.

    Useful when the code adds new token types (e.g., SEP) but we still want to load
    checkpoints trained with an older/smaller type vocabulary.
    """

    if key not in state_dict:
        return state_dict

    w = state_dict[key]
    if not torch.is_tensor(w) or w.ndim != 2:
        return state_dict

    old_n, d = int(w.shape[0]), int(w.shape[1])
    new_n = int(new_n_types)
    if old_n == new_n:
        return state_dict

    new_state = dict(state_dict)
    new_w = torch.zeros((new_n, d), dtype=w.dtype, device=w.device)
    n = min(old_n, new_n)
    new_w[:n] = w[:n]

    if new_n > old_n:
        if init == "copy_bos" and old_n > 0:
            new_w[old_n:new_n] = w[0:1].repeat(new_n - old_n, 1)
        # else: keep zeros

    new_state[key] = new_w
    return new_state


def maybe_resize_type_pos_emb_in_state_dict(
    state_dict: Dict[str, torch.Tensor],
    new_n_types: int,
    new_len: int,
    *,
    key: str = "type_pos_emb.weight",
    pos_key: str = "pos_emb",
) -> Dict[str, torch.Tensor]:
    """Resize flattened (n_types * max_len, d) type_pos_emb in a checkpoint.

    We infer old max_len from `pos_emb` in the same checkpoint.
    """

    if key not in state_dict or pos_key not in state_dict:
        return state_dict
    w = state_dict[key]
    pos = state_dict[pos_key]
    if (not torch.is_tensor(w)) or (not torch.is_tensor(pos)):
        return state_dict
    if w.ndim != 2 or pos.ndim != 3:
        return state_dict

    old_total, d = int(w.shape[0]), int(w.shape[1])
    old_len = int(pos.shape[1])
    if old_len <= 0:
        return state_dict
    if old_total % old_len != 0:
        return state_dict
    old_n_types = old_total // old_len

    new_len = int(new_len)
    new_n_types = int(new_n_types)
    new_total = new_n_types * new_len
    if old_total == new_total:
        return state_dict

    w3 = w.reshape(old_n_types, old_len, d)
    new3 = torch.zeros((new_n_types, new_len, d), dtype=w.dtype, device=w.device)

    nt = min(old_n_types, new_n_types)
    nl = min(old_len, new_len)
    new3[:nt, :nl] = w3[:nt, :nl]

    # Expand length: tile first position.
    if new_len > old_len:
        new3[:nt, old_len:new_len] = new3[:nt, :1].repeat(1, new_len - old_len, 1)

    # Expand types: copy type-0.
    if new_n_types > old_n_types and old_n_types > 0:
        new3[old_n_types:new_n_types] = new3[0:1].repeat(new_n_types - old_n_types, 1, 1)

    new_state = dict(state_dict)
    new_state[key] = new3.reshape(new_total, d)
    return new_state


@dataclass
class LoadStateReport:
    missing_keys: List[str]
    unexpected_keys: List[str]


def load_state_dict_flexible(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    strict: bool = True,
    allow_missing_prefixes: Sequence[str] = (),
    allow_unexpected_prefixes: Sequence[str] = (),
) -> LoadStateReport:
    """Load state_dict with optional allow-lists.

    This is stricter than `strict=False` because it still errors on *unknown*
    mismatches, while allowing intentional optional modules.
    """

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    def _is_allowed(key: str, prefixes: Sequence[str]) -> bool:
        return any(key.startswith(p) for p in prefixes)

    if strict:
        bad_missing = [k for k in missing if not _is_allowed(k, allow_missing_prefixes)]
        bad_unexp = [k for k in unexpected if not _is_allowed(k, allow_unexpected_prefixes)]
        if bad_missing or bad_unexp:
            msg = ["state_dict mismatch:"]
            if bad_missing:
                msg.append(f"  missing_keys({len(bad_missing)}): {bad_missing[:50]}")
            if bad_unexp:
                msg.append(
                    f"  unexpected_keys({len(bad_unexp)}): {bad_unexp[:50]}"
                )
            raise RuntimeError("\n".join(msg))

    return LoadStateReport(missing_keys=missing, unexpected_keys=unexpected)


def infer_causal_backbone_impl(
    pre_args: Mapping[str, object],
    state_dict: Mapping[str, torch.Tensor],
    *,
    default: str = "legacy",
) -> str:
    """Infer 3D causal backbone implementation from checkpoint metadata/state.

    Returns one of: {"legacy", "nepa2d"}.
    """
    impl = str(pre_args.get("backbone_impl", "")).strip().lower()
    if impl in ("legacy", "nepa2d"):
        return impl

    # Legacy CausalTransformer (nn.TransformerEncoderLayer-based)
    if any(k.startswith("backbone.layers.") for k in state_dict.keys()):
        return "legacy"

    # New NEPA2D-style causal blocks
    if any(k.startswith("backbone.blocks.") for k in state_dict.keys()):
        return "nepa2d"

    d = str(default).strip().lower()
    return d if d in ("legacy", "nepa2d") else "legacy"


def infer_causal_support_kwargs(
    pre_args: Mapping[str, object],
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, object]:
    """Infer causal-backbone support kwargs with sensible defaults.

    Defaults are chosen to preserve old checkpoints while enabling new NEPA2D-style
    support parts for freshly trained checkpoints.
    """
    impl = infer_causal_backbone_impl(pre_args, state_dict, default="legacy")
    if impl == "legacy":
        return {
            "backbone_impl": "legacy",
            "qkv_bias": True,
            "qk_norm": False,
            "qk_norm_affine": False,
            "qk_norm_bias": False,
            "layerscale_value": 0.0,
            "rope_theta": 0.0,
            "layer_norm_eps": 1e-5,
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "use_gated_mlp": False,
            "hidden_act": "gelu",
            "final_layernorm": False,
        }

    has_final_ln = any(k.startswith("backbone.final_ln.") for k in state_dict.keys())
    return {
        "backbone_impl": "nepa2d",
        "qkv_bias": bool(int(pre_args.get("qkv_bias", 1))),
        "qk_norm": bool(int(pre_args.get("qk_norm", 1))),
        "qk_norm_affine": bool(int(pre_args.get("qk_norm_affine", 0))),
        "qk_norm_bias": bool(int(pre_args.get("qk_norm_bias", 0))),
        "layerscale_value": float(pre_args.get("layerscale_value", 1e-5)),
        "rope_theta": float(pre_args.get("rope_theta", 100.0)),
        "layer_norm_eps": float(pre_args.get("layer_norm_eps", 1e-12)),
        "hidden_dropout_prob": float(pre_args.get("hidden_dropout_prob", 0.0)),
        "attention_probs_dropout_prob": float(pre_args.get("attention_probs_dropout_prob", 0.0)),
        "use_gated_mlp": bool(int(pre_args.get("use_gated_mlp", 0))),
        "hidden_act": str(pre_args.get("hidden_act", "gelu")),
        "final_layernorm": bool(int(pre_args.get("final_layernorm", 1))) if ("final_layernorm" in pre_args) else bool(has_final_ln),
    }
