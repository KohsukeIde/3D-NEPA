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
