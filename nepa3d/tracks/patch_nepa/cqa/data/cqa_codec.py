"""Fixed discrete answer-language codec for explicit-query CQA.

The vocabulary is intentionally versioned and fixed in code so checkpoints,
evaluation, and visualization stay compatible across runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Versioned query / answer vocabulary spec
# -----------------------------------------------------------------------------

CQA_VOCAB_VERSION = "cqa_v1"

ASK_NORMAL = 0
ASK_VISIBILITY = 1
ASK_CURVATURE = 2
ASK_THICKNESS = 3
ASK_CLEARANCE = 4
ASK_DISTANCE = 5
QUERY_TYPE_VOCAB_SIZE = 6

QUERY_TYPE_NAMES = {
    ASK_NORMAL: "mesh_normal",
    ASK_VISIBILITY: "mesh_visibility",
    ASK_CURVATURE: "mesh_curvature",
    ASK_THICKNESS: "udf_thickness",
    ASK_CLEARANCE: "udf_clearance",
    ASK_DISTANCE: "udf_distance",
}
QUERY_NAME_TO_ID = {v: k for k, v in QUERY_TYPE_NAMES.items()}

# Shared answer vocab with fixed reserved ranges.
ANSWER_RANGES: Dict[str, Tuple[int, int]] = {
    "mesh_normal": (0, 128),
    "mesh_visibility": (128, 384),
    "mesh_curvature": (384, 448),
    "udf_thickness": (448, 512),
    "udf_clearance": (512, 576),
    "udf_distance": (576, 640),
}
ANSWER_VOCAB_SIZE = 640


@dataclass(frozen=True)
class ScalarBinning:
    n_bins: int
    vmin: float
    vmax: float
    log: bool = False


def _fibonacci_dirs(n: int) -> np.ndarray:
    n = int(max(n, 1))
    i = np.arange(n, dtype=np.float32)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - (2.0 * i + 1.0) / float(n)
    r = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
    theta = phi * i
    x = np.cos(theta) * r
    z = np.sin(theta) * r
    d = np.stack([x, y, z], axis=1).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-8
    return d


def _hemisphere_dirs(n: int) -> np.ndarray:
    n = int(max(n, 1))
    i = np.arange(n, dtype=np.float32)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    z = (i + np.float32(0.5)) / np.float32(n)
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    theta = phi * i
    x = np.cos(theta) * r
    y = np.sin(theta) * r
    d = np.stack([x, y, z], axis=1).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-8
    return d


def canonicalize_normals_unsigned(normals: np.ndarray) -> np.ndarray:
    n = np.asarray(normals, dtype=np.float32).reshape(-1, 3).copy()
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-8
    eps = np.float32(1e-8)
    flip = (n[:, 2] < 0.0)
    tie_z = np.abs(n[:, 2]) <= eps
    flip |= tie_z & (n[:, 1] < 0.0)
    tie_zy = tie_z & (np.abs(n[:, 1]) <= eps)
    flip |= tie_zy & (n[:, 0] < 0.0)
    n[flip] *= np.float32(-1.0)
    return n.astype(np.float32, copy=False)


_NORMAL_DIRS = _fibonacci_dirs(ANSWER_RANGES["mesh_normal"][1] - ANSWER_RANGES["mesh_normal"][0])
_NORMAL_UNSIGNED_DIRS = _hemisphere_dirs(ANSWER_RANGES["mesh_normal"][1] - ANSWER_RANGES["mesh_normal"][0])
_CURV_SPEC = ScalarBinning(n_bins=64, vmin=-0.5, vmax=0.5, log=False)
_THICK_SPEC = ScalarBinning(n_bins=64, vmin=1e-3, vmax=1.0, log=True)
_CLEAR_SPEC = ScalarBinning(n_bins=64, vmin=1e-3, vmax=1.0, log=True)
_DIST_SPEC = ScalarBinning(n_bins=64, vmin=1e-3, vmax=1.0, log=True)


def answer_range_for_query_name(query_name: str) -> Tuple[int, int]:
    if query_name not in ANSWER_RANGES:
        raise KeyError(f"unknown query_name={query_name}")
    return ANSWER_RANGES[str(query_name)]


def answer_range_for_query_type(query_type: int) -> Tuple[int, int]:
    qt = int(query_type)
    if qt not in QUERY_TYPE_NAMES:
        raise KeyError(f"unknown query_type={qt}")
    return answer_range_for_query_name(QUERY_TYPE_NAMES[qt])


def valid_answer_mask_for_query_type(query_type: torch.Tensor, *, vocab_size: int = ANSWER_VOCAB_SIZE) -> torch.Tensor:
    qt = torch.as_tensor(query_type, dtype=torch.long)
    if qt.dim() == 0:
        qt = qt.view(1)
    if qt.dim() > 1:
        qt = qt.reshape(-1)
    mask = torch.zeros((int(qt.shape[0]), int(vocab_size)), device=qt.device, dtype=torch.bool)
    for i, qti in enumerate(qt.tolist()):
        lo, hi = answer_range_for_query_type(int(qti))
        mask[i, int(lo) : int(hi)] = True
    return mask


def mask_logits_for_query_type(
    logits: torch.Tensor,
    query_type: torch.Tensor,
    *,
    vocab_size: int = ANSWER_VOCAB_SIZE,
    invalid_fill: float = -1e9,
) -> torch.Tensor:
    if logits.dim() != 3:
        raise ValueError(f"expected logits with shape (B,N,V), got {tuple(logits.shape)}")
    if int(logits.shape[-1]) != int(vocab_size):
        raise ValueError(f"logits vocab mismatch: got V={int(logits.shape[-1])}, expected {int(vocab_size)}")

    qt = torch.as_tensor(query_type, device=logits.device, dtype=torch.long)
    if qt.dim() == 1:
        qt = qt.unsqueeze(1).expand(int(logits.shape[0]), int(logits.shape[1]))
    elif qt.dim() == 2:
        if tuple(qt.shape) != tuple(logits.shape[:2]):
            raise ValueError(f"query_type shape mismatch: {tuple(qt.shape)} vs logits {tuple(logits.shape)}")
    else:
        raise ValueError(f"expected query_type with shape (B,) or (B,N), got {tuple(qt.shape)}")

    flat_mask = valid_answer_mask_for_query_type(qt.reshape(-1), vocab_size=int(vocab_size))
    mask = flat_mask.view(int(logits.shape[0]), int(logits.shape[1]), int(vocab_size))
    fill = torch.full_like(logits, float(invalid_fill))
    return torch.where(mask, logits, fill)


def _quantize_scalar(x: np.ndarray, spec: ScalarBinning) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    lo, hi = float(spec.vmin), float(spec.vmax)
    if spec.log:
        x = np.clip(x, lo, hi)
        x = np.log(np.clip(x, 1e-8, None))
        lo = math.log(max(lo, 1e-8))
        hi = math.log(max(hi, 1e-8))
    else:
        x = np.clip(x, lo, hi)
    denom = max(hi - lo, 1e-8)
    q = ((x - lo) / denom) * float(spec.n_bins)
    idx = np.floor(q).astype(np.int64)
    return np.clip(idx, 0, int(spec.n_bins) - 1)


def _decode_scalar_indices(idx: np.ndarray, spec: ScalarBinning) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    idx = np.clip(idx, 0, int(spec.n_bins) - 1)
    frac = (idx.astype(np.float32) + np.float32(0.5)) / np.float32(max(int(spec.n_bins), 1))
    lo = float(spec.vmin)
    hi = float(spec.vmax)
    if spec.log:
        lo_log = math.log(max(lo, 1e-8))
        hi_log = math.log(max(hi, 1e-8))
        x = lo_log + frac * np.float32(hi_log - lo_log)
        return np.exp(x).astype(np.float32, copy=False)
    return (np.float32(lo) + frac * np.float32(hi - lo)).astype(np.float32, copy=False)


def quantize_normals_to_vocab(normals: np.ndarray) -> np.ndarray:
    n = np.asarray(normals, dtype=np.float32)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)
    dots = n @ _NORMAL_DIRS.T
    idx = np.argmax(dots, axis=1).astype(np.int64)
    off, _ = ANSWER_RANGES["mesh_normal"]
    return idx + int(off)


def quantize_normals_unsigned_to_vocab(normals: np.ndarray) -> np.ndarray:
    n = canonicalize_normals_unsigned(normals)
    dots = n @ _NORMAL_UNSIGNED_DIRS.T
    idx = np.argmax(dots, axis=1).astype(np.int64)
    off, _ = ANSWER_RANGES["mesh_normal"]
    return idx + int(off)


def quantize_visibility_to_vocab(vis_sig: np.ndarray) -> np.ndarray:
    """Bit-pack an 8-d visibility signature into the reserved 256-code range."""
    v = np.asarray(vis_sig, dtype=np.float32)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    if v.shape[1] > 8:
        v = v[:, :8]
    elif v.shape[1] < 8:
        pad = np.zeros((v.shape[0], 8 - v.shape[1]), dtype=np.float32)
        v = np.concatenate([v, pad], axis=1)
    bits = (v > 0.5).astype(np.int64)
    weights = (1 << np.arange(8, dtype=np.int64)).reshape(1, 8)
    code = np.sum(bits * weights, axis=1).astype(np.int64)
    off, _ = ANSWER_RANGES["mesh_visibility"]
    return code + int(off)


def quantize_curvature_to_vocab(curv: np.ndarray) -> np.ndarray:
    idx = _quantize_scalar(curv, _CURV_SPEC)
    off, _ = ANSWER_RANGES["mesh_curvature"]
    return idx + int(off)


def quantize_thickness_to_vocab(thickness: np.ndarray) -> np.ndarray:
    idx = _quantize_scalar(thickness, _THICK_SPEC)
    off, _ = ANSWER_RANGES["udf_thickness"]
    return idx + int(off)


def quantize_clearance_to_vocab(clearance: np.ndarray) -> np.ndarray:
    idx = _quantize_scalar(clearance, _CLEAR_SPEC)
    off, _ = ANSWER_RANGES["udf_clearance"]
    return idx + int(off)


def quantize_distance_to_vocab(dist: np.ndarray) -> np.ndarray:
    idx = _quantize_scalar(dist, _DIST_SPEC)
    off, _ = ANSWER_RANGES["udf_distance"]
    return idx + int(off)


def decode_distance_from_vocab(code: np.ndarray) -> np.ndarray:
    code = np.asarray(code, dtype=np.int64)
    off, hi = ANSWER_RANGES["udf_distance"]
    idx = np.clip(code.reshape(-1) - int(off), 0, int(hi - off) - 1)
    out = _decode_scalar_indices(idx, _DIST_SPEC)
    return out.reshape(code.shape).astype(np.float32, copy=False)


def encode_answers_from_fields(query_type: int, fields: Dict[str, np.ndarray]) -> np.ndarray:
    qt = int(query_type)
    if qt == ASK_NORMAL:
        return quantize_normals_to_vocab(fields["normal"])
    if qt == ASK_VISIBILITY:
        return quantize_visibility_to_vocab(fields["visibility"])
    if qt == ASK_CURVATURE:
        return quantize_curvature_to_vocab(fields["curvature"])
    if qt == ASK_THICKNESS:
        return quantize_thickness_to_vocab(fields["thickness"])
    if qt == ASK_CLEARANCE:
        return quantize_clearance_to_vocab(fields["clearance"])
    if qt == ASK_DISTANCE:
        return quantize_distance_to_vocab(fields["distance"])
    raise KeyError(f"unsupported query_type={query_type}")
