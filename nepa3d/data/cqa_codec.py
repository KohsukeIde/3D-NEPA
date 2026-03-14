"""Fixed discrete answer-language codec for explicit-query CQA.

The vocabulary is intentionally versioned and fixed in code so checkpoints,
evaluation, and visualization stay compatible across runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

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


_NORMAL_DIRS = _fibonacci_dirs(ANSWER_RANGES["mesh_normal"][1] - ANSWER_RANGES["mesh_normal"][0])
_CURV_SPEC = ScalarBinning(n_bins=64, vmin=-0.5, vmax=0.5, log=False)
_THICK_SPEC = ScalarBinning(n_bins=64, vmin=1e-3, vmax=1.0, log=True)
_CLEAR_SPEC = ScalarBinning(n_bins=64, vmin=1e-3, vmax=1.0, log=True)
_DIST_SPEC = ScalarBinning(n_bins=64, vmin=1e-3, vmax=1.0, log=True)


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


def quantize_normals_to_vocab(normals: np.ndarray) -> np.ndarray:
    n = np.asarray(normals, dtype=np.float32)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)
    dots = n @ _NORMAL_DIRS.T
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
