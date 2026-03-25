"""Versioned discrete answer-language codec for explicit-query CQA.

The vocabulary is intentionally fixed in code so checkpoints, evaluation, and
visualization stay compatible across runs. ``cqa_v1`` remains the historical
layout. ``cqa_v2`` is a new profile for the current multi-answer CE mainline.
``cqa_v3`` keeps the same task set as v2 but promotes rescued thickness from
64 to 128 quantile bins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Historical defaults / legacy constants
# -----------------------------------------------------------------------------

CQA_VOCAB_VERSION = "cqa_v1"

ASK_NORMAL = 0
ASK_VISIBILITY = 1
ASK_CURVATURE = 2
ASK_THICKNESS = 3
ASK_CLEARANCE = 4
ASK_DISTANCE = 5
ASK_AO = 6

# Keep legacy constants for existing code / checkpoints. Version-aware helpers
# below should be used for new paths.
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
ANSWER_RANGES: Dict[str, Tuple[int, int]] = {
    "mesh_normal": (0, 128),
    "mesh_visibility": (128, 384),
    "mesh_curvature": (384, 448),
    "udf_thickness": (448, 512),
    "udf_clearance": (512, 576),
    "udf_distance": (576, 640),
}
ANSWER_VOCAB_SIZE = 640

# -----------------------------------------------------------------------------
# Version profiles
# -----------------------------------------------------------------------------

_SUPPORTED_CODEC_VERSIONS = {"cqa_v1", "cqa_v2", "cqa_v3"}

_QUERY_TYPE_NAMES_V1 = dict(QUERY_TYPE_NAMES)
_QUERY_NAME_TO_ID_V1 = {
    **QUERY_NAME_TO_ID,
    "mesh_normal_unsigned": ASK_NORMAL,
    "mesh_viscount": ASK_VISIBILITY,
    "udf_thickness_valid_qbin": ASK_THICKNESS,
}
_ANSWER_RANGES_V1: Dict[str, Tuple[int, int]] = {
    **ANSWER_RANGES,
    "mesh_normal_unsigned": ANSWER_RANGES["mesh_normal"],
    "mesh_viscount": ANSWER_RANGES["mesh_visibility"],
    "udf_thickness_valid_qbin": ANSWER_RANGES["udf_thickness"],
}

# cqa_v2 keeps the audited 64-bin thickness rescue as-is, while increasing the
# resolution of the mainline distance / unsigned-normal branches and adding AO.
_QUERY_TYPE_NAMES_V2 = {
    ASK_NORMAL: "mesh_normal_unsigned",
    ASK_THICKNESS: "udf_thickness_valid_qbin",
    ASK_DISTANCE: "udf_distance",
    ASK_AO: "mesh_ao",
}
_QUERY_NAME_TO_ID_V2 = {
    **{v: k for k, v in _QUERY_TYPE_NAMES_V2.items()},
    "udf_thickness": ASK_THICKNESS,
}
_ANSWER_RANGES_V2: Dict[str, Tuple[int, int]] = {
    "mesh_normal_unsigned": (0, 256),
    "mesh_ao": (256, 384),
    "udf_thickness_valid_qbin": (384, 448),
    "udf_thickness": (384, 448),
    "udf_distance": (448, 704),
}

_QUERY_TYPE_NAMES_V3 = dict(_QUERY_TYPE_NAMES_V2)
_QUERY_NAME_TO_ID_V3 = dict(_QUERY_NAME_TO_ID_V2)
_ANSWER_RANGES_V3: Dict[str, Tuple[int, int]] = {
    "mesh_normal_unsigned": (0, 256),
    "mesh_ao": (256, 384),
    "udf_thickness_valid_qbin": (384, 512),
    "udf_thickness": (384, 512),
    "udf_distance": (512, 768),
}


@dataclass(frozen=True)
class ScalarBinning:
    n_bins: int
    vmin: float
    vmax: float
    log: bool = False


# -----------------------------------------------------------------------------
# Common helpers
# -----------------------------------------------------------------------------


def _require_codec_version(codec_version: str) -> str:
    v = str(codec_version or CQA_VOCAB_VERSION)
    if v not in _SUPPORTED_CODEC_VERSIONS:
        raise KeyError(f"unsupported codec_version={codec_version}")
    return v


def query_type_names(codec_version: str = CQA_VOCAB_VERSION) -> Dict[int, str]:
    v = _require_codec_version(codec_version)
    if v == "cqa_v1":
        return dict(_QUERY_TYPE_NAMES_V1)
    if v == "cqa_v2":
        return dict(_QUERY_TYPE_NAMES_V2)
    return dict(_QUERY_TYPE_NAMES_V3)


def query_name_to_id(codec_version: str = CQA_VOCAB_VERSION) -> Dict[str, int]:
    v = _require_codec_version(codec_version)
    if v == "cqa_v1":
        return dict(_QUERY_NAME_TO_ID_V1)
    if v == "cqa_v2":
        return dict(_QUERY_NAME_TO_ID_V2)
    return dict(_QUERY_NAME_TO_ID_V3)


def answer_ranges(codec_version: str = CQA_VOCAB_VERSION) -> Dict[str, Tuple[int, int]]:
    v = _require_codec_version(codec_version)
    if v == "cqa_v1":
        return dict(_ANSWER_RANGES_V1)
    if v == "cqa_v2":
        return dict(_ANSWER_RANGES_V2)
    return dict(_ANSWER_RANGES_V3)


def query_type_id_list(codec_version: str = CQA_VOCAB_VERSION) -> list[int]:
    return sorted(int(k) for k in query_type_names(codec_version).keys())


def query_type_vocab_size(codec_version: str = CQA_VOCAB_VERSION) -> int:
    ids = query_type_id_list(codec_version)
    return int(max(ids) + 1 if ids else 0)


def answer_vocab_size(codec_version: str = CQA_VOCAB_VERSION) -> int:
    ranges = answer_ranges(codec_version)
    return int(max(hi for _lo, hi in ranges.values()))


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


_NORMAL_DIRS_V1 = _fibonacci_dirs(ANSWER_RANGES["mesh_normal"][1] - ANSWER_RANGES["mesh_normal"][0])
_NORMAL_UNSIGNED_DIRS_V1 = _hemisphere_dirs(ANSWER_RANGES["mesh_normal"][1] - ANSWER_RANGES["mesh_normal"][0])
_NORMAL_UNSIGNED_DIRS_V2 = _hemisphere_dirs(_ANSWER_RANGES_V2["mesh_normal_unsigned"][1] - _ANSWER_RANGES_V2["mesh_normal_unsigned"][0])
_NORMAL_UNSIGNED_DIRS_V3 = _hemisphere_dirs(_ANSWER_RANGES_V3["mesh_normal_unsigned"][1] - _ANSWER_RANGES_V3["mesh_normal_unsigned"][0])

# Keep the historical names for legacy utilities such as type-switch export.
_NORMAL_DIRS = _NORMAL_DIRS_V1
_NORMAL_UNSIGNED_DIRS = _NORMAL_UNSIGNED_DIRS_V1

_CURV_SPEC = ScalarBinning(n_bins=64, vmin=-0.5, vmax=0.5, log=False)
_THICK_SPEC = ScalarBinning(n_bins=64, vmin=1e-3, vmax=1.0, log=True)
_CLEAR_SPEC = ScalarBinning(n_bins=64, vmin=1e-3, vmax=1.0, log=True)
_DIST_SPEC = ScalarBinning(n_bins=64, vmin=1e-3, vmax=1.0, log=True)
_DIST_SPEC_V2 = ScalarBinning(n_bins=256, vmin=1e-3, vmax=1.0, log=True)

_THICK_VALID_QBIN_EDGES = np.asarray(
    [
        2.001892e-04,
        8.393233e-03,
        1.180632e-02,
        1.504406e-02,
        2.287071e-02,
        2.869686e-02,
        2.971412e-02,
        3.019239e-02,
        3.046930e-02,
        3.067061e-02,
        3.172112e-02,
        4.372009e-02,
        4.553617e-02,
        4.603876e-02,
        4.628205e-02,
        4.642287e-02,
        4.828371e-02,
        6.025374e-02,
        6.155672e-02,
        6.192845e-02,
        6.219119e-02,
        7.502553e-02,
        7.719006e-02,
        7.759085e-02,
        7.940852e-02,
        9.233747e-02,
        9.317222e-02,
        9.578139e-02,
        1.082942e-01,
        1.089104e-01,
        1.221686e-01,
        1.244741e-01,
        1.370316e-01,
        1.400667e-01,
        1.511735e-01,
        1.557561e-01,
        1.706998e-01,
        1.779892e-01,
        1.870876e-01,
        2.025662e-01,
        2.181361e-01,
        2.338913e-01,
        2.562042e-01,
        2.785491e-01,
        2.962603e-01,
        3.166968e-01,
        3.432651e-01,
        3.744523e-01,
        4.046624e-01,
        4.347535e-01,
        4.660084e-01,
        4.992469e-01,
        5.404418e-01,
        5.776073e-01,
        6.240864e-01,
        6.697433e-01,
        7.037844e-01,
        7.647385e-01,
        8.241761e-01,
        8.958294e-01,
        9.890134e-01,
        1.092821e00,
        1.249365e00,
        1.475450e00,
        2.451169e00,
    ],
    dtype=np.float32,
)
_THICK_VALID_QBIN128_EDGES = np.asarray(
    [
        4.021325e-04,
        4.558492e-03,
        6.060358e-03,
        7.241583e-03,
        8.216125e-03,
        9.120226e-03,
        9.949663e-03,
        1.079437e-02,
        1.158885e-02,
        1.234929e-02,
        1.309567e-02,
        1.389801e-02,
        1.474561e-02,
        1.579792e-02,
        1.738603e-02,
        1.993513e-02,
        2.360811e-02,
        2.648791e-02,
        2.782968e-02,
        2.853952e-02,
        2.904814e-02,
        2.942120e-02,
        2.969665e-02,
        2.991809e-02,
        3.009855e-02,
        3.024453e-02,
        3.036252e-02,
        3.046403e-02,
        3.055506e-02,
        3.064263e-02,
        3.072882e-02,
        3.084526e-02,
        3.109938e-02,
        3.188213e-02,
        3.412814e-02,
        3.828637e-02,
        4.281968e-02,
        4.455600e-02,
        4.525267e-02,
        4.564634e-02,
        4.591155e-02,
        4.610870e-02,
        4.624873e-02,
        4.635796e-02,
        4.646849e-02,
        4.668411e-02,
        4.717572e-02,
        4.896954e-02,
        5.265784e-02,
        5.817359e-02,
        6.056018e-02,
        6.133099e-02,
        6.169835e-02,
        6.192733e-02,
        6.208714e-02,
        6.263406e-02,
        6.463882e-02,
        6.872021e-02,
        7.454013e-02,
        7.667884e-02,
        7.732449e-02,
        7.763962e-02,
        7.823356e-02,
        8.040971e-02,
        8.505759e-02,
        9.145039e-02,
        9.280278e-02,
        9.322483e-02,
        9.386696e-02,
        9.651419e-02,
        1.016586e-01,
        1.076213e-01,
        1.086135e-01,
        1.089848e-01,
        1.105780e-01,
        1.155176e-01,
        1.216744e-01,
        1.240561e-01,
        1.246027e-01,
        1.257985e-01,
        1.296374e-01,
        1.345428e-01,
        1.388498e-01,
        1.400023e-01,
        1.407656e-01,
        1.424833e-01,
        1.460871e-01,
        1.501950e-01,
        1.537081e-01,
        1.553624e-01,
        1.559743e-01,
        1.570019e-01,
        1.586779e-01,
        1.622103e-01,
        1.664217e-01,
        1.694596e-01,
        1.709857e-01,
        1.715426e-01,
        1.722866e-01,
        1.731450e-01,
        1.749666e-01,
        1.775239e-01,
        1.811913e-01,
        1.850376e-01,
        1.867594e-01,
        1.875125e-01,
        1.893061e-01,
        1.925588e-01,
        1.973159e-01,
        2.018841e-01,
        2.026665e-01,
        2.045015e-01,
        2.091944e-01,
        2.162183e-01,
        2.185643e-01,
        2.217163e-01,
        2.312028e-01,
        2.338154e-01,
        2.377729e-01,
        2.494719e-01,
        2.637297e-01,
        2.798127e-01,
        3.048303e-01,
        3.359771e-01,
        3.747080e-01,
        4.206318e-01,
        4.874778e-01,
        6.573749e-01,
        2.015502e00,
    ],
    dtype=np.float32,
)
_VISCOUNT_MAX = 8
# AO is currently derived from 8-ray visibility, so the raw support is already
# discrete at multiples of 1/8. Keep the v2 range large enough for future
# densification, but map the current support levels cleanly.
_AO_LEVELS = np.asarray([i / 8.0 for i in range(9)], dtype=np.float32)
_AO_CODEBOOK_V2 = np.rint(np.linspace(0, 127, _AO_LEVELS.size)).astype(np.int64)


def answer_range_for_query_name(query_name: str, codec_version: str = CQA_VOCAB_VERSION) -> Tuple[int, int]:
    ranges = answer_ranges(codec_version)
    key = str(query_name)
    if key not in ranges:
        raise KeyError(f"unknown query_name={query_name} for codec_version={codec_version}")
    return ranges[key]


def answer_range_for_query_type(query_type: int, codec_version: str = CQA_VOCAB_VERSION) -> Tuple[int, int]:
    qt = int(query_type)
    qtypes = query_type_names(codec_version)
    if qt not in qtypes:
        raise KeyError(f"unknown query_type={qt} for codec_version={codec_version}")
    return answer_range_for_query_name(qtypes[qt], codec_version=codec_version)


def valid_answer_mask_for_query_type(
    query_type: torch.Tensor,
    *,
    codec_version: str = CQA_VOCAB_VERSION,
    vocab_size: int | None = None,
) -> torch.Tensor:
    qt = torch.as_tensor(query_type, dtype=torch.long)
    if qt.dim() == 0:
        qt = qt.view(1)
    if qt.dim() > 1:
        qt = qt.reshape(-1)
    resolved_vocab = int(answer_vocab_size(codec_version) if vocab_size is None else vocab_size)
    mask = torch.zeros((int(qt.shape[0]), resolved_vocab), device=qt.device, dtype=torch.bool)
    for i, qti in enumerate(qt.tolist()):
        lo, hi = answer_range_for_query_type(int(qti), codec_version=codec_version)
        mask[i, int(lo) : int(hi)] = True
    return mask


def mask_logits_for_query_type(
    logits: torch.Tensor,
    query_type: torch.Tensor,
    *,
    codec_version: str = CQA_VOCAB_VERSION,
    vocab_size: int | None = None,
    invalid_fill: float = -1e9,
) -> torch.Tensor:
    if logits.dim() != 3:
        raise ValueError(f"expected logits with shape (B,N,V), got {tuple(logits.shape)}")
    resolved_vocab = int(answer_vocab_size(codec_version) if vocab_size is None else vocab_size)
    if int(logits.shape[-1]) != resolved_vocab:
        raise ValueError(f"logits vocab mismatch: got V={int(logits.shape[-1])}, expected {resolved_vocab}")

    qt = torch.as_tensor(query_type, device=logits.device, dtype=torch.long)
    if qt.dim() == 1:
        qt = qt.unsqueeze(1).expand(int(logits.shape[0]), int(logits.shape[1]))
    elif qt.dim() == 2:
        if tuple(qt.shape) != tuple(logits.shape[:2]):
            raise ValueError(f"query_type shape mismatch: {tuple(qt.shape)} vs logits {tuple(logits.shape)}")
    else:
        raise ValueError(f"expected query_type with shape (B,) or (B,N), got {tuple(qt.shape)}")

    flat_mask = valid_answer_mask_for_query_type(qt.reshape(-1), codec_version=codec_version, vocab_size=resolved_vocab)
    mask = flat_mask.view(int(logits.shape[0]), int(logits.shape[1]), resolved_vocab)
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


def _quantize_scalar_by_edges(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32).reshape(-1)
    ed = np.asarray(edges, dtype=np.float32).reshape(-1)
    if ed.size < 2:
        raise ValueError("bin edges must contain at least two values")
    inner = ed[1:-1]
    idx = np.digitize(values, inner, right=False).astype(np.int64)
    return np.clip(idx, 0, int(ed.size) - 2)


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


def quantize_normals_to_vocab(normals: np.ndarray, *, codec_version: str = CQA_VOCAB_VERSION) -> np.ndarray:
    if _require_codec_version(codec_version) != "cqa_v1":
        raise KeyError("signed mesh_normal is only supported in cqa_v1")
    n = np.asarray(normals, dtype=np.float32)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)
    dots = n @ _NORMAL_DIRS_V1.T
    idx = np.argmax(dots, axis=1).astype(np.int64)
    off, _ = answer_range_for_query_name("mesh_normal", codec_version=codec_version)
    return idx + int(off)


def quantize_normals_unsigned_to_vocab(normals: np.ndarray, *, codec_version: str = CQA_VOCAB_VERSION) -> np.ndarray:
    n = canonicalize_normals_unsigned(normals)
    v = _require_codec_version(codec_version)
    if v == "cqa_v1":
        dirs = _NORMAL_UNSIGNED_DIRS_V1
        key = "mesh_normal_unsigned"
    elif v == "cqa_v2":
        dirs = _NORMAL_UNSIGNED_DIRS_V2
        key = "mesh_normal_unsigned"
    else:
        dirs = _NORMAL_UNSIGNED_DIRS_V3
        key = "mesh_normal_unsigned"
    dots = n @ dirs.T
    idx = np.argmax(dots, axis=1).astype(np.int64)
    off, _ = answer_range_for_query_name(key, codec_version=codec_version)
    return idx + int(off)


def quantize_visibility_to_vocab(vis_sig: np.ndarray, *, codec_version: str = CQA_VOCAB_VERSION) -> np.ndarray:
    if _require_codec_version(codec_version) != "cqa_v1":
        raise KeyError("mesh_visibility bitpack is only supported in cqa_v1")
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
    off, _ = answer_range_for_query_name("mesh_visibility", codec_version=codec_version)
    return code + int(off)


def quantize_curvature_to_vocab(curv: np.ndarray, *, codec_version: str = CQA_VOCAB_VERSION) -> np.ndarray:
    if _require_codec_version(codec_version) != "cqa_v1":
        raise KeyError("mesh_curvature is only supported in cqa_v1")
    idx = _quantize_scalar(curv, _CURV_SPEC)
    off, _ = answer_range_for_query_name("mesh_curvature", codec_version=codec_version)
    return idx + int(off)


def quantize_thickness_to_vocab(thickness: np.ndarray, *, codec_version: str = CQA_VOCAB_VERSION) -> np.ndarray:
    if _require_codec_version(codec_version) != "cqa_v1":
        raise KeyError("historical udf_thickness is only supported in cqa_v1")
    idx = _quantize_scalar(thickness, _THICK_SPEC)
    off, _ = answer_range_for_query_name("udf_thickness", codec_version=codec_version)
    return idx + int(off)


def quantize_thickness_valid_qbin_to_vocab(thickness: np.ndarray, *, codec_version: str = CQA_VOCAB_VERSION) -> np.ndarray:
    v = _require_codec_version(codec_version)
    edges = _THICK_VALID_QBIN_EDGES if v in {"cqa_v1", "cqa_v2"} else _THICK_VALID_QBIN128_EDGES
    idx = _quantize_scalar_by_edges(thickness, edges)
    off, _ = answer_range_for_query_name("udf_thickness_valid_qbin", codec_version=codec_version)
    return idx + int(off)


def quantize_clearance_to_vocab(clearance: np.ndarray, *, codec_version: str = CQA_VOCAB_VERSION) -> np.ndarray:
    if _require_codec_version(codec_version) != "cqa_v1":
        raise KeyError("udf_clearance is only supported in cqa_v1")
    idx = _quantize_scalar(clearance, _CLEAR_SPEC)
    off, _ = answer_range_for_query_name("udf_clearance", codec_version=codec_version)
    return idx + int(off)


def quantize_distance_to_vocab(dist: np.ndarray, *, codec_version: str = CQA_VOCAB_VERSION) -> np.ndarray:
    spec = _DIST_SPEC if _require_codec_version(codec_version) == "cqa_v1" else _DIST_SPEC_V2
    idx = _quantize_scalar(dist, spec)
    off, _ = answer_range_for_query_name("udf_distance", codec_version=codec_version)
    return idx + int(off)


def quantize_viscount_to_vocab(viscount: np.ndarray, *, codec_version: str = CQA_VOCAB_VERSION) -> np.ndarray:
    if _require_codec_version(codec_version) != "cqa_v1":
        raise KeyError("mesh_viscount is only supported in cqa_v1")
    values = np.asarray(viscount, dtype=np.float32).reshape(-1)
    idx = np.rint(values).astype(np.int64)
    idx = np.clip(idx, 0, _VISCOUNT_MAX)
    off, _ = answer_range_for_query_name("mesh_visibility", codec_version=codec_version)
    return idx + int(off)


def quantize_ao_to_vocab(ao: np.ndarray, *, codec_version: str = CQA_VOCAB_VERSION) -> np.ndarray:
    if _require_codec_version(codec_version) not in {"cqa_v2", "cqa_v3"}:
        raise KeyError("mesh_ao discrete codec is only supported in cqa_v2/cqa_v3")
    values = np.asarray(ao, dtype=np.float32).reshape(-1)
    idx = np.argmin(np.abs(values[:, None] - _AO_LEVELS[None, :]), axis=1).astype(np.int64)
    code = _AO_CODEBOOK_V2[idx]
    off, _ = answer_range_for_query_name("mesh_ao", codec_version=codec_version)
    return code + int(off)


def decode_distance_from_vocab(code: np.ndarray, *, codec_version: str = CQA_VOCAB_VERSION) -> np.ndarray:
    code = np.asarray(code, dtype=np.int64)
    off, hi = answer_range_for_query_name("udf_distance", codec_version=codec_version)
    idx = np.clip(code.reshape(-1) - int(off), 0, int(hi - off) - 1)
    spec = _DIST_SPEC if _require_codec_version(codec_version) == "cqa_v1" else _DIST_SPEC_V2
    out = _decode_scalar_indices(idx, spec)
    return out.reshape(code.shape).astype(np.float32, copy=False)


def encode_answers_from_fields(
    query_name_or_type: str | int,
    fields: Dict[str, np.ndarray],
    *,
    codec_version: str = CQA_VOCAB_VERSION,
    encode_mode: str | None = None,
) -> np.ndarray:
    if encode_mode == "normal_unsigned":
        return quantize_normals_unsigned_to_vocab(fields["normal"], codec_version=codec_version)
    if encode_mode == "mesh_viscount":
        return quantize_viscount_to_vocab(fields["viscount"], codec_version=codec_version)
    if encode_mode == "udf_thickness_valid_qbin":
        return quantize_thickness_valid_qbin_to_vocab(fields["thickness"], codec_version=codec_version)
    if encode_mode == "mesh_ao":
        return quantize_ao_to_vocab(fields["ao"], codec_version=codec_version)

    if isinstance(query_name_or_type, str):
        query_name = str(query_name_or_type)
    else:
        qtypes = query_type_names(codec_version)
        qt = int(query_name_or_type)
        if qt not in qtypes:
            raise KeyError(f"unsupported query_type={qt} for codec_version={codec_version}")
        query_name = qtypes[qt]

    if query_name == "mesh_normal":
        return quantize_normals_to_vocab(fields["normal"], codec_version=codec_version)
    if query_name == "mesh_normal_unsigned":
        return quantize_normals_unsigned_to_vocab(fields["normal"], codec_version=codec_version)
    if query_name == "mesh_visibility":
        return quantize_visibility_to_vocab(fields["visibility"], codec_version=codec_version)
    if query_name == "mesh_curvature":
        return quantize_curvature_to_vocab(fields["curvature"], codec_version=codec_version)
    if query_name == "mesh_ao":
        return quantize_ao_to_vocab(fields["ao"], codec_version=codec_version)
    if query_name == "udf_thickness":
        return quantize_thickness_to_vocab(fields["thickness"], codec_version=codec_version)
    if query_name == "udf_thickness_valid_qbin":
        return quantize_thickness_valid_qbin_to_vocab(fields["thickness"], codec_version=codec_version)
    if query_name == "udf_clearance":
        return quantize_clearance_to_vocab(fields["clearance"], codec_version=codec_version)
    if query_name == "udf_distance":
        return quantize_distance_to_vocab(fields["distance"], codec_version=codec_version)
    raise KeyError(f"unsupported query_name={query_name} for codec_version={codec_version}")
