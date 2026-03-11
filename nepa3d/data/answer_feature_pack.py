"""Utilities to assemble rich Answer feature vectors (ans_feat) from v2 NPZ files.

Motivation
----------
We want to store *rich* primitive-native signals in v2 NPZs (normals, curvature proxies,
UDF distance/gradient, density, ...), and decide later (by config) which subset to
feed as `ans_feat` into PatchNEPA's Answer embedding.

Key design points:
- The model's Answer embedding expects a *fixed* feature dimension across samples.
  Therefore this packer concatenates features in a fixed schema order and fills
  missing fields with zeros.
- Naming convention: we prefer `surf_*` and `qry_*` prefixes. For backward
  compatibility, we support common aliases (e.g., `qry_dist_udf`).

This module is intentionally lightweight and numpy-only; torch conversion is done
in the Dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------
# Feature schema / dimensions
# -----------------------------

FEATURE_DIMS: Dict[str, int] = {
    # surface-like
    "n": 3,
    "normal": 3,
    "curv": 1,
    "curvature": 1,
    # udf-like
    "dist": 1,
    "udf": 1,
    "grad": 3,
    "grad_udf": 3,
    "occ": 1,
    "near": 1,
    # surface-anchored strict UDF clearance features
    "t_in": 1,
    "t_out": 1,
    "hit_out": 1,
    "thickness": 1,
    "clear_front": 1,
    "clear_back": 1,
    # fixed dims for the current world-package defaults:
    # - probe bank uses 3 deltas by default
    # - visibility signature uses 8 directions by default
    "probe_front": 3,
    "probe_back": 3,
    "probe_thickness": 3,
    "vis": 8,
    "vis_sig": 8,
    "viscount": 1,
    "ao": 1,
    # pointcloud-like
    "density": 1,
}

# Aliases to accept legacy or alternative key names in NPZ.
# Each value is a list of *suffixes* (without prefix) to try.
FEATURE_KEY_ALIASES: Dict[str, List[str]] = {
    "n": ["n", "normal", "nrm"],
    "curv": ["curv", "curvature", "kappa"],
    "dist": ["dist", "dist_udf", "udf", "udf_dist"],
    "grad": ["grad", "grad_udf", "udf_grad"],
    "density": ["density", "dens"],
    "occ": ["occ", "occupancy"],
    "near": ["near", "near_surface"],
    "t_in": ["t_in", "tin", "clear_in", "probe_in"],
    "t_out": ["t_out", "tout", "clear_out", "probe_out"],
    "hit_out": ["hit_out", "hout"],
    "thickness": ["thickness", "thick"],
    "clear_front": ["clear_front", "clear_fwd", "clearance_front"],
    "clear_back": ["clear_back", "clear_bwd", "clearance_back"],
    "probe_front": ["probe_front", "probe_fwd", "offset_front"],
    "probe_back": ["probe_back", "probe_bwd", "offset_back"],
    "probe_thickness": ["probe_thickness", "probe_thick", "offset_thickness"],
    "vis": ["vis", "visibility", "vis_sig", "visibility_sig"],
    "vis_sig": ["vis_sig", "visibility_sig"],
    "viscount": ["viscount", "viewcount", "vis_count"],
    "ao": ["ao", "ambient_occlusion"],
}


def parse_schema(schema: Optional[Sequence[str]]) -> List[str]:
    """Normalize schema input.

    Accepts:
    - None: returns default schema
    - list/tuple of strings
    - single comma-separated string
    """
    if schema is None:
        return ["n", "curv", "dist", "grad", "density"]
    if isinstance(schema, str):
        parts = [p.strip() for p in schema.split(",")]
        return [p for p in parts if p]
    return [str(s).strip() for s in schema if str(s).strip()]


def _resolve_key(npz: "np.lib.npyio.NpzFile", prefix: str, feat_name: str) -> Optional[str]:
    """Resolve a concrete NPZ key for a feature name.

    Tries canonical `f"{prefix}_{suffix}"` for suffixes in FEATURE_KEY_ALIASES.
    Returns the first existing key or None.
    """
    base = feat_name.strip()
    suffixes = FEATURE_KEY_ALIASES.get(base, [base])
    for suf in suffixes:
        k = f"{prefix}_{suf}"
        if k in npz:
            return k
    return None


def _as_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    # Accept [N,1,1] etc by flattening trailing dims.
    return arr.reshape(arr.shape[0], -1)


def _select_bank(arr: np.ndarray, bank_idx: Optional[int]) -> np.ndarray:
    arr = np.asarray(arr)
    if bank_idx is None:
        return arr
    if arr.ndim >= 3:
        if bank_idx < 0 or bank_idx >= arr.shape[0]:
            raise IndexError(f"bank_idx={bank_idx} out of range for shape {arr.shape}")
        return arr[int(bank_idx)]
    return arr




@dataclass
class PackedFeatures:
    feat: np.ndarray  # [N, C]
    schema: List[str]
    dims: List[int]
    keys_used: List[Optional[str]]


class V2AnswerFeaturePacker:
    """Build a concatenated Answer feature matrix from a v2 NPZ."""

    def __init__(self, schema: Optional[Sequence[str]] = None, *, fill_value: float = 0.0):
        self.schema = parse_schema(schema)
        self.fill_value = float(fill_value)

    def pack(self, npz: "np.lib.npyio.NpzFile", *, prefix: str, n_rows: Optional[int] = None, bank_idx: Optional[int] = None) -> PackedFeatures:
        """Pack features under the given prefix ('surf' or 'qry').

        Args:
            npz: opened NpzFile
            prefix: 'surf' or 'qry'
            n_rows: optional row count override; if None, inferred from `{prefix}_xyz`
        """
        if n_rows is None:
            xyz_key = f"{prefix}_xyz"
            if xyz_key not in npz:
                raise KeyError(f"V2AnswerFeaturePacker: missing {xyz_key} to infer row count")
            xyz_arr = _select_bank(np.asarray(npz[xyz_key]), bank_idx)
            n_rows = int(np.asarray(xyz_arr).shape[0])

        feats: List[np.ndarray] = []
        dims: List[int] = []
        keys_used: List[Optional[str]] = []

        for name in self.schema:
            key = _resolve_key(npz, prefix, name)
            keys_used.append(key)

            if key is None:
                # Missing field -> zeros
                d = int(FEATURE_DIMS.get(name, 1))
                dims.append(d)
                feats.append(np.full((n_rows, d), self.fill_value, dtype=np.float32))
                continue

            arr = _select_bank(np.asarray(npz[key], dtype=np.float32), bank_idx)
            arr = _as_2d(arr)
            if arr.shape[0] != n_rows:
                raise ValueError(
                    f"V2AnswerFeaturePacker: key {key} first-dim mismatch: {arr.shape[0]} != {n_rows}"
                )
            dims.append(int(arr.shape[1]))
            feats.append(arr)

        feat = np.concatenate(feats, axis=1) if len(feats) > 0 else np.zeros((n_rows, 0), dtype=np.float32)
        return PackedFeatures(feat=feat, schema=list(self.schema), dims=dims, keys_used=keys_used)
