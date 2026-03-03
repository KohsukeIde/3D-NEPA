"""V2 dataset loader for PatchNEPA token-based pretraining.

This loader targets the **v2** NPZ format where we explicitly separate:

- **surface/context points** (e.g. `surf_xyz`, `pc_xyz`) used to build PatchNEPA patches
- **query points** (e.g. `udf_qry_xyz`, `mesh_qry_xyz`) where primitive-native supervision is defined

Optional Point-MAE compatibility controls are provided for train-time preprocessing:
- per-iteration random subsampling in train mode,
- Point-MAE style `pc_norm` normalization,
- Point-MAE style scale+translate augmentation.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .answer_feature_pack import FEATURE_DIMS, V2AnswerFeaturePacker


def _np_to_torch_f32(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32))


def _choice(n: int, k: int, rng: Any, replace: bool = False) -> np.ndarray:
    if k >= n:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=k, replace=replace).astype(np.int64)


def _infer_prefix_from_xyz_key(qry_xyz_key: str) -> str:
    if qry_xyz_key.endswith("_xyz"):
        return qry_xyz_key[: -len("_xyz")]
    return qry_xyz_key


def _schema_slices(schema: Sequence[str]) -> List[Tuple[str, slice]]:
    out: List[Tuple[str, slice]] = []
    off = 0
    for name in schema:
        d = int(FEATURE_DIMS.get(str(name), 1))
        out.append((str(name), slice(off, off + d)))
        off += d
    return out


def _pc_norm_np(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Point-MAE style per-shape normalization."""
    centroid = np.mean(xyz, axis=0, keepdims=True).astype(np.float32, copy=False)
    out = xyz - centroid
    radius = float(np.max(np.linalg.norm(out, axis=1)))
    if radius < 1e-8:
        radius = 1.0
    out = out / radius
    return out.astype(np.float32, copy=False), centroid, float(radius)


def _transform_vec_diag(vec: np.ndarray, scales_xyz: np.ndarray) -> np.ndarray:
    """Transform normal/gradient-like vectors under diagonal scaling."""
    inv_sc = 1.0 / np.clip(scales_xyz.reshape(1, 3), 1e-8, None)
    out = vec * inv_sc
    nrm = np.linalg.norm(out, axis=1, keepdims=True)
    out = out / np.clip(nrm, 1e-8, None)
    return out.astype(np.float32, copy=False)


def _apply_answer_scale_rules(
    ans_feat: np.ndarray,
    schema_slices: Sequence[Tuple[str, slice]],
    *,
    norm_radius: Optional[float] = None,
    aug_scales_xyz: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Best-effort answer-feature transform under xyz normalization/augmentation."""
    out = np.asarray(ans_feat, dtype=np.float32).copy()

    if norm_radius is not None and float(norm_radius) > 1e-8:
        r = float(norm_radius)
        for name, sl in schema_slices:
            if name in {"dist", "udf"}:
                out[:, sl] = out[:, sl] / r
            elif name in {"curv", "curvature"}:
                out[:, sl] = out[:, sl] * r
            elif name in {"density"}:
                out[:, sl] = out[:, sl] * r

    if aug_scales_xyz is not None:
        sc = np.asarray(aug_scales_xyz, dtype=np.float32).reshape(3)
        mean_sc = float(np.mean(np.abs(sc)))
        if mean_sc < 1e-8:
            mean_sc = 1.0
        det_sc = float(np.abs(np.prod(sc)))
        if det_sc < 1e-8:
            det_sc = 1.0

        for name, sl in schema_slices:
            if name in {"dist", "udf"}:
                out[:, sl] = out[:, sl] * mean_sc
            elif name in {"curv", "curvature"}:
                out[:, sl] = out[:, sl] / mean_sc
            elif name in {"density"}:
                out[:, sl] = out[:, sl] / det_sc
            elif name in {"n", "normal", "grad", "grad_udf"}:
                out[:, sl] = _transform_vec_diag(out[:, sl], sc)

    return out.astype(np.float32, copy=False)


class V2SurfaceQueryDataset(Dataset):
    """Load v2 NPZ files and emit a unified dict batch item."""

    def __init__(
        self,
        paths: Sequence[str],
        *,
        n_surf: int,
        n_qry: Optional[int] = None,
        n_ray: Optional[int] = None,
        answer_schema: Optional[Sequence[str]] = None,
        seed: int = 0,
        return_qry: bool = True,
        return_rays: bool = False,
        return_pt_ans: bool = False,
        surf_xyz_key: str = "surf_xyz",
        qry_xyz_key: str = "qry_xyz",
        answer_prefix: Optional[str] = None,
        pt_answer_prefix: Optional[str] = None,
        pt_answer_key: Optional[str] = None,
        mode: str = "train",
        pointmae_pc_norm: bool = False,
        pointmae_scale_translate: bool = False,
        pointmae_scale_low: float = (2.0 / 3.0),
        pointmae_scale_high: float = (3.0 / 2.0),
        pointmae_translate: float = 0.2,
        transform_answers: bool = True,
        primitive_label: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.paths = list(paths)
        self.n_surf = int(n_surf)
        self.n_qry = None if n_qry is None else int(n_qry)
        self.n_ray = None if n_ray is None else int(n_ray)
        self.seed = int(seed)
        self.return_qry = bool(return_qry)
        self.return_rays = bool(return_rays)
        self.return_pt_ans = bool(return_pt_ans)
        self.surf_xyz_key = str(surf_xyz_key)
        self.qry_xyz_key = str(qry_xyz_key)
        self.answer_prefix = (
            str(answer_prefix)
            if answer_prefix is not None
            else _infer_prefix_from_xyz_key(self.qry_xyz_key)
        )
        self.pt_answer_prefix = (
            str(pt_answer_prefix)
            if pt_answer_prefix is not None
            else self.surf_xyz_key
        )
        self.pt_answer_key = None if pt_answer_key is None else str(pt_answer_key)
        self.answer_packer = V2AnswerFeaturePacker(answer_schema)
        self.answer_schema_slices = _schema_slices(self.answer_packer.schema)
        self.mode = str(mode)
        self.pointmae_pc_norm = bool(pointmae_pc_norm)
        self.pointmae_scale_translate = bool(pointmae_scale_translate)
        self.pointmae_scale_low = float(pointmae_scale_low)
        self.pointmae_scale_high = float(pointmae_scale_high)
        self.pointmae_translate = float(pointmae_translate)
        self.transform_answers = bool(transform_answers)
        self.primitive_label = None if primitive_label is None else str(primitive_label)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.paths[idx]
        rng = np.random if self.mode == "train" else np.random.RandomState(self.seed + idx)

        with np.load(path, allow_pickle=False) as npz:
            if self.surf_xyz_key not in npz:
                raise KeyError(f"v2 NPZ missing {self.surf_xyz_key}: {path}")
            surf_xyz_all = np.asarray(npz[self.surf_xyz_key], dtype=np.float32)
            n_surf_all = int(surf_xyz_all.shape[0])
            surf_idx = _choice(n_surf_all, self.n_surf, rng)
            surf_xyz = surf_xyz_all[surf_idx]

            prim = self.primitive_label
            if prim is None and ("primitive" in npz):
                prim_val = npz["primitive"]
                if isinstance(prim_val, np.ndarray) and prim_val.shape == ():
                    prim_val = prim_val.item()
                prim = str(prim_val)
            if prim is None:
                prim = os.path.basename(os.path.dirname(path))

            qry_xyz_np: Optional[np.ndarray] = None
            ans_feat_np: Optional[np.ndarray] = None
            pt_ans_feat_np: Optional[np.ndarray] = None
            ray_o_np: Optional[np.ndarray] = None
            ray_d_np: Optional[np.ndarray] = None
            ray_t_np: Optional[np.ndarray] = None
            ray_hit_np: Optional[np.ndarray] = None
            ray_n_np: Optional[np.ndarray] = None

            if self.return_pt_ans:
                if self.pt_answer_key is not None:
                    if self.pt_answer_key not in npz:
                        raise KeyError(f"v2 NPZ missing {self.pt_answer_key}: {path}")
                    pt_full = np.asarray(npz[self.pt_answer_key], dtype=np.float32)
                    if int(pt_full.shape[0]) != n_surf_all:
                        raise ValueError(
                            f"pt_answer_key rows mismatch: key={self.pt_answer_key} "
                            f"rows={int(pt_full.shape[0])} surf_rows={n_surf_all} path={path}"
                        )
                else:
                    packed_pt = self.answer_packer.pack(npz, prefix=self.pt_answer_prefix, n_rows=n_surf_all)
                    pt_full = packed_pt.feat
                pt_ans_feat_np = pt_full[surf_idx]

            if self.return_qry:
                if self.qry_xyz_key in npz:
                    qry_xyz = np.asarray(npz[self.qry_xyz_key], dtype=np.float32)
                    n_q_all = int(qry_xyz.shape[0])
                    if self.n_qry is None:
                        qry_idx = np.arange(n_q_all, dtype=np.int64)
                    else:
                        qry_idx = _choice(n_q_all, self.n_qry, rng)
                    qry_xyz_np = qry_xyz[qry_idx]

                    packed = self.answer_packer.pack(npz, prefix=self.answer_prefix, n_rows=n_q_all)
                    ans_feat_np = packed.feat[qry_idx]
                else:
                    if self.n_qry is not None:
                        c = 0
                        for name in self.answer_packer.schema:
                            c += int(FEATURE_DIMS.get(name, 1))
                        qry_xyz_np = np.zeros((self.n_qry, 3), dtype=np.float32)
                        ans_feat_np = np.zeros((self.n_qry, c), dtype=np.float32)

            if self.return_rays:
                if ("ray_o" in npz) and ("ray_d" in npz):
                    ray_o = np.asarray(npz["ray_o"], dtype=np.float32)
                    ray_d = np.asarray(npz["ray_d"], dtype=np.float32)
                    n_r_all = int(ray_o.shape[0])
                    if self.n_ray is None:
                        ray_idx = np.arange(n_r_all, dtype=np.int64)
                    else:
                        ray_idx = _choice(n_r_all, self.n_ray, rng)
                    ray_o_np = ray_o[ray_idx]
                    ray_d_np = ray_d[ray_idx]
                    ray_t_np = np.asarray(npz.get("ray_t", np.zeros((n_r_all, 1))), dtype=np.float32)[ray_idx]
                    ray_hit_np = np.asarray(npz.get("ray_hit", np.zeros((n_r_all, 1))), dtype=np.float32)[ray_idx]
                    ray_n_np = np.asarray(npz.get("ray_n", np.zeros((n_r_all, 3))), dtype=np.float32)[ray_idx]
                else:
                    if self.n_ray is not None:
                        ray_o_np = np.zeros((self.n_ray, 3), dtype=np.float32)
                        ray_d_np = np.zeros((self.n_ray, 3), dtype=np.float32)
                        ray_t_np = np.zeros((self.n_ray, 1), dtype=np.float32)
                        ray_hit_np = np.zeros((self.n_ray, 1), dtype=np.float32)
                        ray_n_np = np.zeros((self.n_ray, 3), dtype=np.float32)

            # Optional Point-MAE style normalization (train/eval).
            if self.pointmae_pc_norm:
                surf_xyz, center, radius = _pc_norm_np(surf_xyz)
                if qry_xyz_np is not None:
                    qry_xyz_np = ((qry_xyz_np - center) / radius).astype(np.float32, copy=False)
                if ray_o_np is not None:
                    ray_o_np = ((ray_o_np - center) / radius).astype(np.float32, copy=False)
                if ray_t_np is not None:
                    ray_t_np = (ray_t_np / radius).astype(np.float32, copy=False)
                if ans_feat_np is not None and self.transform_answers:
                    ans_feat_np = _apply_answer_scale_rules(
                        ans_feat_np,
                        self.answer_schema_slices,
                        norm_radius=radius,
                        aug_scales_xyz=None,
                    )
                if pt_ans_feat_np is not None and self.transform_answers:
                    pt_ans_feat_np = _apply_answer_scale_rules(
                        pt_ans_feat_np,
                        self.answer_schema_slices,
                        norm_radius=radius,
                        aug_scales_xyz=None,
                    )

            # Optional Point-MAE style train-time scale+translate.
            if self.pointmae_scale_translate and self.mode == "train":
                sc = rng.uniform(
                    low=self.pointmae_scale_low,
                    high=self.pointmae_scale_high,
                    size=(1, 3),
                ).astype(np.float32, copy=False)
                sh = rng.uniform(
                    low=-self.pointmae_translate,
                    high=self.pointmae_translate,
                    size=(1, 3),
                ).astype(np.float32, copy=False)

                surf_xyz = (surf_xyz * sc + sh).astype(np.float32, copy=False)
                if qry_xyz_np is not None:
                    qry_xyz_np = (qry_xyz_np * sc + sh).astype(np.float32, copy=False)
                if ray_o_np is not None:
                    ray_o_np = (ray_o_np * sc + sh).astype(np.float32, copy=False)

                if ray_d_np is not None:
                    ray_d_scaled = (ray_d_np * sc).astype(np.float32, copy=False)
                    ray_len = np.linalg.norm(ray_d_scaled, axis=1, keepdims=True)
                    ray_len = np.clip(ray_len, 1e-8, None).astype(np.float32, copy=False)
                    ray_d_np = (ray_d_scaled / ray_len).astype(np.float32, copy=False)
                    if ray_t_np is not None:
                        ray_t_np = (ray_t_np * ray_len).astype(np.float32, copy=False)
                if ray_n_np is not None:
                    ray_n_np = _transform_vec_diag(ray_n_np, sc.reshape(3))

                if ans_feat_np is not None and self.transform_answers:
                    ans_feat_np = _apply_answer_scale_rules(
                        ans_feat_np,
                        self.answer_schema_slices,
                        norm_radius=None,
                        aug_scales_xyz=sc.reshape(3),
                    )
                if pt_ans_feat_np is not None and self.transform_answers:
                    pt_ans_feat_np = _apply_answer_scale_rules(
                        pt_ans_feat_np,
                        self.answer_schema_slices,
                        norm_radius=None,
                        aug_scales_xyz=sc.reshape(3),
                    )

            out: Dict[str, Any] = {
                "surf_xyz": _np_to_torch_f32(surf_xyz),
                "pt_xyz": _np_to_torch_f32(surf_xyz),
                "primitive": prim,
                "path": path,
            }

            if self.return_pt_ans:
                out["pt_ans_feat"] = None if pt_ans_feat_np is None else _np_to_torch_f32(pt_ans_feat_np)

            if self.return_qry:
                if qry_xyz_np is None:
                    out["qry_xyz"] = None
                    out["ans_feat"] = None
                else:
                    out["qry_xyz"] = _np_to_torch_f32(qry_xyz_np)
                    out["ans_feat"] = _np_to_torch_f32(ans_feat_np)

            if self.return_rays:
                if ray_o_np is None:
                    out["ray_o"] = None
                    out["ray_d"] = None
                    out["ray_t"] = None
                    out["ray_hit"] = None
                    out["ray_n"] = None
                else:
                    out["ray_o"] = _np_to_torch_f32(ray_o_np)
                    out["ray_d"] = _np_to_torch_f32(ray_d_np)
                    out["ray_t"] = _np_to_torch_f32(ray_t_np)
                    out["ray_hit"] = _np_to_torch_f32(ray_hit_np)
                    out["ray_n"] = _np_to_torch_f32(ray_n_np)

            return out


def v2_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate for v2 dict batches."""
    out: Dict[str, Any] = {}
    out["surf_xyz"] = torch.stack([b["surf_xyz"] for b in batch], dim=0)
    out["pt_xyz"] = torch.stack([b["pt_xyz"] for b in batch], dim=0)
    out["primitive"] = [b.get("primitive") for b in batch]
    out["path"] = [b.get("path") for b in batch]

    if batch[0].get("pt_ans_feat") is None:
        out["pt_ans_feat"] = None
    else:
        out["pt_ans_feat"] = torch.stack([b["pt_ans_feat"] for b in batch], dim=0)

    if batch[0].get("qry_xyz") is None:
        out["qry_xyz"] = None
        out["ans_feat"] = None
    else:
        out["qry_xyz"] = torch.stack([b["qry_xyz"] for b in batch], dim=0)
        out["ans_feat"] = torch.stack([b["ans_feat"] for b in batch], dim=0)

    if batch[0].get("ray_o") is None:
        out["ray_o"] = None
        out["ray_d"] = None
        out["ray_t"] = None
        out["ray_hit"] = None
        out["ray_n"] = None
    else:
        out["ray_o"] = torch.stack([b["ray_o"] for b in batch], dim=0)
        out["ray_d"] = torch.stack([b["ray_d"] for b in batch], dim=0)
        out["ray_t"] = torch.stack([b["ray_t"] for b in batch], dim=0)
        out["ray_hit"] = torch.stack([b["ray_hit"] for b in batch], dim=0)
        out["ray_n"] = torch.stack([b["ray_n"] for b in batch], dim=0)

    return out
