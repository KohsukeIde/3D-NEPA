from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .noise_ops import NoiseSchedule, add_gaussian_noise, normalize_point_cloud


def _read_ply_ascii(path: Path) -> np.ndarray:
    with path.open("r", errors="ignore") as f:
        header = []
        n_vertices = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            header.append(line.strip())
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            if line.strip() == "end_header":
                break
        if n_vertices is None:
            raise ValueError(f"No vertex count in PLY: {path}")
        pts = []
        for _ in range(n_vertices):
            vals = f.readline().split()
            if len(vals) >= 3:
                pts.append([float(vals[0]), float(vals[1]), float(vals[2])])
    return np.asarray(pts, dtype=np.float32)


def load_points(path: Path) -> np.ndarray:
    suf = path.suffix.lower()
    if suf == ".npy":
        arr = np.load(path)
    elif suf == ".npz":
        z = np.load(path)
        for k in ["points", "pc", "data", "pointcloud"]:
            if k in z:
                arr = z[k]
                break
        else:
            arr = z[z.files[0]]
    elif suf == ".ply":
        arr = _read_ply_ascii(path)
    else:
        raise ValueError(f"Unsupported point file: {path}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape[-1] > 3:
        arr = arr[:, :3]
    return arr


def discover_files(root: str | Path, max_shapes: int = 0, split_file: str | None = None) -> List[Path]:
    root = Path(root)
    if split_file:
        paths = []
        for line in Path(split_file).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            p = root / line
            if p.exists():
                paths.append(p)
                continue
            for ext in [".npy", ".npz", ".ply"]:
                p = root / (line + ext)
                if p.exists():
                    paths.append(p)
                    break
        files = paths
    else:
        files = []
        for ext in ["*.npy", "*.npz", "*.ply"]:
            files.extend(root.rglob(ext))
        files = sorted(files)
    if max_shapes and max_shapes > 0:
        files = files[:max_shapes]
    if not files:
        raise RuntimeError(f"No point files discovered under {root}")
    return files


class NoisePairDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        npoints: int = 1024,
        max_shapes: int = 0,
        split_file: str | None = None,
        num_steps: int = 1000,
        min_gap: int = 50,
        fixed_eval: bool = False,
        seed: int = 0,
        deterministic: bool = False,
    ):
        self.root = Path(root)
        self.files = discover_files(root, max_shapes=max_shapes, split_file=split_file)
        self.npoints = npoints
        self.schedule = NoiseSchedule(num_steps=num_steps)
        self.num_steps = num_steps
        self.min_gap = min_gap
        self.fixed_eval = fixed_eval
        self.seed = int(seed)
        self.deterministic = bool(deterministic)

    def __len__(self):
        return len(self.files)

    def _sample_points(self, pts: np.ndarray, rng: np.random.Generator) -> torch.Tensor:
        if pts.shape[0] >= self.npoints:
            idx = rng.choice(pts.shape[0], self.npoints, replace=False)
        else:
            idx = rng.choice(pts.shape[0], self.npoints, replace=True)
        x = torch.from_numpy(pts[idx].astype(np.float32))
        return normalize_point_cloud(x)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        rng_seed = self.seed * 1_000_003 + idx if (self.fixed_eval or self.deterministic) else None
        rng = np.random.default_rng(rng_seed)
        pts = load_points(path)
        x0 = self._sample_points(pts, rng)
        # Sample on CPU; collate will move to GPU. In fixed/deterministic mode,
        # rng is seeded from idx so eval variants use identical noise-time pairs.
        t_val = rng.integers(max(int(self.num_steps * 0.35), 2 * self.min_gap), self.num_steps)
        s_val = rng.integers(self.min_gap, max(self.min_gap + 1, t_val - self.min_gap))
        r_val = rng.integers(0, max(1, s_val - self.min_gap))
        t = torch.tensor(t_val, dtype=torch.long)
        s = torch.tensor(s_val, dtype=torch.long)
        r = torch.tensor(r_val, dtype=torch.long)
        # Generate independent noise for each level. For fixed eval, use stateless
        # seeds so separate variant processes compare exactly the same tensors.
        if self.fixed_eval or self.deterministic:
            gen_t = torch.Generator().manual_seed(self.seed * 10_000_019 + idx * 31 + 1)
            gen_s = torch.Generator().manual_seed(self.seed * 10_000_019 + idx * 31 + 2)
            gen_r = torch.Generator().manual_seed(self.seed * 10_000_019 + idx * 31 + 3)
        else:
            gen_t = gen_s = gen_r = None
        x_t = add_gaussian_noise(x0, t, self.schedule, generator=gen_t)
        x_s = add_gaussian_noise(x0, s, self.schedule, generator=gen_s)
        x_r = add_gaussian_noise(x0, r, self.schedule, generator=gen_r)
        return {
            "x0": x0.float(),
            "x_t": x_t.float(),
            "x_s": x_s.float(),
            "x_r": x_r.float(),
            "t": t,
            "s": s,
            "r": r,
            "path": str(path),
            "idx": idx,
        }
