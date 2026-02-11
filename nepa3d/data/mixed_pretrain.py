import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Sampler

try:
    import yaml
except Exception:
    yaml = None

from .dataset import ModelNet40QueryDataset
from .modelnet40_index import list_npz


@dataclass
class MixDatasetSpec:
    """One component in a mixed pretraining corpus."""
    name: str
    cache_root: str
    split: str = "train"
    backend: str = "mesh"
    weight: float = 1.0

    # Optional per-dataset overrides (fallbacks are passed from pretrain args).
    drop_ray_prob: Optional[float] = None
    force_missing_ray: Optional[bool] = None
    add_eos: Optional[bool] = None
    voxel_grid: Optional[int] = None
    voxel_dilate: Optional[int] = None
    voxel_max_steps: Optional[int] = None

    extra: Dict[str, Any] = field(default_factory=dict)


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(v)


def load_mix_config(path: str) -> Tuple[List[MixDatasetSpec], Dict[str, Any]]:
    """Load YAML mix config.

    Expected schema:
      datasets:
        - name: mn40_mesh
          cache_root: data/modelnet40_cache_v2
          split: train
          backend: mesh
          weight: 0.5
          drop_ray_prob: 0.0   # optional
          force_missing_ray: false  # optional
      mix_num_samples: 200000     # optional
      replacement: true           # optional

    Returns:
      specs: list of MixDatasetSpec
      cfg: raw dict for extra top-level fields (mix_num_samples, etc.)
    """
    if yaml is None:
        raise ImportError("PyYAML is required for --mix_config but is not installed.")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"empty mix config: {path}")
    ds_cfg = cfg.get("datasets", None)
    if not isinstance(ds_cfg, list) or len(ds_cfg) == 0:
        raise ValueError("mix config must have non-empty 'datasets' list")

    specs: List[MixDatasetSpec] = []
    for i, d in enumerate(ds_cfg):
        if not isinstance(d, dict):
            raise ValueError(f"datasets[{i}] must be a dict")
        name = str(d.get("name", f"ds{i}"))
        cache_root = str(d["cache_root"])
        split = str(d.get("split", "train"))
        backend = str(d.get("backend", "mesh"))
        weight = float(d.get("weight", 1.0))
        spec = MixDatasetSpec(
            name=name,
            cache_root=cache_root,
            split=split,
            backend=backend,
            weight=weight,
            drop_ray_prob=d.get("drop_ray_prob", None),
            force_missing_ray=_as_bool(d.get("force_missing_ray", False)) if "force_missing_ray" in d else None,
            add_eos=_as_bool(d.get("add_eos", True)) if "add_eos" in d else None,
            voxel_grid=int(d["voxel_grid"]) if "voxel_grid" in d else None,
            voxel_dilate=int(d["voxel_dilate"]) if "voxel_dilate" in d else None,
            voxel_max_steps=int(d["voxel_max_steps"]) if "voxel_max_steps" in d else None,
            extra={k: v for k, v in d.items() if k not in {
                "name","cache_root","split","backend","weight",
                "drop_ray_prob","force_missing_ray","add_eos",
                "voxel_grid","voxel_dilate","voxel_max_steps"
            }},
        )
        specs.append(spec)

    # normalize weights
    w = np.array([max(0.0, s.weight) for s in specs], dtype=np.float64)
    if not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("invalid weights in mix config")
    w = w / w.sum()
    for s, wi in zip(specs, w.tolist()):
        s.weight = float(wi)

    return specs, cfg


class MixedPretrainDataset(Dataset):
    """A thin wrapper over ConcatDataset, plus metadata.

    Use with MixtureSampler to sample per-dataset with specified weights.
    """

    def __init__(self, datasets: Sequence[Dataset], names: Sequence[str]):
        if len(datasets) != len(names):
            raise ValueError("datasets and names must have the same length")
        self.datasets = list(datasets)
        self.names = list(names)
        self.concat = ConcatDataset(self.datasets)
        self.sizes = [len(d) for d in self.datasets]
        # offsets into concatenated index space
        self.offsets = [0]
        for n in self.sizes[:-1]:
            self.offsets.append(self.offsets[-1] + int(n))

    def __len__(self) -> int:
        return len(self.concat)

    def __getitem__(self, idx: int):
        return self.concat[idx]


class MixtureSampler(Sampler[int]):
    """Sample from a concatenated dataset by first sampling dataset-id, then local index.

    This avoids allocating a weight vector of size sum(len(ds)).
    """

    def __init__(
        self,
        dataset_sizes: Sequence[int],
        dataset_weights: Sequence[float],
        num_samples: int,
        replacement: bool = True,
        seed: int = 0,
    ):
        self.dataset_sizes = [int(x) for x in dataset_sizes]
        self.offsets = [0]
        for n in self.dataset_sizes[:-1]:
            self.offsets.append(self.offsets[-1] + int(n))

        w = np.asarray(dataset_weights, dtype=np.float64)
        if w.ndim != 1 or w.size != len(self.dataset_sizes):
            raise ValueError("dataset_weights must have the same length as dataset_sizes")
        if (w < 0).any() or not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError("invalid dataset_weights")
        self.dataset_weights = (w / w.sum()).astype(np.float64)

        self.num_samples = int(num_samples)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0

        if not self.replacement:
            # Without replacement, require we can draw enough unique samples.
            total = int(sum(self.dataset_sizes))
            if self.num_samples > total:
                raise ValueError("num_samples > total_size with replacement=False")

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.RandomState((self.seed + 1009 * self.epoch) & 0xFFFFFFFF)

        if self.replacement:
            for _ in range(self.num_samples):
                ds_id = int(rng.choice(len(self.dataset_sizes), p=self.dataset_weights))
                local = int(rng.randint(0, self.dataset_sizes[ds_id]))
                yield self.offsets[ds_id] + local
        else:
            # Simple no-replacement: sample global indices uniformly, then map to datasets.
            # (This ignores per-dataset weights; intended only for small debugging runs.)
            total = int(sum(self.dataset_sizes))
            idx = rng.choice(total, size=self.num_samples, replace=False)
            for x in idx.tolist():
                yield int(x)


def build_mixed_pretrain(
    mix_config_path: str,
    n_point: int,
    n_ray: int,
    num_workers: int = 4,
    mode: str = "train",
    eval_seed: int = 0,
    mc_eval_k: int = 1,
    # defaults for dataset overrides:
    drop_ray_prob: float = 0.0,
    force_missing_ray: bool = False,
    add_eos: bool = True,
    voxel_grid: int = 64,
    voxel_dilate: int = 1,
    voxel_max_steps: int = 0,
) -> Tuple[MixedPretrainDataset, MixtureSampler, Dict[str, Any]]:
    """Build (dataset, sampler) for mixed pretraining from a YAML config."""
    specs, cfg = load_mix_config(mix_config_path)

    datasets: List[Dataset] = []
    names: List[str] = []
    weights: List[float] = []
    for s in specs:
        paths = list_npz(s.cache_root, s.split)
        if len(paths) == 0:
            raise FileNotFoundError(f"no npz found: cache_root={s.cache_root} split={s.split}")

        ds = ModelNet40QueryDataset(
            paths,
            backend=s.backend,
            n_point=n_point,
            n_ray=n_ray,
            mode=mode,
            eval_seed=eval_seed,
            mc_eval_k=mc_eval_k,
            drop_ray_prob=float(drop_ray_prob if s.drop_ray_prob is None else s.drop_ray_prob),
            force_missing_ray=bool(force_missing_ray if s.force_missing_ray is None else s.force_missing_ray),
            add_eos=bool(add_eos if s.add_eos is None else s.add_eos),
            voxel_grid=int(voxel_grid if s.voxel_grid is None else s.voxel_grid),
            voxel_dilate=int(voxel_dilate if s.voxel_dilate is None else s.voxel_dilate),
            voxel_max_steps=int(voxel_max_steps if s.voxel_max_steps is None else s.voxel_max_steps),
            return_label=False,
        )
        datasets.append(ds)
        names.append(s.name)
        weights.append(s.weight)

    mixed = MixedPretrainDataset(datasets, names)

    num_samples = int(cfg.get("mix_num_samples", len(mixed)))
    replacement = _as_bool(cfg.get("replacement", True))
    seed = int(cfg.get("mix_seed", 0))

    sampler = MixtureSampler(
        dataset_sizes=mixed.sizes,
        dataset_weights=weights,
        num_samples=num_samples,
        replacement=replacement,
        seed=seed,
    )
    info = {
        "names": names,
        "weights": weights,
        "sizes": mixed.sizes,
        "num_samples": num_samples,
        "replacement": replacement,
        "seed": seed,
    }
    return mixed, sampler, info
