"""ShapeNetPart raw txt dataset for local part-seg fine-tuning.

This follows the Point-MAE / PCP-MAE raw data contract:

- root/synsetoffset2category.txt
- root/train_test_split/shuffled_{train,val,test}_file_list.json
- root/<synset>/<shape_id>.txt

Each txt row is expected to be:
- xyz(3) + optional normal(3) + seg_label(1)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


SEG_CLASSES: Dict[str, List[int]] = {
    "Earphone": [16, 17, 18],
    "Motorbike": [30, 31, 32, 33, 34, 35],
    "Rocket": [41, 42, 43],
    "Car": [8, 9, 10, 11],
    "Laptop": [28, 29],
    "Cap": [6, 7],
    "Skateboard": [44, 45, 46],
    "Mug": [36, 37],
    "Guitar": [19, 20, 21],
    "Bag": [4, 5],
    "Lamp": [24, 25, 26, 27],
    "Table": [47, 48, 49],
    "Airplane": [0, 1, 2, 3],
    "Pistol": [38, 39, 40],
    "Chair": [12, 13, 14, 15],
    "Knife": [22, 23],
}


@dataclass
class ShapeNetPartAugConfig:
    scale_low: float = 0.8
    scale_high: float = 1.25
    shift_range: float = 0.1


def pc_normalize(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    centroid = np.mean(xyz, axis=0, keepdims=True)
    xyz = xyz - centroid
    radius = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
    if float(radius) > 0.0:
        xyz = xyz / float(radius)
    return xyz


def _random_scale_translate(
    xyz: np.ndarray,
    rng: np.random.RandomState,
    cfg: ShapeNetPartAugConfig,
) -> np.ndarray:
    scale = float(rng.uniform(float(cfg.scale_low), float(cfg.scale_high)))
    shift = rng.uniform(-float(cfg.shift_range), float(cfg.shift_range), size=(1, 3)).astype(np.float32)
    return xyz * scale + shift


class ShapeNetPartDataset(Dataset):
    def __init__(
        self,
        root: str,
        *,
        n_point: int = 2048,
        split: str = "trainval",
        normal_channel: bool = False,
        class_choice: Optional[List[str]] = None,
        aug: bool = False,
        aug_cfg: Optional[ShapeNetPartAugConfig] = None,
        seed: int = 0,
        deterministic_eval_sampling: bool = True,
        cache_size: int = 20000,
    ) -> None:
        super().__init__()
        self.root = str(root)
        self.n_point = int(n_point)
        self.normal_channel = bool(normal_channel)
        self.aug = bool(aug)
        self.aug_cfg = aug_cfg or ShapeNetPartAugConfig()
        self.seed = int(seed)
        self.deterministic_eval_sampling = bool(deterministic_eval_sampling)
        self.cache_size = int(cache_size)
        self.cache: Dict[int, Tuple[np.ndarray, int, np.ndarray, str]] = {}

        catfile = Path(self.root) / "synsetoffset2category.txt"
        if not catfile.is_file():
            raise FileNotFoundError(f"missing ShapeNetPart category file: {catfile}")

        cat: Dict[str, str] = {}
        with open(catfile, "r") as f:
            for line in f:
                name, synset = line.strip().split()
                cat[name] = synset
        if class_choice is not None:
            choice = set(class_choice)
            cat = {k: v for k, v in cat.items() if k in choice}
        self.cat = cat
        self.classes_original = dict(zip(self.cat.keys(), range(len(self.cat))))
        self.class_idx_to_cat = {v: k for k, v in self.classes_original.items()}

        split_root = Path(self.root) / "train_test_split"
        with open(split_root / "shuffled_train_file_list.json", "r") as f:
            train_ids = {str(d.split("/")[2]) for d in json.load(f)}
        with open(split_root / "shuffled_val_file_list.json", "r") as f:
            val_ids = {str(d.split("/")[2]) for d in json.load(f)}
        with open(split_root / "shuffled_test_file_list.json", "r") as f:
            test_ids = {str(d.split("/")[2]) for d in json.load(f)}

        datapath: List[Tuple[str, Path]] = []
        for item, synset in self.cat.items():
            dir_point = Path(self.root) / synset
            fns = sorted(os.listdir(dir_point))
            if split == "trainval":
                fns = [fn for fn in fns if ((fn[:-4] in train_ids) or (fn[:-4] in val_ids))]
            elif split == "train":
                fns = [fn for fn in fns if fn[:-4] in train_ids]
            elif split == "val":
                fns = [fn for fn in fns if fn[:-4] in val_ids]
            elif split == "test":
                fns = [fn for fn in fns if fn[:-4] in test_ids]
            else:
                raise ValueError(f"unknown ShapeNetPart split={split}")
            for fn in fns:
                datapath.append((item, dir_point / fn))
        self.datapath = datapath

    def __len__(self) -> int:
        return len(self.datapath)

    def _get_rng(self, index: int) -> np.random.RandomState:
        if self.aug or (not self.deterministic_eval_sampling):
            seed = int(self.seed + index * 10007 + np.random.randint(0, 2**16))
        else:
            seed = int(self.seed + index * 10007)
        return np.random.RandomState(seed & 0xFFFFFFFF)

    def _load_item(self, index: int) -> Tuple[np.ndarray, int, np.ndarray, str]:
        if index in self.cache:
            return self.cache[index]
        cat_name, path = self.datapath[index]
        cls = int(self.classes_original[cat_name])
        data = np.loadtxt(path).astype(np.float32)
        pts = data[:, :6] if self.normal_channel else data[:, :3]
        seg = data[:, -1].astype(np.int64)
        item = (pts, cls, seg, cat_name)
        if len(self.cache) < self.cache_size:
            self.cache[index] = item
        return item

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        point_set, cls, seg, cat_name = self._load_item(index)
        point_set = np.asarray(point_set, dtype=np.float32).copy()
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        rng = self._get_rng(index)
        choice = rng.choice(int(seg.shape[0]), self.n_point, replace=True)
        point_set = point_set[choice, :]
        seg = seg[choice]

        if self.aug:
            point_set[:, 0:3] = _random_scale_translate(point_set[:, 0:3], rng, self.aug_cfg)

        out = {
            "xyz": torch.from_numpy(point_set[:, 0:3].astype(np.float32, copy=False)),
            "cls_label": torch.tensor(cls, dtype=torch.long),
            "seg_label": torch.from_numpy(seg.astype(np.int64, copy=False)),
            "cat_name": cat_name,
        }
        if self.normal_channel:
            out["normal"] = torch.from_numpy(point_set[:, 3:6].astype(np.float32, copy=False))
        return out
