from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def list_cache_entries(cache_root: str | Path) -> list[dict]:
    root = Path(cache_root)
    manifest = root / "manifest.json"
    entries: list[dict] = []
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        for row in payload.get("rows", []):
            p = root / row.get("cache_path", "")
            if p.exists():
                entries.append(
                    {
                        "path": p,
                        "source_split": str(row.get("source_split", "")),
                        "category": str(row.get("category", p.parent.name)),
                        "shape_id": str(row.get("shape_id", p.stem)),
                    }
                )
    if not entries:
        for p in sorted([p for p in root.rglob("*.npz") if p.name not in {"view_graph.npz"}]):
            source_split = ""
            category = p.parent.name
            shape_id = p.stem
            try:
                with np.load(p, allow_pickle=True) as data:
                    if "source_split" in data:
                        source_split = str(np.asarray(data["source_split"]).item())
                    if "category" in data:
                        category = str(np.asarray(data["category"]).item())
                    if "shape_id" in data:
                        shape_id = str(np.asarray(data["shape_id"]).item())
            except Exception:
                pass
            entries.append({"path": p, "source_split": source_split, "category": category, "shape_id": shape_id})
    return entries


def list_cache_files(cache_root: str | Path) -> list[Path]:
    return [row["path"] for row in list_cache_entries(cache_root)]


def split_files(files: list[Path], split: str = "train", val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 0) -> list[Path]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    n_test = int(round(len(files) * test_frac))
    n_val = int(round(len(files) * val_frac))
    test_idx = set(idx[:n_test].tolist())
    val_idx = set(idx[n_test : n_test + n_val].tolist())
    out = []
    for i, p in enumerate(files):
        if split == "test" and i in test_idx:
            out.append(p)
        elif split == "val" and i in val_idx:
            out.append(p)
        elif split == "train" and i not in test_idx and i not in val_idx:
            out.append(p)
        elif split == "all":
            out.append(p)
    return out


def select_cache_files(
    cache_root: str | Path,
    split: str = "train",
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
) -> list[Path]:
    entries = list_cache_entries(cache_root)
    if split == "all":
        return [e["path"] for e in entries]
    source_splits = {e.get("source_split", "") for e in entries}
    has_source_split = bool(source_splits & {"train", "val", "test"})
    if not has_source_split:
        return split_files([e["path"] for e in entries], split=split, seed=seed, val_frac=val_frac, test_frac=test_frac)

    train_entries = [e for e in entries if e.get("source_split") == "train"]
    val_entries = [e for e in entries if e.get("source_split") == "val"]
    test_entries = [e for e in entries if e.get("source_split") == "test"]
    rng = np.random.default_rng(seed)
    if not val_entries and train_entries:
        idx = np.arange(len(train_entries))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(train_entries) * val_frac))) if len(train_entries) > 1 else 0
        val_idx = set(idx[:n_val].tolist())
        derived_val = [e for i, e in enumerate(train_entries) if i in val_idx]
        derived_train = [e for i, e in enumerate(train_entries) if i not in val_idx]
    else:
        derived_train = train_entries
        derived_val = val_entries
    pools = {"train": derived_train, "val": derived_val, "test": test_entries}
    out = [e["path"] for e in pools.get(split, [])]
    if not out and split == "test":
        # Some smoke caches are built only from train/all shapes.
        return split_files([e["path"] for e in entries], split=split, seed=seed, val_frac=val_frac, test_frac=test_frac)
    return out


class CoveragePairDataset(Dataset):
    def __init__(
        self,
        cache_root: str | Path,
        split: str = "train",
        seed: int = 0,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        deterministic: bool = False,
        pair_mode: str = "forward",
    ) -> None:
        self.cache_root = Path(cache_root)
        self.files = select_cache_files(cache_root, split=split, seed=seed, val_frac=val_frac, test_frac=test_frac)
        self.seed = seed
        self.deterministic = deterministic
        self.pair_mode = pair_mode
        if not self.files:
            raise RuntimeError(f"no files for split={split} in {cache_root}")

    def __len__(self) -> int:
        return len(self.files)

    def _rng(self, index: int) -> np.random.Generator:
        if self.deterministic:
            return np.random.default_rng(self.seed + index * 1_000_003)
        return np.random.default_rng()

    def __getitem__(self, index: int) -> dict:
        path = self.files[index]
        data = np.load(path, allow_pickle=True)
        cov = np.asarray(data["coverage"], dtype=np.float32)
        levels = np.asarray(data.get("levels", np.arange(cov.shape[0])), dtype=np.int64)
        kmax = cov.shape[0]
        rng = self._rng(index)
        if kmax < 2:
            k, m = 0, 0
        elif self.pair_mode == "adjacent":
            k = int(rng.integers(0, kmax - 1))
            m = k + 1
        else:
            k = int(rng.integers(0, kmax - 1))
            m = int(rng.integers(k + 1, kmax))
        return {
            "x_k": torch.from_numpy(cov[k]).float(),
            "x_m": torch.from_numpy(cov[m]).float(),
            "level_k": torch.tensor(int(levels[k]), dtype=torch.long),
            "level_m": torch.tensor(int(levels[m]), dtype=torch.long),
            "file_index": torch.tensor(index, dtype=torch.long),
            "shape_id": str(data.get("shape_id", path.stem)),
            "category": str(data.get("category", path.parent.name)),
        }


class CoverageStateDataset(Dataset):
    def __init__(self, cache_root: str | Path, split: str = "all", seed: int = 0) -> None:
        self.cache_root = Path(cache_root)
        self.files = select_cache_files(cache_root, split=split, seed=seed)
        self.rows: list[tuple[Path, int]] = []
        for p in self.files:
            data = np.load(p, allow_pickle=True)
            k = int(np.asarray(data["coverage"]).shape[0])
            self.rows.extend([(p, i) for i in range(k)])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        p, level = self.rows[index]
        data = np.load(p, allow_pickle=True)
        cov = np.asarray(data["coverage"], dtype=np.float32)
        raw_stats = np.asarray(data.get("raw_stats", np.zeros((cov.shape[0], 1))), dtype=np.float32)
        raw_count = np.asarray(data.get("raw_union_count", np.zeros(cov.shape[0])), dtype=np.int64)
        return {
            "points": torch.from_numpy(cov[level]).float(),
            "level": torch.tensor(level, dtype=torch.long),
            "raw_stats": torch.from_numpy(raw_stats[level]).float(),
            "raw_count": torch.tensor(int(raw_count[level]), dtype=torch.float32),
            "shape_index": torch.tensor(self.files.index(p), dtype=torch.long),
            "shape_id": str(data.get("shape_id", p.stem)),
            "category": str(data.get("category", p.parent.name)),
        }
