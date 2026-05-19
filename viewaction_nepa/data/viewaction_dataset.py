from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class ViewActionDataset(Dataset):
    """Dataset of partial-view transitions from the generated cache."""

    def __init__(self, cache_root: str | Path, split: str = "train", mode: str = "pair",
                 max_shapes: int = 0, seed: int = 0):
        self.cache_root = Path(cache_root)
        self.split = split
        self.mode = mode
        graph_npz = np.load(self.cache_root / "view_graph.npz")
        self.edges = graph_npz["edges"].astype(np.int64)
        self.action_id = graph_npz["action_id"].astype(np.int64)
        self.action_vec = graph_npz["action_vec"].astype(np.float32)
        self.camera_pos = graph_npz["camera_pos"].astype(np.float32)
        self.neighbor_index = graph_npz["neighbor_index"].astype(np.int64)
        manifest_path = self.cache_root / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            rows = manifest.get("rows", [])
            if split not in {"all", "full"}:
                matched = [r for r in rows if r.get("source_split", "all") == split]
                if matched:
                    rows = matched
                elif all(r.get("source_split", "all") in {"all", ""} for r in rows):
                    rng = np.random.default_rng(seed)
                    order = np.arange(len(rows))
                    rng.shuffle(order)
                    rows = [rows[i] for i in order]
                    n = len(rows)
                    if split == "train":
                        rows = rows[: int(0.9 * n)]
                    elif split == "val":
                        rows = rows[int(0.9 * n): int(0.95 * n)]
                    elif split == "test":
                        rows = rows[int(0.95 * n):]
                    else:
                        rows = []
                else:
                    rows = []
            self.files = [self.cache_root / r["cache_path"] for r in rows]
        else:
            files = sorted([p for p in self.cache_root.rglob("*.npz") if p.name != "view_graph.npz"])
            # Simple deterministic split by file order if no manifest exists.
            rng = np.random.default_rng(seed)
            files = np.asarray(files, dtype=object)
            order = np.arange(len(files))
            rng.shuffle(order)
            files = files[order].tolist()
            n = len(files)
            if split == "train":
                self.files = files[: int(0.9 * n)]
            elif split == "val":
                self.files = files[int(0.9 * n): int(0.95 * n)]
            elif split == "test":
                self.files = files[int(0.95 * n):]
            else:
                self.files = files
        if max_shapes:
            self.files = self.files[:max_shapes]
        self.index = []
        if mode == "all_views":
            for fi, _ in enumerate(self.files):
                self.index.append((fi, -1))
        else:
            for fi, _ in enumerate(self.files):
                for ei in range(len(self.edges)):
                    self.index.append((fi, ei))

    def __len__(self) -> int:
        return len(self.index)

    def _load(self, fi: int) -> dict[str, Any]:
        data = np.load(self.files[fi], allow_pickle=True)
        return {k: data[k] for k in data.files}

    @staticmethod
    def _as_str(value: Any, fallback: str) -> str:
        if value is None:
            return fallback
        arr = np.asarray(value)
        if arr.shape == ():
            return str(arr.item())
        return str(value)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        fi, ei = self.index[idx]
        data = self._load(fi)
        views = data["views"].astype(np.float32)
        category = self._as_str(data.get("category"), "unknown")
        shape_id = self._as_str(data.get("shape_id"), self.files[fi].stem)
        if self.mode == "all_views":
            return {
                "views": torch.from_numpy(views),
                "shape_id": shape_id,
                "category": category,
            }
        s, t = self.edges[ei]
        # Next edge for 2-step rollout: choose first outgoing edge from target if available.
        outgoing = np.nonzero(self.edges[:, 0] == t)[0]
        ei2 = int(outgoing[0]) if len(outgoing) else int(ei)
        _, u = self.edges[ei2]
        outgoing_src = np.nonzero(self.edges[:, 0] == s)[0]
        candidate_targets = self.edges[outgoing_src, 1]
        candidate_match = np.nonzero(outgoing_src == ei)[0]
        candidate_index = int(candidate_match[0]) if len(candidate_match) else 0
        wrong_outgoing = np.nonzero((self.edges[:, 0] == s) & (self.edges[:, 1] != t))[0]
        wrong_ei = int(wrong_outgoing[0]) if len(wrong_outgoing) else int(ei)
        return {
            "points_t": torch.from_numpy(views[s]),
            "points_tp1": torch.from_numpy(views[t]),
            "points_tp2": torch.from_numpy(views[u]),
            "candidate_points": torch.from_numpy(views[candidate_targets].astype(np.float32)),
            "candidate_index": torch.tensor(candidate_index, dtype=torch.long),
            "view_t": torch.tensor(int(s), dtype=torch.long),
            "view_tp1": torch.tensor(int(t), dtype=torch.long),
            "view_tp2": torch.tensor(int(u), dtype=torch.long),
            "action_id": torch.tensor(int(self.action_id[ei]), dtype=torch.long),
            "action_vec": torch.from_numpy(self.action_vec[ei]),
            "action_id_next": torch.tensor(int(self.action_id[ei2]), dtype=torch.long),
            "action_vec_next": torch.from_numpy(self.action_vec[ei2]),
            "wrong_action_id": torch.tensor(int(self.action_id[wrong_ei]), dtype=torch.long),
            "wrong_action_vec": torch.from_numpy(self.action_vec[wrong_ei]),
            "shape_id": shape_id,
            "category": category,
        }
