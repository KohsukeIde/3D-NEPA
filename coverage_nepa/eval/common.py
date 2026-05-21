from __future__ import annotations
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader

from coverage_nepa.data.coverage_dataset import CoveragePairDataset, CoverageStateDataset, list_cache_files, split_files
from coverage_nepa.models.coverage_nepa import CoverageNEPA


def load_ckpt(ckpt: str | Path, device: torch.device) -> CoverageNEPA:
    try:
        obj = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        obj = torch.load(ckpt, map_location=device)
    args = obj.get("args", {})
    model = CoverageNEPA(dim=int(args.get("dim", 256)), hidden=int(args.get("hidden", 512)), max_levels=int(args.get("max_levels", 12)), ema_momentum=float(args.get("ema_momentum", 0.996)))
    model.load_state_dict(obj["model"], strict=True)
    return model.to(device).eval()


def move(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
