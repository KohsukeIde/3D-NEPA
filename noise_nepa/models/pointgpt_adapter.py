from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointGPTFeatureAdapter(nn.Module):
    """Best-effort feature adapter for existing PointGPT code.

    This adapter is intentionally conservative. It tries to call common feature
    extraction methods if present. If the local PointGPT model API differs, add
    a `forward_features(points)->features` method in `PointGPT/models/PointGPT.py`
    and this wrapper will use it.
    """
    def __init__(self, repo_root: str | Path, config_path: str, ckpt_path: str | None = None, out_dim: int = 384):
        super().__init__()
        self.repo_root = Path(repo_root).resolve()
        pointgpt_dir = self.repo_root / "PointGPT"
        sys.path.insert(0, str(pointgpt_dir))
        from utils.config import cfg_from_yaml_file  # type: ignore
        from tools import builder  # type: ignore
        cfg = cfg_from_yaml_file(str(pointgpt_dir / config_path if not Path(config_path).is_absolute() else config_path))
        self.model = builder.model_builder(cfg.model)
        if ckpt_path:
            builder.load_model(self.model, ckpt_path, logger=None)
        self.out_dim = out_dim
        self.proj = None

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        m = self.model
        if hasattr(m, "forward_features"):
            z = m.forward_features(points)
        elif hasattr(m, "extract_features"):
            z = m.extract_features(points)
        elif hasattr(m, "encoder"):
            z = m.encoder(points)
        else:
            raise RuntimeError(
                "Local PointGPT model has no feature API. Add forward_features(points) to PointGPT model or use SimplePointEncoder."
            )
        if isinstance(z, (tuple, list)):
            z = z[0]
        if z.dim() == 3:
            z = z.mean(dim=1)
        if z.shape[-1] != self.out_dim:
            if self.proj is None:
                self.proj = nn.Linear(z.shape[-1], self.out_dim).to(z.device)
            z = self.proj(z)
        return F.normalize(z, dim=-1)
