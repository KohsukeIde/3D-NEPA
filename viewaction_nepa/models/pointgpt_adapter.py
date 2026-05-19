from __future__ import annotations

from pathlib import Path
import sys
import torch
import torch.nn as nn


class PointGPTFeatureAdapter(nn.Module):
    """Best-effort PointGPT feature adapter.

    The PointGPT codebase exposes several variants. This adapter is intentionally
    conservative and may need a one-line adjustment depending on the current
    PointGPT model API. The Phase-0 pipeline defaults to SimplePointEncoder.
    """
    def __init__(self, pointgpt_dir: str, config_path: str, ckpt_path: str | None = None):
        super().__init__()
        pg = Path(pointgpt_dir).resolve()
        sys.path.insert(0, str(pg))
        from utils.config import cfg_from_yaml_file  # type: ignore
        from tools import builder  # type: ignore
        cfg = cfg_from_yaml_file(str(pg / config_path if not Path(config_path).is_absolute() else config_path))
        self.model = builder.model_builder(cfg.model)
        if ckpt_path:
            builder.load_model(self.model, ckpt_path, logger=None)
        self.output_dim = getattr(cfg.model, "trans_dim", getattr(cfg.model, "cls_dim", 384))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "extract_features"):
            return self.model.extract_features(points)
        if hasattr(self.model, "forward_features"):
            return self.model.forward_features(points)
        # Fallback: call model and hope it returns feature tuple/dict.
        out = self.model(points)
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, (tuple, list)):
            for x in out:
                if isinstance(x, torch.Tensor) and x.dim() == 2:
                    return x
        if isinstance(out, dict):
            for k in ["feat", "features", "cls", "z"]:
                if k in out:
                    return out[k]
        raise RuntimeError("Could not extract features from PointGPT model. Add a small extract_features method or use SimplePointEncoder for smoke.")
