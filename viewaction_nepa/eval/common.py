from __future__ import annotations
from pathlib import Path
import torch

from viewaction_nepa.data.viewaction_dataset import ViewActionDataset
from viewaction_nepa.models.simple_point_encoder import SimplePointEncoder
from viewaction_nepa.models.viewaction_nepa import ViewActionNEPA


def load_model(ckpt_path: str, cache_root: str, device: str = "cuda") -> ViewActionNEPA:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})
    ds = ViewActionDataset(cache_root, split="all", mode="pair", max_shapes=1)
    latent_dim = int(args.get("latent_dim", 384))
    encoder = SimplePointEncoder(latent_dim)
    model = ViewActionNEPA(encoder=encoder, num_actions=int(ds.action_id.max()) + 1,
                           action_vec_dim=int(ds.action_vec.shape[1]), latent_dim=latent_dim,
                           ema_momentum=float(args.get("ema_momentum", 0.996)),
                           inverse_weight=float(args.get("inverse_weight", 0.1)),
                           variance_weight=float(args.get("variance_weight", 0.0)),
                           contrast_weight=float(args.get("contrast_weight", 0.0)),
                           contrast_temperature=float(args.get("contrast_temperature", 0.1)),
                           hard_contrast_weight=float(args.get("hard_contrast_weight", 0.0)))
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model


def encode_all_views(model: ViewActionNEPA, views: torch.Tensor, batch_size: int = 32, device: str = "cuda") -> torch.Tensor:
    zs = []
    with torch.no_grad():
        for i in range(0, views.shape[0], batch_size):
            pts = views[i:i+batch_size].to(device)
            zs.append(model.encode_target(pts).detach().cpu())
    return torch.cat(zs, dim=0)
