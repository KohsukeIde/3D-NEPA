from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from noise_nepa.data.noise_dataset import NoisePairDataset
from noise_nepa.models.noise_nepa import NoiseNEPA


def load_model(ckpt_path: str | Path, device="cuda"):
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})
    model = NoiseNEPA(
        embed_dim=int(args.get("embed_dim", 384)),
        num_steps=int(args.get("num_steps", 1000)),
        ema_momentum=float(args.get("ema_momentum", 0.996)),
        semigroup_weight=float(args.get("semigroup_weight", 0.0)),
        var_weight=float(args.get("var_weight", 0.0)),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, args


def make_loader(
    data_root,
    split_file="",
    max_shapes=512,
    npoints=1024,
    batch_size=64,
    num_workers=4,
    fixed_eval=True,
    num_steps=1000,
    min_gap=50,
    seed=0,
    epsilon_mode="independent",
):
    ds = NoisePairDataset(
        data_root,
        split_file=split_file or None,
        max_shapes=max_shapes,
        npoints=npoints,
        fixed_eval=fixed_eval,
        num_steps=num_steps,
        min_gap=min_gap,
        seed=seed,
        epsilon_mode=epsilon_mode,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)


def topk_retrieval(pred, target, topk=(1, 5)):
    pred = torch.nn.functional.normalize(pred, dim=-1)
    target = torch.nn.functional.normalize(target, dim=-1)
    sim = pred @ target.t()
    labels = torch.arange(pred.shape[0], device=pred.device)
    out = {}
    for k in topk:
        hit = sim.topk(k=min(k, sim.shape[1]), dim=1).indices.eq(labels[:, None]).any(dim=1).float().mean()
        out[f"top{k}"] = float(hit.cpu())
    ranks = torch.argsort(sim, dim=1, descending=True)
    pos = (ranks == labels[:, None]).nonzero()[:, 1].float() + 1.0
    out["mrr"] = float((1.0 / pos).mean().cpu())
    out["mean_rank"] = float(pos.mean().cpu())
    pos_sim = sim[labels, labels]
    masked = sim.clone()
    masked[labels, labels] = -1e9
    neg_sim = masked.max(dim=1).values
    out["margin"] = float((pos_sim - neg_sim).mean().cpu())
    return out


def effective_rank(z: torch.Tensor) -> float:
    if z.shape[0] < 2:
        return 1.0
    centered = z - z.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(centered.float())
    p = s / s.sum().clamp_min(1e-12)
    return float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()).cpu())


def write_json_md(metrics: dict, out_json: str | Path, out_md: str | Path, title: str):
    out_json = Path(out_json); out_md = Path(out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2))
    lines = [f"# {title}", "", "| metric | value |", "|---|---:|"]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"| `{k}` | {v:.6f} |")
        else:
            lines.append(f"| `{k}` | {v} |")
    out_md.write_text("\n".join(lines) + "\n")
