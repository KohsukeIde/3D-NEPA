#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import builder
from utils import misc
from utils.config import cfg_from_yaml_file


SCANOBJECTNN_CLASS_NAMES = [
    "bag",
    "bin",
    "box",
    "cabinet",
    "chair",
    "desk",
    "display",
    "door",
    "shelf",
    "table",
    "bed",
    "pillow",
    "sink",
    "sofa",
    "toilet",
]


def parse_args():
    p = argparse.ArgumentParser("ScanObjectNN support-stress battery for PointGPT family")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_json", type=str, default="")
    p.add_argument("--output_md", type=str, default="")
    p.add_argument("--local_noise_sigma", type=float, default=0.08,
                   help="Gaussian noise sigma as a fraction of the object bbox diagonal for local_jitter.")
    return p.parse_args()


def load_config(config_path, batch_size, split):
    cfg = cfg_from_yaml_file(config_path)
    cfg.dataset.test.others.bs = batch_size
    cfg.dataset.test.others.subset = split
    return cfg


def build_loader(cfg, split, num_workers):
    cfg_split = cfg.dataset.test
    cfg_split.others.subset = split
    args = SimpleNamespace(distributed=False, num_workers=num_workers)
    _, loader = builder.dataset_builder(args, cfg_split)
    return loader


def build_model(cfg, ckpt):
    model = builder.model_builder(cfg.model)
    builder.load_model(model, ckpt, logger=None)
    model = model.cuda()
    model.eval()
    return model


def _resample_to_npoints(points, keep_idx, npoints, g):
    kept = points[keep_idx]
    if kept.shape[0] >= npoints:
        perm = torch.randperm(kept.shape[0], generator=g)[:npoints]
        return kept[perm]
    extra_idx = torch.randint(0, kept.shape[0], (npoints - kept.shape[0],), generator=g)
    return torch.cat([kept, kept[extra_idx]], dim=0)


def _local_patch_indices(pts, ratio, generator):
    npoints = pts.shape[0]
    patch_n = max(1, int(round(npoints * ratio)))
    anchor = int(torch.randint(0, npoints, (1,), generator=generator).item())
    dist = torch.sum((pts - pts[anchor : anchor + 1]) ** 2, dim=-1)
    return torch.topk(dist, k=patch_n, largest=False).indices


def _bbox_diag(pts):
    extent = pts.max(dim=0).values - pts.min(dim=0).values
    return torch.linalg.norm(extent).clamp_min(1e-6)


def apply_condition(points, condition, keep_ratio, generator, noise_sigma=0.08):
    bsz, npoints, _ = points.shape
    if condition == "clean":
        return points
    if condition == "xyz_zero":
        return torch.zeros_like(points)

    keep_n = max(1, int(round(npoints * keep_ratio)))
    out = []
    for b in range(bsz):
        pts = points[b]
        if condition == "random_drop":
            keep_idx = torch.randperm(npoints, generator=generator)[:keep_n]
        elif condition == "structured_drop":
            anchor = int(torch.randint(0, npoints, (1,), generator=generator).item())
            dist = torch.sum((pts - pts[anchor : anchor + 1]) ** 2, dim=-1)
            keep_idx = torch.topk(dist, k=keep_n, largest=True).indices
        else:
            raise ValueError(f"Unsupported condition: {condition}")
        out.append(_resample_to_npoints(pts, keep_idx, npoints, generator))
    return torch.stack(out, dim=0)


def apply_local_noise(points, condition, ratio, generator, noise_sigma=0.08):
    bsz, _, _ = points.shape
    out = points.clone()
    for b in range(bsz):
        pts = out[b]
        patch_idx = _local_patch_indices(pts, ratio, generator)
        if condition == "local_jitter":
            sigma = float(noise_sigma) * float(_bbox_diag(pts).item())
            noise = torch.randn((patch_idx.numel(), pts.shape[1]), generator=generator, dtype=pts.dtype) * sigma
            pts[patch_idx] = pts[patch_idx] + noise
        elif condition == "local_replace":
            mins = pts.min(dim=0).values
            maxs = pts.max(dim=0).values
            rand = torch.rand((patch_idx.numel(), pts.shape[1]), generator=generator, dtype=pts.dtype)
            pts[patch_idx] = mins + rand * (maxs - mins)
        else:
            raise ValueError(f"Unsupported local noise condition: {condition}")
    return out


def eval_condition(model, loader, npoints, condition, keep_ratio, max_batches, seed, noise_sigma):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    total = 0
    correct = 0
    per_total = torch.zeros(len(SCANOBJECTNN_CLASS_NAMES), dtype=torch.long)
    per_correct = torch.zeros(len(SCANOBJECTNN_CLASS_NAMES), dtype=torch.long)

    with torch.no_grad():
        for batch_idx, (_, _, data) in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            points = data[0]
            labels = data[1].view(-1)
            points = misc.fps(points.cuda(), npoints).cpu()
            if condition in {"local_jitter", "local_replace"}:
                stressed = apply_local_noise(points, condition, keep_ratio, g, noise_sigma=noise_sigma).cuda()
            else:
                stressed = apply_condition(points, condition, keep_ratio, g, noise_sigma=noise_sigma).cuda()
            logits, _ = model(stressed, compute_recon=False)
            pred = logits.argmax(dim=-1).cpu()

            total += labels.numel()
            correct += int((pred == labels).sum().item())
            for cls in labels.unique():
                cls = int(cls.item())
                cls_mask = labels == cls
                per_total[cls] += int(cls_mask.sum().item())
                per_correct[cls] += int((pred[cls_mask] == labels[cls_mask]).sum().item())

    per_class = {}
    for i, name in enumerate(SCANOBJECTNN_CLASS_NAMES):
        denom = int(per_total[i].item())
        per_class[name] = float(per_correct[i].item() / denom) if denom > 0 else float("nan")
    return {
        "acc": float(correct / max(1, total)),
        "per_class_acc": per_class,
    }


def to_md(summary):
    lines = []
    lines.append("# ScanObjectNN Support Stress")
    lines.append("")
    lines.append(f"- config: `{summary['config']}`")
    lines.append(f"- ckpt: `{summary['ckpt']}`")
    lines.append(f"- split: `{summary['split']}`")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| condition | acc |")
    lines.append("|---|---:|")
    for row in summary["conditions"]:
        lines.append(f"| `{row['name']}` | `{row['acc']:.4f}` |")
    lines.append("")
    focus = ["cabinet", "chair", "desk", "display", "door", "sink", "table", "toilet"]
    lines.append("## Focus Classes")
    lines.append("")
    lines.append("| condition | " + " | ".join(f"`{x}`" for x in focus) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(focus)) + "|")
    for row in summary["conditions"]:
        vals = [f"`{row['per_class_acc'].get(x, float('nan')):.4f}`" for x in focus]
        lines.append(f"| `{row['name']}` | " + " | ".join(vals) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    cfg = load_config(args.config, args.batch_size, args.split)
    loader = build_loader(cfg, args.split, args.num_workers)
    model = build_model(cfg, args.ckpt)

    conditions = [
        ("clean", "clean", 1.0),
        ("random_keep80", "random_drop", 0.8),
        ("random_keep50", "random_drop", 0.5),
        ("random_keep20", "random_drop", 0.2),
        ("random_keep10", "random_drop", 0.1),
        ("structured_keep80", "structured_drop", 0.8),
        ("structured_keep50", "structured_drop", 0.5),
        ("structured_keep20", "structured_drop", 0.2),
        ("structured_keep10", "structured_drop", 0.1),
        ("local_jitter80", "local_jitter", 0.8),
        ("local_jitter50", "local_jitter", 0.5),
        ("local_jitter20", "local_jitter", 0.2),
        ("local_jitter10", "local_jitter", 0.1),
        ("local_replace80", "local_replace", 0.8),
        ("local_replace50", "local_replace", 0.5),
        ("local_replace20", "local_replace", 0.2),
        ("local_replace10", "local_replace", 0.1),
        ("xyz_zero", "xyz_zero", 0.0),
    ]
    rows = []
    for idx, (name, cond, keep_ratio) in enumerate(conditions):
        result = eval_condition(
            model,
            loader,
            cfg.npoints,
            cond,
            keep_ratio,
            max_batches=args.max_batches,
            seed=args.seed + idx,
            noise_sigma=args.local_noise_sigma,
        )
        result["name"] = name
        rows.append(result)

    summary = {
        "config": args.config,
        "ckpt": args.ckpt,
        "split": args.split,
        "conditions": rows,
    }

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(to_md(summary))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
