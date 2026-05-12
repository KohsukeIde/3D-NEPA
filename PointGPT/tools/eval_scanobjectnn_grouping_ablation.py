#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import builder
from tools.eval_scanobjectnn_support_stress import (  # noqa: E402
    SCANOBJECTNN_CLASS_NAMES,
    apply_condition,
    load_config,
)
from utils import misc  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser("ScanObjectNN grouping/patchization ablation for PointGPT family")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--group-modes",
        default="fps_knn,random_center_knn,voxel_center_knn,radius_fps,random_group",
    )
    p.add_argument("--support-conditions", default="clean,random_keep20,structured_keep20")
    p.add_argument("--radius", type=float, default=0.22)
    p.add_argument("--voxel-grid", type=int, default=6)
    p.add_argument("--output_json", type=str, default="")
    p.add_argument("--output_csv", type=str, default="")
    p.add_argument("--output_md", type=str, default="")
    return p.parse_args()


def split_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


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


def batched_gather_points(xyz: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    batch_size, num_points, _ = xyz.shape
    idx_base = torch.arange(batch_size, device=xyz.device).view(batch_size, 1, 1) * num_points
    flat_idx = (idx + idx_base).reshape(-1)
    gathered = xyz.reshape(batch_size * num_points, 3)[flat_idx]
    return gathered.reshape(batch_size, idx.shape[1], idx.shape[2], 3)


def select_random_centers(xyz: torch.Tensor, num_group: int, generator: torch.Generator | None) -> torch.Tensor:
    batch_size, num_points, _ = xyz.shape
    idx_rows = []
    for _ in range(batch_size):
        if num_points >= num_group:
            idx = torch.randperm(num_points, device=xyz.device, generator=generator)[:num_group]
        else:
            idx = torch.randint(0, num_points, (num_group,), device=xyz.device, generator=generator)
        idx_rows.append(idx)
    center_idx = torch.stack(idx_rows, dim=0)
    return xyz.gather(1, center_idx.unsqueeze(-1).expand(-1, -1, 3))


def select_voxel_centers(xyz: torch.Tensor, num_group: int, grid: int) -> torch.Tensor:
    batch_size, num_points, _ = xyz.shape
    centers = []
    for b in range(batch_size):
        pts = xyz[b]
        lo = pts.min(dim=0, keepdim=True).values
        hi = pts.max(dim=0, keepdim=True).values
        norm = (pts - lo) / (hi - lo + 1e-6)
        cell = torch.clamp((norm * grid).long(), 0, grid - 1)
        key = cell[:, 0] * grid * grid + cell[:, 1] * grid + cell[:, 2]
        chosen = []
        for key_value in torch.unique(key, sorted=True):
            idx = torch.nonzero(key == key_value, as_tuple=False).view(-1)
            centroid = pts[idx].mean(dim=0, keepdim=True)
            nearest = idx[torch.argmin(torch.sum((pts[idx] - centroid) ** 2, dim=1))]
            chosen.append(nearest)
        if not chosen:
            chosen = [torch.tensor(0, device=xyz.device)]
        chosen_idx = torch.stack(chosen).long()
        if chosen_idx.numel() >= num_group:
            chosen_pts = pts[chosen_idx]
            sub = misc.fps(chosen_pts.unsqueeze(0), num_group).squeeze(0)
            centers.append(sub)
        else:
            fill = misc.fps(pts.unsqueeze(0), num_group - chosen_idx.numel()).squeeze(0)
            centers.append(torch.cat([pts[chosen_idx], fill], dim=0))
    return torch.stack(centers, dim=0)


def radius_indices(xyz: torch.Tensor, center: torch.Tensor, group_size: int, radius: float) -> torch.Tensor:
    dist = torch.cdist(center, xyz)  # B G N
    nearest = torch.argsort(dist, dim=-1)
    idx = nearest[:, :, :group_size].clone()
    within = dist <= radius * radius if radius <= 0 else dist <= radius
    for b in range(xyz.shape[0]):
        for g in range(center.shape[1]):
            candidates = torch.nonzero(within[b, g], as_tuple=False).view(-1)
            if candidates.numel() == 0:
                continue
            cand_dist = dist[b, g, candidates]
            candidates = candidates[torch.argsort(cand_dist)]
            if candidates.numel() >= group_size:
                idx[b, g] = candidates[:group_size]
            else:
                idx[b, g, : candidates.numel()] = candidates
                idx[b, g, candidates.numel() :] = candidates[-1]
    return idx


def random_group_indices(xyz: torch.Tensor, num_group: int, group_size: int, generator: torch.Generator | None) -> torch.Tensor:
    batch_size, num_points, _ = xyz.shape
    rows = []
    for _ in range(batch_size):
        rows.append(torch.randint(0, num_points, (num_group, group_size), device=xyz.device, generator=generator))
    return torch.stack(rows, dim=0)


def make_group_forward(mode: str, radius: float, voxel_grid: int, seed: int):
    def forward(self, xyz):
        batch_size, num_points, _ = xyz.shape
        generator = torch.Generator(device=xyz.device)
        generator.manual_seed(int(seed))
        if mode == "fps_knn":
            center = misc.fps(xyz, self.num_group)
            _, idx = self.knn(xyz, center)
        elif mode == "random_center_knn":
            center = select_random_centers(xyz, self.num_group, generator)
            _, idx = self.knn(xyz, center)
        elif mode == "voxel_center_knn":
            center = select_voxel_centers(xyz, self.num_group, voxel_grid)
            _, idx = self.knn(xyz, center)
        elif mode == "radius_fps":
            center = misc.fps(xyz, self.num_group)
            idx = radius_indices(xyz, center, self.group_size, radius)
        elif mode == "random_group":
            center = select_random_centers(xyz, self.num_group, generator)
            idx = random_group_indices(xyz, self.num_group, self.group_size, generator)
        else:
            raise ValueError(f"unsupported group mode: {mode}")

        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        neighborhood = batched_gather_points(xyz, idx)
        neighborhood = neighborhood - center.unsqueeze(2)

        if hasattr(self, "build_sorted_indices"):
            sorted_indices = self.build_sorted_indices(xyz, center)
        else:
            sorted_indices = self.simplied_morton_sorting(xyz, center)
        neighborhood = neighborhood.reshape(batch_size * self.num_group, self.group_size, 3)[sorted_indices]
        neighborhood = neighborhood.reshape(batch_size, self.num_group, self.group_size, 3).contiguous()
        center = center.reshape(batch_size * self.num_group, 3)[sorted_indices]
        center = center.reshape(batch_size, self.num_group, 3).contiguous()
        return neighborhood, center

    return forward


def patch_group_modules(model, mode: str, radius: float, voxel_grid: int, seed: int) -> int:
    count = 0
    for module in model.modules():
        if all(hasattr(module, attr) for attr in ("num_group", "group_size", "knn")):
            module.forward = MethodType(make_group_forward(mode, radius, voxel_grid, seed + count * 997), module)
            count += 1
    return count


def condition_to_base(condition: str) -> tuple[str, float]:
    if condition == "clean":
        return "clean", 1.0
    if condition == "xyz_zero":
        return "xyz_zero", 0.0
    if condition.startswith("random_keep"):
        return "random_drop", int(condition.replace("random_keep", "")) / 100.0
    if condition.startswith("structured_keep"):
        return "structured_drop", int(condition.replace("structured_keep", "")) / 100.0
    raise ValueError(condition)


def evaluate(model, loader, npoints, group_mode: str, condition: str, max_batches: int, seed: int):
    cond, keep_ratio = condition_to_base(condition)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
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
            stressed = apply_condition(points, cond, keep_ratio, generator).cuda()
            logits, _ = model(stressed, compute_recon=False)
            pred = logits.argmax(dim=-1).cpu()

            total += labels.numel()
            correct += int((pred == labels).sum().item())
            for cls in labels.unique():
                cls_i = int(cls.item())
                cls_mask = labels == cls_i
                per_total[cls_i] += int(cls_mask.sum().item())
                per_correct[cls_i] += int((pred[cls_mask] == labels[cls_mask]).sum().item())

    per_class = {}
    for i, name in enumerate(SCANOBJECTNN_CLASS_NAMES):
        denom = int(per_total[i].item())
        per_class[name] = float(per_correct[i].item() / denom) if denom > 0 else float("nan")
    return {
        "group_mode": group_mode,
        "condition": condition,
        "acc": float(correct / max(1, total)),
        "per_class_acc": per_class,
    }


def to_md(summary):
    lines = [
        "# ScanObjectNN Grouping Ablation",
        "",
        f"- config: `{summary['config']}`",
        f"- ckpt: `{summary['ckpt']}`",
        f"- split: `{summary['split']}`",
        f"- radius: `{summary['radius']}`",
        f"- voxel grid: `{summary['voxel_grid']}`",
        "",
        "| group mode | condition | acc |",
        "|---|---|---:|",
    ]
    for row in summary["rows"]:
        lines.append(f"| `{row['group_mode']}` | `{row['condition']}` | `{row['acc']:.4f}` |")
    lines += [
        "",
        "## Notes",
        "",
        "- `fps_knn` is the trained/default patchization.",
        "- `random_center_knn` keeps local kNN neighborhoods but changes center selection.",
        "- `voxel_center_knn` keeps local kNN neighborhoods but chooses grid-distributed centers.",
        "- `radius_fps` keeps FPS centers but changes neighborhood construction to a radius query with nearest fallback.",
        "- `random_group` destroys local neighborhoods and is a destructive architecture sanity check.",
        "- These are inference-time grouping perturbations, not retrained architectures.",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = load_config(args.config, args.batch_size, args.split)
    loader = build_loader(cfg, args.split, args.num_workers)

    rows = []
    for mode_idx, mode in enumerate(split_csv(args.group_modes)):
        model = build_model(cfg, args.ckpt)
        patched = patch_group_modules(model, mode, args.radius, args.voxel_grid, args.seed + mode_idx * 10000)
        if patched <= 0:
            raise RuntimeError("no group modules patched")
        for cond_idx, condition in enumerate(split_csv(args.support_conditions)):
            rows.append(
                evaluate(
                    model,
                    loader,
                    cfg.npoints,
                    mode,
                    condition,
                    args.max_batches,
                    args.seed + mode_idx * 1000 + cond_idx,
                )
            )
        del model
        torch.cuda.empty_cache()

    summary = {
        "config": args.config,
        "ckpt": args.ckpt,
        "split": args.split,
        "radius": args.radius,
        "voxel_grid": args.voxel_grid,
        "rows": rows,
    }
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["group_mode", "condition", "acc"])
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row[k] for k in ["group_mode", "condition", "acc"]})
    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(to_md(summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
