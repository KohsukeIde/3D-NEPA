#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from knn_cuda import KNN
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "models"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from dataset import PartNormalDataset  # noqa: E402
from eval_shapenetpart_support_stress import (  # noqa: E402
    build_model,
    evaluate_condition,
    to_md as support_to_md,
)
from pt import fps  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser("ShapeNetPart grouping/patchization ablation")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--model", default="pt")
    p.add_argument("--model_name", default="PointGPT_S", choices=["PointGPT_S", "PointGPT_B", "PointGPT_L"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--npoint", type=int, default=2048)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--normal", action="store_true", default=False)
    p.add_argument("--radius", type=float, default=0.22)
    p.add_argument("--voxel_grid", type=int, default=6)
    # build_model() reuses the training-time model factory, which expects the
    # base grouping arguments even though this script later patches grouping
    # modes per condition.
    p.add_argument("--group_mode", default="fps_knn",
                   choices=["fps_knn", "random_center_knn", "voxel_center_knn", "radius_fps", "random_group"])
    p.add_argument("--group_radius", type=float, default=0.22)
    p.add_argument("--group_voxel_grid", type=int, default=6)
    p.add_argument(
        "--group_modes",
        default="fps_knn,random_center_knn,voxel_center_knn,radius_fps,random_group",
    )
    p.add_argument(
        "--conditions",
        default="clean,random_keep20,structured_keep20,part_drop_largest,part_keep20_per_part,xyz_zero",
    )
    p.add_argument("--output_json", default="")
    p.add_argument("--output_csv", default="")
    p.add_argument("--output_md", default="")
    return p.parse_args()


def split_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


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
    batch_size, _, _ = xyz.shape
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
            sub = fps(chosen_pts.unsqueeze(0), num_group).squeeze(0)
            centers.append(sub)
        else:
            fill = fps(pts.unsqueeze(0), num_group - chosen_idx.numel()).squeeze(0)
            centers.append(torch.cat([pts[chosen_idx], fill], dim=0))
    return torch.stack(centers, dim=0)


def radius_indices(xyz: torch.Tensor, center: torch.Tensor, group_size: int, radius: float) -> torch.Tensor:
    dist = torch.cdist(center, xyz)
    nearest = torch.argsort(dist, dim=-1)
    idx = nearest[:, :, :group_size].clone()
    within = dist <= radius
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


def identity_sorting(module, xyz: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    batch_size, _, _ = xyz.shape
    idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1) * module.num_group
    seq = torch.arange(module.num_group, device=xyz.device).view(1, -1)
    return (seq + idx_base).reshape(-1)


def make_group_forward(group_mode: str, radius: float, voxel_grid: int, seed: int):
    def forward(self, xyz):
        batch_size, num_points, _ = xyz.shape
        generator = None
        if group_mode in {"random_center_knn", "random_group"}:
            generator = torch.Generator(device=xyz.device)
            generator.manual_seed(seed + int(getattr(self, "_ablation_calls", 0)))
            self._ablation_calls = int(getattr(self, "_ablation_calls", 0)) + 1

        if group_mode == "fps_knn":
            center = fps(xyz, self.num_group)
            _, idx = self.knn(xyz, center)
        elif group_mode == "random_center_knn":
            center = select_random_centers(xyz, self.num_group, generator)
            _, idx = self.knn(xyz, center)
        elif group_mode == "voxel_center_knn":
            center = select_voxel_centers(xyz, self.num_group, voxel_grid)
            _, idx = self.knn(xyz, center)
        elif group_mode == "radius_fps":
            center = fps(xyz, self.num_group)
            idx = radius_indices(xyz, center, self.group_size, radius)
        elif group_mode == "random_group":
            center = select_random_centers(xyz, self.num_group, generator)
            idx = random_group_indices(xyz, self.num_group, self.group_size, generator)
        else:
            raise ValueError(f"Unsupported group mode: {group_mode}")

        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        neighborhood = batched_gather_points(xyz, idx).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)

        if hasattr(self, "simplied_morton_sorting") and group_mode != "random_group":
            sorted_indices = self.simplied_morton_sorting(xyz, center)
        else:
            sorted_indices = identity_sorting(self, xyz, center)
        neighborhood = neighborhood.reshape(batch_size * self.num_group, self.group_size, 3)[sorted_indices, :, :]
        neighborhood = neighborhood.reshape(batch_size, self.num_group, self.group_size, 3).contiguous()
        center = center.reshape(batch_size * self.num_group, 3)[sorted_indices, :]
        center = center.reshape(batch_size, self.num_group, 3).contiguous()
        return neighborhood, center

    return forward


def patch_group_modules(model, group_mode: str, radius: float, voxel_grid: int, seed: int):
    patched = 0
    for module in model.modules():
        if all(hasattr(module, attr) for attr in ("num_group", "group_size", "knn")):
            module._ablation_calls = 0
            if not isinstance(module.knn, KNN):
                module.knn = KNN(k=module.group_size, transpose_mode=True)
            module.forward = MethodType(make_group_forward(group_mode, radius, voxel_grid, seed), module)
            patched += 1
    if patched == 0:
        raise RuntimeError("No PointGPT Group-like modules were patched")


def row_to_md(summary):
    lines = [
        "# ShapeNetPart Grouping Ablation",
        "",
        f"- ckpt: `{summary['ckpt']}`",
        f"- root: `{summary['root']}`",
        "",
        "| group mode | condition | accuracy | class avg IoU | instance avg IoU |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| `{row['group_mode']}` | `{row['condition']}` | "
            f"`{row['accuracy']:.4f}` | `{row['class_avg_iou']:.4f}` | "
            f"`{row['instance_avg_iou']:.4f}` |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- The checkpoint/head are fixed. Only grouping center/neighborhood construction is changed at inference time.")
    lines.append("- `random_group` destroys local neighborhoods and is a destructive sanity check.")
    lines.append("- This is a diagnostic ablation, not a retrained architecture comparison.")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = PartNormalDataset(root=args.root, npoints=args.npoint, split="test", normal_channel=args.normal)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = build_model(args)

    rows = []
    for mode_idx, group_mode in enumerate(split_csv(args.group_modes)):
        patch_group_modules(model, group_mode, args.radius, args.voxel_grid, args.seed + 1000 * mode_idx)
        for condition in split_csv(args.conditions):
            result = evaluate_condition(model, loader, condition, args)
            result["group_mode"] = group_mode
            rows.append(result)

    summary = {
        "ckpt": args.ckpt,
        "root": args.root,
        "group_modes": split_csv(args.group_modes),
        "conditions": split_csv(args.conditions),
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
            writer = csv.DictWriter(
                f,
                fieldnames=["group_mode", "condition", "accuracy", "class_avg_iou", "instance_avg_iou"],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row[k] for k in writer.fieldnames})
    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(row_to_md(summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
