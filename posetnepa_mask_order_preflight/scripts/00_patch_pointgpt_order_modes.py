#!/usr/bin/env python3
"""Patch PointGPT/models/PointGPT.py with geometry-induced order modes.

This patch is intentionally small: it only extends Group.build_sorted_indices().
It does not change grouping, the transformer, or the NEPA loss.

Run from the 3D-NEPA repo root:
    python pointnepa_poset_preflight/scripts/00_patch_pointgpt_order_modes.py --apply
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ORDER_METHODS = r'''
    def fixed_random_sorting(self, xyz, center):
        batch_size, _, _ = center.shape
        idx = torch.arange(self.num_group, device=center.device, dtype=torch.long)
        mixed = torch.remainder(idx * 1103515245 + 12345, 2147483647)
        scores = torch.remainder(
            mixed * mixed + idx * 2654435761 + 1013904223, 2147483647)
        order = torch.argsort(scores * self.num_group + idx, dim=0).view(
            1, -1).expand(batch_size, -1)
        return self._flatten_order(order)

    def _flatten_order(self, order):
        batch_size = order.shape[0]
        idx_base = torch.arange(0, batch_size, device=order.device).view(-1, 1) * self.num_group
        return (order + idx_base).reshape(-1)

    def axis_sorting(self, xyz, center, axis=0, descending=False):
        values = center[..., int(axis)]
        order = torch.argsort(values, dim=1, descending=bool(descending))
        return self._flatten_order(order)

    def radial_sorting(self, xyz, center):
        centroid = center.mean(dim=1, keepdim=True)
        radius = torch.norm(center - centroid, dim=-1)
        order = torch.argsort(radius, dim=1, descending=False)
        return self._flatten_order(order)

    def farthest_greedy_sorting(self, xyz, center):
        # Deterministic farthest traversal over patch centers.
        batch_size, num_group, _ = center.shape
        orders = []
        dist_all = torch.cdist(center, center)
        centroid = center.mean(dim=1, keepdim=True)
        start = torch.argmax(torch.norm(center - centroid, dim=-1), dim=1)
        for b in range(batch_size):
            selected = [int(start[b].item())]
            remaining = torch.ones(num_group, dtype=torch.bool, device=center.device)
            remaining[selected[0]] = False
            min_dist = dist_all[b, selected[0]].clone()
            for _ in range(num_group - 1):
                masked = min_dist.masked_fill(~remaining, -1.0)
                nxt = int(torch.argmax(masked).item())
                selected.append(nxt)
                remaining[nxt] = False
                min_dist = torch.minimum(min_dist, dist_all[b, nxt])
            orders.append(torch.tensor(selected, dtype=torch.long, device=center.device))
        return self._flatten_order(torch.stack(orders, dim=0))

    def _knn_adjacency(self, center, k=6, weighted=False):
        batch_size, num_group, _ = center.shape
        dist = torch.cdist(center, center)
        eye = torch.eye(num_group, device=center.device, dtype=torch.bool).unsqueeze(0)
        dist_masked = dist.masked_fill(eye, float('inf'))
        nn = torch.topk(dist_masked, k=min(int(k), num_group - 1), dim=-1, largest=False).indices
        adj = torch.zeros(batch_size, num_group, num_group, device=center.device, dtype=torch.bool)
        adj.scatter_(2, nn, True)
        adj = adj | adj.transpose(1, 2)
        if weighted:
            weights = dist.masked_fill(~adj, float('inf'))
            return adj, weights
        return adj, dist

    def bfs_shell_sorting(self, xyz, center, k=6):
        batch_size, num_group, _ = center.shape
        adj, _ = self._knn_adjacency(center, k=k, weighted=False)
        centroid = center.mean(dim=1, keepdim=True)
        seed = torch.argmax(torch.norm(center - centroid, dim=-1), dim=1)
        orders = []
        secondary = torch.norm(center - centroid, dim=-1)
        for b in range(batch_size):
            dist = torch.full((num_group,), 10_000, dtype=torch.long, device=center.device)
            dist[seed[b]] = 0
            frontier = torch.zeros(num_group, dtype=torch.bool, device=center.device)
            frontier[seed[b]] = True
            for depth in range(1, num_group + 1):
                nbr = (adj[b][frontier].any(dim=0) if frontier.any() else torch.zeros_like(frontier))
                nbr = nbr & (dist == 10_000)
                if not nbr.any():
                    break
                dist[nbr] = depth
                frontier = nbr
            score = dist.float() * 1000.0 + secondary[b]
            orders.append(torch.argsort(score, dim=0))
        return self._flatten_order(torch.stack(orders, dim=0))

    def geodesic_shell_sorting(self, xyz, center, k=6):
        # Weighted shortest-path shells on the patch-center kNN graph.
        batch_size, num_group, _ = center.shape
        _, weights = self._knn_adjacency(center, k=k, weighted=True)
        centroid = center.mean(dim=1, keepdim=True)
        seed = torch.argmax(torch.norm(center - centroid, dim=-1), dim=1)
        D = weights.clone()
        eye = torch.eye(num_group, device=center.device, dtype=torch.bool).unsqueeze(0)
        D = D.masked_fill(eye, 0.0)
        # Batched Floyd-Warshall is acceptable for G=64 and pre-flight screening.
        for kk in range(num_group):
            D = torch.minimum(D, D[:, :, kk:kk+1] + D[:, kk:kk+1, :])
        geo = D[torch.arange(batch_size, device=center.device), seed]
        geo = torch.nan_to_num(geo, nan=1e6, posinf=1e6, neginf=1e6)
        return self._flatten_order(torch.argsort(geo, dim=1))

    def diffusion_shell_sorting(self, xyz, center, k=8, n_eigs=8, diffusion_time=2):
        # Diffusion-map distance from a deterministic seed. This is the default
        # PosetNEPA-Lite order: smoother than BFS and less boundary-sensitive than
        # shortest-path shells on noisy patch graphs.
        batch_size, num_group, _ = center.shape
        dist = torch.cdist(center, center)
        eye = torch.eye(num_group, device=center.device, dtype=torch.bool).unsqueeze(0)
        dist_noeye = dist.masked_fill(eye, float('inf'))
        kth = torch.topk(dist_noeye, k=min(int(k), num_group - 1), dim=-1, largest=False).values[..., -1]
        sigma = torch.clamp(kth.median(dim=1).values.view(batch_size, 1, 1), min=1e-6)
        nn = torch.topk(dist_noeye, k=min(int(k), num_group - 1), dim=-1, largest=False).indices
        mask = torch.zeros(batch_size, num_group, num_group, device=center.device, dtype=torch.bool)
        mask.scatter_(2, nn, True)
        mask = mask | mask.transpose(1, 2)
        W = torch.exp(-(dist ** 2) / (sigma ** 2)).masked_fill(~mask, 0.0)
        W = W.masked_fill(eye, 0.0)
        deg = torch.clamp(W.sum(dim=-1), min=1e-6)
        D_inv_sqrt = deg.rsqrt()
        K = W * D_inv_sqrt.unsqueeze(2) * D_inv_sqrt.unsqueeze(1)
        centroid = center.mean(dim=1, keepdim=True)
        seed = torch.argmax(torch.norm(center - centroid, dim=-1), dim=1)
        try:
            evals, evecs = torch.linalg.eigh(K)
            # Largest eigenvalues carry slow diffusion geometry.
            idx = torch.argsort(evals, dim=1, descending=True)[:, 1: min(int(n_eigs) + 1, num_group)]
            eidx = idx.unsqueeze(1).expand(-1, num_group, -1)
            coords = torch.gather(evecs, 2, eidx)
            lamb = torch.gather(evals, 1, idx).clamp(min=0.0) ** int(diffusion_time)
            coords = coords * lamb.unsqueeze(1)
            seed_coords = coords[torch.arange(batch_size, device=center.device), seed].unsqueeze(1)
            dmap = torch.norm(coords - seed_coords, dim=-1)
        except Exception:
            # Fallback to radial if eigensolver fails on pathological input.
            seed_center = center[torch.arange(batch_size, device=center.device), seed].unsqueeze(1)
            dmap = torch.norm(center - seed_center, dim=-1)
        return self._flatten_order(torch.argsort(dmap, dim=1))
'''

FIXED_RANDOM_METHOD = r'''
    def fixed_random_sorting(self, xyz, center):
        batch_size, _, _ = center.shape
        idx = torch.arange(self.num_group, device=center.device, dtype=torch.long)
        mixed = torch.remainder(idx * 1103515245 + 12345, 2147483647)
        scores = torch.remainder(
            mixed * mixed + idx * 2654435761 + 1013904223, 2147483647)
        order = torch.argsort(scores * self.num_group + idx, dim=0).view(
            1, -1).expand(batch_size, -1)
        return self._flatten_order(order)
'''

FIXED_RANDOM_BRANCH = r'''        if self.order_mode in {"fixed_random", "stable_random"}:
            return self.fixed_random_sorting(xyz, center)
'''

BRANCHES = r'''
        if self.order_mode == "axis_x":
            return self.axis_sorting(xyz, center, axis=0)
        if self.order_mode == "axis_y":
            return self.axis_sorting(xyz, center, axis=1)
        if self.order_mode == "axis_z":
            return self.axis_sorting(xyz, center, axis=2)
        if self.order_mode == "radial":
            return self.radial_sorting(xyz, center)
        if self.order_mode == "farthest_greedy":
            return self.farthest_greedy_sorting(xyz, center)
        if self.order_mode == "bfs_shell":
            return self.bfs_shell_sorting(xyz, center)
        if self.order_mode == "geodesic_shell":
            return self.geodesic_shell_sorting(xyz, center)
        if self.order_mode == "diffusion_shell":
            return self.diffusion_shell_sorting(xyz, center)
'''


def patch_text(text: str) -> str:
    has_poset_patch = "def diffusion_shell_sorting" in text and "order_mode == \"diffusion_shell\"" in text
    fixed_random_branch = 'order_mode in {"fixed_random", "stable_random"}'
    if has_poset_patch and "def fixed_random_sorting" in text and fixed_random_branch in text:
        return text

    if not has_poset_patch:
        marker = "    def build_sorted_indices(self, xyz, center):"
        if marker not in text:
            raise RuntimeError("Could not find Group.build_sorted_indices marker in PointGPT.py")
        text = text.replace(marker, ORDER_METHODS + "\n" + marker, 1)
    elif "def fixed_random_sorting" not in text:
        method_marker = "    def identity_sorting(self, xyz, center):"
        if method_marker not in text:
            raise RuntimeError("Could not find Group.identity_sorting marker in PointGPT.py")
        text = text.replace(method_marker, FIXED_RANDOM_METHOD + "\n" + method_marker, 1)

    if fixed_random_branch not in text:
        random_marker = '        if self.order_mode == "random":\n            return self.random_sorting(xyz, center)\n'
        if random_marker not in text:
            raise RuntimeError("Could not find random order branch in PointGPT.py")
        text = text.replace(random_marker, random_marker + FIXED_RANDOM_BRANCH, 1)

    branch_marker = '        if self.order_mode in {"identity", "none"}:\n            return self.identity_sorting(xyz, center)\n'
    if (not has_poset_patch) and branch_marker in text:
        text = text.replace(branch_marker, branch_marker + BRANCHES, 1)
    elif not has_poset_patch:
        # Robust fallback: insert branches immediately before the unsupported-order raise.
        raise_marker = '        raise ValueError(f"Unsupported PointGPT order_mode: {self.order_mode}")'
        if raise_marker not in text:
            raise RuntimeError("Could not find unsupported order raise marker in PointGPT.py")
        text = text.replace(raise_marker, BRANCHES + raise_marker, 1)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--target", default="PointGPT/models/PointGPT.py")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    path = repo / args.target
    if not path.exists():
        raise SystemExit(f"[error] target not found: {path}")
    original = path.read_text()
    patched = patch_text(original)
    if original == patched:
        print(f"[ok] already patched: {path}")
        return
    backup = path.with_suffix(path.suffix + ".poset_preflight.bak")
    if args.apply:
        if not backup.exists():
            backup.write_text(original)
        path.write_text(patched)
        print(f"[done] patched {path}")
        print(f"[backup] {backup}")
    else:
        print("[dry-run] patch would modify:", path)
        print("[dry-run] re-run with --apply to write changes")


if __name__ == "__main__":
    main()
