"""DDA quality metrics for pointcloud-derived ray pools.

This script compares, per cached sample (.npz), the mesh-based ray answers
(`ray_*_pool`) against the pointcloud-derived DDA answers (`ray_*_pc_pool`).

It reports:
  - hit agreement / confusion matrix (TP/FP/FN/TN)
  - depth error |t_pc - t_mesh| on rays where both hit
  - normal cosine similarity on rays where both hit

Typical usage:

  python -m nepa3d.analysis.dda_metrics \
    --cache_root data/modelnet40_cache_v1 --split test \
    --out_csv results/dda_test.csv --plot_dir results/figs

Notes:
  - The cache must contain `ray_hit_pc_pool`, `ray_t_pc_pool`, `ray_n_pc_pool`.
  - For large splits, store only a reservoir sample of per-ray errors.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..data.modelnet40_index import list_npz, label_from_path


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, tp: int, fp: int, fn: int, tn: int) -> None:
        self.tp += int(tp)
        self.fp += int(fp)
        self.fn += int(fn)
        self.tn += int(tn)

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    def precision(self) -> float:
        d = self.tp + self.fp
        return float(self.tp) / d if d > 0 else 0.0

    def recall(self) -> float:
        d = self.tp + self.fn
        return float(self.tp) / d if d > 0 else 0.0

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    def accuracy(self) -> float:
        n = self.n
        return float(self.tp + self.tn) / n if n > 0 else 0.0


class Reservoir:
    """Reservoir sample for streaming quantiles."""

    def __init__(self, k: int, seed: int = 0):
        self.k = int(k)
        self.rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
        self.buf = np.empty((self.k,), dtype=np.float32)
        self.n_seen = 0
        self.n_filled = 0

    def add_many(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        for v in x:
            if self.n_filled < self.k:
                self.buf[self.n_filled] = v
                self.n_filled += 1
            else:
                j = self.rng.randint(0, self.n_seen + 1)
                if j < self.k:
                    self.buf[j] = v
            self.n_seen += 1

    def values(self) -> np.ndarray:
        return self.buf[: self.n_filled]


def _safe_normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, eps)


def compute_file_metrics(
    npz_path: str,
    ray_subsample: int = 0,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[str, Confusion, np.ndarray, np.ndarray]:
    """Return (class_name, confusion, abs_depth_errors, normal_cos)."""
    d = np.load(npz_path)
    if "ray_hit_pc_pool" not in d.files:
        raise KeyError("ray_hit_pc_pool missing")

    hit_m = d["ray_hit_pool"].astype(np.float32)
    t_m = d["ray_t_pool"].astype(np.float32)
    n_m = d["ray_n_pool"].astype(np.float32)

    hit_p = d["ray_hit_pc_pool"].astype(np.float32)
    t_p = d["ray_t_pc_pool"].astype(np.float32)
    n_p = d["ray_n_pc_pool"].astype(np.float32)

    assert hit_m.shape == hit_p.shape

    m = hit_m.shape[0]
    idx = np.arange(m)
    if ray_subsample and ray_subsample > 0 and ray_subsample < m:
        if rng is None:
            rng = np.random.RandomState(0)
        idx = rng.choice(idx, size=int(ray_subsample), replace=False)

    hm = hit_m[idx] > 0.5
    hp = hit_p[idx] > 0.5

    tp = np.sum(hm & hp)
    fp = np.sum(~hm & hp)
    fn = np.sum(hm & ~hp)
    tn = np.sum(~hm & ~hp)
    conf = Confusion(int(tp), int(fp), int(fn), int(tn))

    both = hm & hp
    if np.any(both):
        err_abs = np.abs(t_p[idx][both] - t_m[idx][both]).astype(np.float32)

        nm = n_m[idx][both]
        np_ = n_p[idx][both]
        nm = _safe_normalize(nm)
        np_ = _safe_normalize(np_)
        cos = np.sum(nm * np_, axis=1).astype(np.float32)
    else:
        err_abs = np.zeros((0,), dtype=np.float32)
        cos = np.zeros((0,), dtype=np.float32)

    return label_from_path(npz_path), conf, err_abs, cos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "test"])
    ap.add_argument("--max_files", type=int, default=0, help="debug: limit number of files")
    ap.add_argument("--ray_subsample", type=int, default=0, help="subsample rays per file")
    ap.add_argument("--reservoir", type=int, default=200000, help="reservoir size for depth errors")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", type=str, default="")
    ap.add_argument("--plot_dir", type=str, default="")
    args = ap.parse_args()

    paths = list_npz(args.cache_root, args.split)
    if args.max_files and args.max_files > 0:
        paths = paths[: int(args.max_files)]

    rng = np.random.RandomState(int(args.seed) & 0xFFFFFFFF)

    conf_all = Confusion()
    conf_per_cls: Dict[str, Confusion] = {}

    depth_res = Reservoir(args.reservoir, seed=args.seed)
    cos_res = Reservoir(args.reservoir, seed=args.seed + 12345)

    skipped = 0
    for p in paths:
        try:
            cls, conf, err_abs, cos = compute_file_metrics(
                p, ray_subsample=args.ray_subsample, rng=rng
            )
        except Exception:
            skipped += 1
            continue

        conf_all.add(conf.tp, conf.fp, conf.fn, conf.tn)
        if cls not in conf_per_cls:
            conf_per_cls[cls] = Confusion()
        conf_per_cls[cls].add(conf.tp, conf.fp, conf.fn, conf.tn)

        depth_res.add_many(err_abs)
        cos_res.add_many(cos)

    depth_vals = depth_res.values()
    cos_vals = cos_res.values()

    def q(a, qq):
        return float(np.quantile(a, qq)) if a.size > 0 else float("nan")

    summary = {
        "split": args.split,
        "num_files": len(paths),
        "skipped": skipped,
        "rays": conf_all.n,
        "hit_acc": conf_all.accuracy(),
        "precision": conf_all.precision(),
        "recall": conf_all.recall(),
        "f1": conf_all.f1(),
        "depth_abs_mean": float(depth_vals.mean()) if depth_vals.size > 0 else float("nan"),
        "depth_abs_median": q(depth_vals, 0.5),
        "depth_abs_p90": q(depth_vals, 0.9),
        "depth_abs_p99": q(depth_vals, 0.99),
        "normal_cos_mean": float(cos_vals.mean()) if cos_vals.size > 0 else float("nan"),
        "normal_cos_median": q(cos_vals, 0.5),
    }

    print("=== DDA quality summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    print("confusion: TP=%d FP=%d FN=%d TN=%d" % (conf_all.tp, conf_all.fp, conf_all.fn, conf_all.tn))
    print("(depth/normal stats are from a reservoir sample; increase --reservoir for more accuracy)")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w") as f:
            f.write(",".join([
                "class","tp","fp","fn","tn","hit_acc","precision","recall","f1"\
            ]) + "\n")
            for cls in sorted(conf_per_cls.keys()):
                c = conf_per_cls[cls]
                f.write(
                    f"{cls},{c.tp},{c.fp},{c.fn},{c.tn},{c.accuracy():.6f},{c.precision():.6f},{c.recall():.6f},{c.f1():.6f}\n"
                )
        print(f"wrote per-class CSV: {args.out_csv}")

    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        try:
            import matplotlib.pyplot as plt

            if depth_vals.size > 0:
                # Histogram + CDF of abs depth error
                # Choose a robust max for x-axis based on quantile
                xmax = q(depth_vals, 0.99)
                if not np.isfinite(xmax) or xmax <= 0:
                    xmax = float(depth_vals.max()) if depth_vals.size > 0 else 1.0
                xmax = max(xmax, 1e-3)
                bins = np.linspace(0.0, xmax, 60)
                hist, edges = np.histogram(depth_vals, bins=bins, density=True)
                cdf = np.cumsum(hist) * (edges[1] - edges[0])

                plt.figure()
                plt.plot(edges[1:], cdf)
                plt.xlabel("|t_pc - t_mesh| (abs depth error)")
                plt.ylabel("CDF")
                plt.title(f"DDA depth error CDF ({args.split})")
                out = os.path.join(args.plot_dir, f"dda_depth_cdf_{args.split}.png")
                plt.savefig(out, dpi=200, bbox_inches="tight")
                plt.close()
                print(f"wrote plot: {out}")

            if cos_vals.size > 0:
                plt.figure()
                plt.hist(cos_vals, bins=60, density=True)
                plt.xlabel("cos(n_pc, n_mesh)")
                plt.ylabel("density")
                plt.title(f"DDA normal cosine hist ({args.split})")
                out = os.path.join(args.plot_dir, f"dda_normal_cos_hist_{args.split}.png")
                plt.savefig(out, dpi=200, bbox_inches="tight")
                plt.close()
                print(f"wrote plot: {out}")

        except Exception as e:
            print(f"plot failed: {e}")


if __name__ == "__main__":
    main()
