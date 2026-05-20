#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MODES = (
    "same_source_outgoing",
    "include_current_negative",
    "same_object_all_views",
    "cross_object_same_action",
)


@dataclass(frozen=True)
class ShapeRecord:
    path: Path
    shape_id: str
    category: str
    source_split: str
    views: np.ndarray


@dataclass(frozen=True)
class Candidate:
    shape_idx: int
    view_idx: int
    edge_idx: int | None = None


@dataclass(frozen=True)
class Trial:
    mode: str
    shape_idx: int
    edge_idx: int
    source_view: int
    target_view: int
    action_id: int
    candidates: tuple[Candidate, ...]
    target_pos: int
    action_prior_pos: int | None


class RunningMetrics:
    def __init__(self) -> None:
        self.n = 0
        self.sum_candidates = 0.0
        self.min_candidates: int | None = None
        self.max_candidates = 0
        self.target_acc = 0.0
        self.current_acc = 0.0
        self.target_margin = 0.0
        self.current_margin = 0.0
        self.target_ties = 0
        self.current_ties = 0
        self.random_expected = 0.0
        self.action_prior_correct = 0.0
        self.action_prior_n = 0
        self.subsampled = 0

    def update(
        self,
        *,
        num_candidates: int,
        target_scores: np.ndarray,
        current_scores: np.ndarray,
        target_pos: int,
        action_prior_pos: int | None,
        tie_eps: float,
        was_subsampled: bool,
    ) -> None:
        self.n += 1
        self.sum_candidates += float(num_candidates)
        self.min_candidates = (
            num_candidates if self.min_candidates is None else min(self.min_candidates, num_candidates)
        )
        self.max_candidates = max(self.max_candidates, num_candidates)
        self.random_expected += 1.0 / float(num_candidates)
        if was_subsampled:
            self.subsampled += 1

        credit, margin, tied = top1_credit_and_margin(target_scores, target_pos, tie_eps)
        self.target_acc += credit
        self.target_margin += margin
        self.target_ties += int(tied > 1)

        credit, margin, tied = top1_credit_and_margin(current_scores, target_pos, tie_eps)
        self.current_acc += credit
        self.current_margin += margin
        self.current_ties += int(tied > 1)

        if action_prior_pos is not None:
            self.action_prior_n += 1
            self.action_prior_correct += float(action_prior_pos == target_pos)

    def summary(self) -> dict[str, Any]:
        denom = max(self.n, 1)
        out: dict[str, Any] = {
            "n": self.n,
            "mean_candidates": self.sum_candidates / denom,
            "min_candidates": int(self.min_candidates or 0),
            "max_candidates": int(self.max_candidates),
            "target_query_oracle_acc": self.target_acc / denom,
            "current_query_shortcut_acc": self.current_acc / denom,
            "random_expected_acc": self.random_expected / denom,
            "target_query_margin_mean": self.target_margin / denom,
            "current_query_margin_mean": self.current_margin / denom,
            "target_query_tie_rate": self.target_ties / denom,
            "current_query_tie_rate": self.current_ties / denom,
            "subsampled_trials": self.subsampled,
        }
        if self.action_prior_n:
            out["action_prior_acc"] = self.action_prior_correct / float(self.action_prior_n)
            out["action_prior_n"] = self.action_prior_n
        else:
            out["action_prior_acc"] = None
            out["action_prior_n"] = 0
        return out


def scalar_str(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def load_manifest_rows(cache_root: Path, split: str, seed: int) -> list[dict[str, Any]]:
    manifest_path = cache_root / "manifest.json"
    if not manifest_path.exists():
        files = sorted(p for p in cache_root.rglob("*.npz") if p.name != "view_graph.npz")
        return [{"cache_path": str(p.relative_to(cache_root))} for p in split_file_list(files, split, seed)]

    manifest = json.loads(manifest_path.read_text())
    rows = list(manifest.get("rows", []))
    if split in {"all", "full", ""}:
        return rows

    matched = [r for r in rows if r.get("source_split", "all") == split]
    if matched:
        return matched

    if all(r.get("source_split", "all") in {"all", ""} for r in rows):
        rng = np.random.default_rng(seed)
        order = np.arange(len(rows))
        rng.shuffle(order)
        shuffled = [rows[int(i)] for i in order]
        n = len(shuffled)
        if split == "train":
            return shuffled[: int(0.9 * n)]
        if split == "val":
            return shuffled[int(0.9 * n): int(0.95 * n)]
        if split == "test":
            return shuffled[int(0.95 * n):]
    return []


def split_file_list(files: list[Path], split: str, seed: int) -> list[Path]:
    if split in {"all", "full", ""}:
        return files
    rng = np.random.default_rng(seed)
    order = np.arange(len(files))
    rng.shuffle(order)
    shuffled = [files[int(i)] for i in order]
    n = len(shuffled)
    if split == "train":
        return shuffled[: int(0.9 * n)]
    if split == "val":
        return shuffled[int(0.9 * n): int(0.95 * n)]
    if split == "test":
        return shuffled[int(0.95 * n):]
    return []


def load_records(cache_root: Path, split: str, max_shapes: int, seed: int) -> list[ShapeRecord]:
    rows = load_manifest_rows(cache_root, split, seed)
    if max_shapes:
        rows = rows[:max_shapes]
    records: list[ShapeRecord] = []
    for row in rows:
        path = cache_root / row["cache_path"]
        data = np.load(path, allow_pickle=True)
        views = np.asarray(data["views"], dtype=np.float32)
        records.append(
            ShapeRecord(
                path=path,
                shape_id=scalar_str(data.get("shape_id"), path.stem),
                category=scalar_str(data.get("category"), path.parent.name),
                source_split=scalar_str(data.get("source_split"), row.get("source_split", "all")),
                views=views,
            )
        )
    return records


def point_descriptor(points: np.ndarray, radial_bins: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    centroid = pts.mean(axis=0)
    centered = pts - centroid[None, :]
    std = centered.std(axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    cov = (centered.T @ centered) / float(max(len(pts) - 1, 1))
    cov_ut = cov[np.triu_indices(3)]
    quant = np.quantile(pts, [0.1, 0.25, 0.5, 0.75, 0.9], axis=0).reshape(-1)
    radius = np.linalg.norm(centered, axis=1)
    r_quant = np.quantile(radius, [0.1, 0.25, 0.5, 0.75, 0.9])
    hist, _ = np.histogram(radius, bins=radial_bins, range=(0.0, 2.0))
    hist = hist.astype(np.float32) / float(max(len(pts), 1))
    return np.concatenate(
        [
            centroid,
            std,
            mins,
            maxs,
            cov_ut,
            quant.astype(np.float32),
            r_quant.astype(np.float32),
            hist,
        ],
        axis=0,
    ).astype(np.float32)


def build_descriptors(records: list[ShapeRecord], radial_bins: int, metric: str) -> np.ndarray:
    desc = np.stack(
        [
            np.stack([point_descriptor(view, radial_bins=radial_bins) for view in rec.views], axis=0)
            for rec in records
        ],
        axis=0,
    ).astype(np.float32)
    flat = desc.reshape(-1, desc.shape[-1])
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    desc = (desc - mean.reshape(1, 1, -1)) / np.maximum(std.reshape(1, 1, -1), 1e-6)
    if metric == "cosine":
        norm = np.linalg.norm(desc, axis=-1, keepdims=True)
        desc = desc / np.maximum(norm, 1e-8)
    return desc.astype(np.float32)


def sample_chamfer_points(records: list[ShapeRecord], num_points: int) -> np.ndarray:
    sampled = []
    for rec in records:
        views = []
        for view in rec.views:
            if len(view) <= num_points:
                views.append(view.astype(np.float32))
            else:
                idx = np.linspace(0, len(view) - 1, num_points).round().astype(np.int64)
                views.append(view[idx].astype(np.float32))
        sampled.append(np.stack(views, axis=0))
    return np.stack(sampled, axis=0)


def descriptor_scores(
    descriptors: np.ndarray,
    query: Candidate,
    candidates: tuple[Candidate, ...],
    metric: str,
) -> np.ndarray:
    q = descriptors[query.shape_idx, query.view_idx]
    cand = np.stack([descriptors[c.shape_idx, c.view_idx] for c in candidates], axis=0)
    if metric == "cosine":
        return (1.0 - cand @ q).astype(np.float64)
    diff = cand - q[None, :]
    return np.sum(diff * diff, axis=1).astype(np.float64)


def chamfer_scores(
    points: np.ndarray,
    query: Candidate,
    candidates: tuple[Candidate, ...],
    metric: str,
    batch_size: int,
) -> np.ndarray:
    q = points[query.shape_idx, query.view_idx].astype(np.float32)
    out: list[np.ndarray] = []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        cand = np.stack([points[c.shape_idx, c.view_idx] for c in batch], axis=0).astype(np.float32)
        d2 = np.sum((cand[:, :, None, :] - q[None, None, :, :]) ** 2, axis=-1)
        if metric == "l1":
            d = np.sqrt(np.maximum(d2, 0.0))
        else:
            d = d2
        out.append((d.min(axis=2).mean(axis=1) + d.min(axis=1).mean(axis=1)).astype(np.float64))
    return np.concatenate(out, axis=0)


def top1_credit_and_margin(scores: np.ndarray, target_pos: int, eps: float) -> tuple[float, float, int]:
    target = float(scores[target_pos])
    best = float(np.min(scores))
    tied = np.flatnonzero(scores <= best + eps)
    credit = (1.0 / float(len(tied))) if target_pos in set(int(i) for i in tied) else 0.0
    if len(scores) <= 1:
        margin = 0.0
    else:
        neg = np.delete(scores, target_pos)
        margin = float(np.min(neg) - target)
    return credit, margin, int(len(tied))


def outgoing_by_source(edges: np.ndarray, num_views: int) -> dict[int, np.ndarray]:
    return {v: np.nonzero(edges[:, 0] == v)[0].astype(np.int64) for v in range(num_views)}


def build_trials(
    *,
    modes: Iterable[str],
    records: list[ShapeRecord],
    edges: np.ndarray,
    action_id: np.ndarray,
    cross_object_scope: str,
) -> list[Trial]:
    num_views = int(edges.max()) + 1
    outgoing = outgoing_by_source(edges, num_views)
    same_action_edges = {
        int(a): np.nonzero(action_id == int(a))[0].astype(np.int64) for a in sorted(set(action_id.tolist()))
    }
    trials: list[Trial] = []
    mode_set = set(modes)
    for shape_idx in range(len(records)):
        for edge_idx, (source_view, target_view) in enumerate(edges.tolist()):
            aid = int(action_id[edge_idx])

            if "same_source_outgoing" in mode_set:
                cand_edges = outgoing[int(source_view)]
                candidates = tuple(
                    Candidate(shape_idx, int(edges[int(e), 1]), int(e)) for e in cand_edges
                )
                target_pos = int(np.nonzero(cand_edges == edge_idx)[0][0])
                action_prior_pos = first_matching_action(candidates, action_id, aid)
                trials.append(
                    Trial(
                        "same_source_outgoing",
                        shape_idx,
                        edge_idx,
                        int(source_view),
                        int(target_view),
                        aid,
                        candidates,
                        target_pos,
                        action_prior_pos,
                    )
                )

            if "include_current_negative" in mode_set:
                cand_edges = outgoing[int(source_view)]
                candidates = (Candidate(shape_idx, int(source_view), None),) + tuple(
                    Candidate(shape_idx, int(edges[int(e), 1]), int(e)) for e in cand_edges
                )
                target_pos = int(np.nonzero(cand_edges == edge_idx)[0][0]) + 1
                action_prior_pos = first_matching_action(candidates, action_id, aid)
                trials.append(
                    Trial(
                        "include_current_negative",
                        shape_idx,
                        edge_idx,
                        int(source_view),
                        int(target_view),
                        aid,
                        candidates,
                        target_pos,
                        action_prior_pos,
                    )
                )

            if "same_object_all_views" in mode_set:
                candidates = tuple(Candidate(shape_idx, v, None) for v in range(num_views))
                trials.append(
                    Trial(
                        "same_object_all_views",
                        shape_idx,
                        edge_idx,
                        int(source_view),
                        int(target_view),
                        aid,
                        candidates,
                        int(target_view),
                        int(target_view),
                    )
                )

            if "cross_object_same_action" in mode_set:
                if cross_object_scope == "action_id":
                    candidates = tuple(
                        Candidate(other_idx, int(edges[int(e), 1]), int(e))
                        for other_idx in range(len(records))
                        for e in same_action_edges[aid]
                    )
                    target_pos = shape_idx * len(same_action_edges[aid]) + int(
                        np.nonzero(same_action_edges[aid] == edge_idx)[0][0]
                    )
                else:
                    candidates = tuple(
                        Candidate(other_idx, int(target_view), edge_idx) for other_idx in range(len(records))
                    )
                    target_pos = shape_idx
                trials.append(
                    Trial(
                        "cross_object_same_action",
                        shape_idx,
                        edge_idx,
                        int(source_view),
                        int(target_view),
                        aid,
                        candidates,
                        target_pos,
                        None,
                    )
                )
    return trials


def first_matching_action(
    candidates: tuple[Candidate, ...],
    action_id: np.ndarray,
    target_action_id: int,
) -> int | None:
    for idx, cand in enumerate(candidates):
        if cand.edge_idx is not None and int(action_id[cand.edge_idx]) == target_action_id:
            return idx
    return None


def maybe_subsample_candidates(
    trial: Trial,
    max_candidates: int,
    rng: np.random.Generator,
) -> tuple[tuple[Candidate, ...], int, int | None, bool]:
    candidates = trial.candidates
    if max_candidates <= 0 or len(candidates) <= max_candidates:
        return candidates, trial.target_pos, trial.action_prior_pos, False
    if max_candidates < 2:
        raise ValueError("--chamfer-max-candidates must be 0 or >=2")
    neg_idx = [i for i in range(len(candidates)) if i != trial.target_pos]
    keep_neg = rng.choice(np.asarray(neg_idx, dtype=np.int64), size=max_candidates - 1, replace=False)
    keep = np.concatenate([np.asarray([trial.target_pos], dtype=np.int64), keep_neg])
    keep.sort()
    old_to_new = {int(old): new for new, old in enumerate(keep.tolist())}
    new_candidates = tuple(candidates[int(i)] for i in keep)
    new_target_pos = old_to_new[trial.target_pos]
    new_action_prior_pos = (
        old_to_new[trial.action_prior_pos]
        if trial.action_prior_pos is not None and trial.action_prior_pos in old_to_new
        else None
    )
    return new_candidates, new_target_pos, new_action_prior_pos, True


def evaluate_descriptor(
    trials: list[Trial],
    descriptors: np.ndarray,
    metric: str,
    tie_eps: float,
) -> dict[str, Any]:
    metrics = {mode: RunningMetrics() for mode in MODES}
    for trial in trials:
        target_query = Candidate(trial.shape_idx, trial.target_view, trial.edge_idx)
        current_query = Candidate(trial.shape_idx, trial.source_view, None)
        target_scores = descriptor_scores(descriptors, target_query, trial.candidates, metric)
        current_scores = descriptor_scores(descriptors, current_query, trial.candidates, metric)
        metrics[trial.mode].update(
            num_candidates=len(trial.candidates),
            target_scores=target_scores,
            current_scores=current_scores,
            target_pos=trial.target_pos,
            action_prior_pos=trial.action_prior_pos,
            tie_eps=tie_eps,
            was_subsampled=False,
        )
    return {mode: m.summary() for mode, m in metrics.items() if m.n}


def evaluate_chamfer(
    trials: list[Trial],
    points: np.ndarray,
    metric: str,
    tie_eps: float,
    max_candidates: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    metrics = {mode: RunningMetrics() for mode in MODES}
    rng = np.random.default_rng(seed)
    for trial in trials:
        candidates, target_pos, action_prior_pos, was_subsampled = maybe_subsample_candidates(
            trial, max_candidates=max_candidates, rng=rng
        )
        target_query = Candidate(trial.shape_idx, trial.target_view, trial.edge_idx)
        current_query = Candidate(trial.shape_idx, trial.source_view, None)
        target_scores = chamfer_scores(points, target_query, candidates, metric, batch_size)
        current_scores = chamfer_scores(points, current_query, candidates, metric, batch_size)
        metrics[trial.mode].update(
            num_candidates=len(candidates),
            target_scores=target_scores,
            current_scores=current_scores,
            target_pos=target_pos,
            action_prior_pos=action_prior_pos,
            tie_eps=tie_eps,
            was_subsampled=was_subsampled,
        )
    return {mode: m.summary() for mode, m in metrics.items() if m.n}


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Raw Geometry Oracle",
        "",
        f"- cache: `{payload['config']['cache_root']}`",
        f"- split: `{payload['config']['split']}`",
        f"- shapes: `{payload['config']['num_shapes']}`",
        f"- views/edges: `{payload['config']['num_views']}` / `{payload['config']['num_edges']}`",
        "",
        "| distance | mode | target_oracle | current_shortcut | random | action_prior | margin_target | margin_current | candidates | n |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for distance, by_mode in payload["results"].items():
        for mode, row in by_mode.items():
            action_prior = row.get("action_prior_acc")
            action_text = "nan" if action_prior is None else f"{float(action_prior):.4f}"
            lines.append(
                f"| {distance} | {mode} | "
                f"{float(row['target_query_oracle_acc']):.4f} | "
                f"{float(row['current_query_shortcut_acc']):.4f} | "
                f"{float(row['random_expected_acc']):.4f} | "
                f"{action_text} | "
                f"{float(row['target_query_margin_mean']):.4f} | "
                f"{float(row['current_query_margin_mean']):.4f} | "
                f"{float(row['mean_candidates']):.1f} | {int(row['n'])} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_modes(raw: str) -> list[str]:
    if raw == "all":
        return list(MODES)
    modes = [m.strip() for m in raw.split(",") if m.strip()]
    unknown = sorted(set(modes) - set(MODES))
    if unknown:
        raise ValueError(f"unknown mode(s): {', '.join(unknown)}")
    return modes


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Raw-geometry oracle for view-action candidate retrieval. "
            "Uses cached partial views only; no checkpoint or training is required."
        )
    )
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--split", default="all")
    ap.add_argument("--max-shapes", type=int, default=200)
    ap.add_argument("--modes", default="all", help="Comma-separated modes or 'all'.")
    ap.add_argument("--candidate-mode", default="", help="Compatibility alias for --modes.")
    ap.add_argument("--distance", default="descriptor", choices=["descriptor", "chamfer", "both"])
    ap.add_argument("--descriptor-metric", default="l2", choices=["l2", "cosine"])
    ap.add_argument("--descriptor-radial-bins", type=int, default=8)
    ap.add_argument("--chamfer-metric", default="l2", choices=["l1", "l2"])
    ap.add_argument("--chamfer-points", type=int, default=128)
    ap.add_argument("--chamfer-max-candidates", type=int, default=64)
    ap.add_argument("--chamfer-batch-size", type=int, default=64)
    ap.add_argument("--cross-object-scope", default="exact_edge", choices=["exact_edge", "action_id"])
    ap.add_argument("--cross-object-negatives", type=int, default=0,
                    help="Compatibility argument; raw descriptor cross-object keeps all candidates.")
    ap.add_argument("--tie-eps", type=float, default=1e-9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", default="")
    args = ap.parse_args()

    if args.candidate_mode:
        args.modes = args.candidate_mode

    cache_root = Path(args.cache_root)
    graph = np.load(cache_root / "view_graph.npz")
    edges = graph["edges"].astype(np.int64)
    action_id = graph["action_id"].astype(np.int64)
    modes = parse_modes(args.modes)
    records = load_records(cache_root, split=args.split, max_shapes=args.max_shapes, seed=args.seed)
    if not records:
        raise SystemExit(f"no cache records matched split={args.split!r} under {cache_root}")

    trials = build_trials(
        modes=modes,
        records=records,
        edges=edges,
        action_id=action_id,
        cross_object_scope=args.cross_object_scope,
    )
    results: dict[str, Any] = {}
    if args.distance in {"descriptor", "both"}:
        descriptors = build_descriptors(
            records,
            radial_bins=args.descriptor_radial_bins,
            metric=args.descriptor_metric,
        )
        key = f"descriptor_{args.descriptor_metric}"
        results[key] = evaluate_descriptor(
            trials,
            descriptors=descriptors,
            metric=args.descriptor_metric,
            tie_eps=args.tie_eps,
        )
    if args.distance in {"chamfer", "both"}:
        points = sample_chamfer_points(records, num_points=args.chamfer_points)
        key = f"chamfer_{args.chamfer_metric}"
        results[key] = evaluate_chamfer(
            trials,
            points=points,
            metric=args.chamfer_metric,
            tie_eps=args.tie_eps,
            max_candidates=args.chamfer_max_candidates,
            batch_size=args.chamfer_batch_size,
            seed=args.seed,
        )

    out = {
        "config": {
            "cache_root": str(cache_root),
            "split": args.split,
            "max_shapes": args.max_shapes,
            "num_shapes": len(records),
            "num_views": int(edges.max()) + 1,
            "num_edges": int(len(edges)),
            "modes": modes,
            "distance": args.distance,
            "descriptor_metric": args.descriptor_metric,
            "descriptor_radial_bins": args.descriptor_radial_bins,
            "chamfer_metric": args.chamfer_metric,
            "chamfer_points": args.chamfer_points,
            "chamfer_max_candidates": args.chamfer_max_candidates,
            "cross_object_scope": args.cross_object_scope,
            "tie_eps": args.tie_eps,
            "seed": args.seed,
        },
        "notes": {
            "target_query_oracle_acc": "Nearest-candidate accuracy when the query is the actual target view geometry.",
            "current_query_shortcut_acc": "Nearest-candidate accuracy when the query is the current/source view geometry.",
            "action_prior_acc": "Uses graph/action labels when the candidate set exposes a unique action-derived candidate; null when action cannot distinguish candidates.",
            "random_expected_acc": "Mean expected top-1 accuracy from uniform random choice over the evaluated candidate set.",
        },
        "results": results,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    if args.out_md:
        write_markdown(Path(args.out_md), out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
