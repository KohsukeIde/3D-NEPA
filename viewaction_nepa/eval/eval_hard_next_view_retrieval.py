#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from viewaction_nepa.data.viewaction_dataset import ViewActionDataset
from viewaction_nepa.eval.common import encode_all_views, load_model


INFERENCE_VARIANTS = ["action", "no_action", "shuffled_action", "action_only"]
CHECKPOINT_VARIANTS = ["action", "no_action", "shuffled_action", "action_only"]
HARD_MODES = [
    "same_object_all_views",
    "same_source_outgoing",
    "include_current_negative",
    "cross_object_same_action",
]


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    path: Path


class RetrievalMeter:
    def __init__(self) -> None:
        self.top1 = 0
        self.top3 = 0
        self.top5 = 0
        self.mrr = 0.0
        self.current_selected = 0
        self.current_candidate = 0
        self.n = 0

    def update(self, rank: int, selected_is_current: bool, has_current: bool) -> None:
        self.top1 += int(rank <= 1)
        self.top3 += int(rank <= 3)
        self.top5 += int(rank <= 5)
        self.mrr += 1.0 / float(rank)
        self.current_selected += int(selected_is_current)
        self.current_candidate += int(has_current)
        self.n += 1

    def summary(self) -> dict[str, float | int]:
        denom = max(self.n, 1)
        out: dict[str, float | int] = {
            "top1": self.top1 / denom,
            "top3": self.top3 / denom,
            "top5": self.top5 / denom,
            "mrr": self.mrr / denom,
            "n": self.n,
        }
        if self.current_candidate:
            out["current_candidate_rate"] = self.current_candidate / denom
            out["current_selected_rate"] = self.current_selected / max(self.current_candidate, 1)
        return out


def parse_csv_or_all(raw: str, valid: list[str]) -> list[str]:
    if raw == "all":
        return list(valid)
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    bad = [v for v in vals if v not in valid]
    if bad:
        raise SystemExit(f"[error] unknown value(s): {bad}; valid={valid + ['all']}")
    if not vals:
        raise SystemExit("[error] empty comma-separated selection")
    return vals


def resolve_device(raw: str) -> str:
    if raw == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(raw)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[error] CUDA requested but torch.cuda.is_available() is false")
    return str(dev)


def resolve_checkpoints(args: argparse.Namespace) -> list[CheckpointSpec]:
    specs: list[CheckpointSpec] = []
    if args.ckpt:
        ckpt = Path(args.ckpt)
        label = args.ckpt_name or ckpt.parent.name or ckpt.stem
        specs.append(CheckpointSpec(label=label, path=ckpt))
    if args.ckpt_root:
        root = Path(args.ckpt_root)
        variants = parse_csv_or_all(args.checkpoint_variant, CHECKPOINT_VARIANTS)
        for variant in variants:
            ckpt = root / variant / args.checkpoint_name
            if ckpt.exists():
                specs.append(CheckpointSpec(label=variant, path=ckpt))
            elif args.require_all_checkpoints:
                raise SystemExit(f"[error] checkpoint not found: {ckpt}")
            else:
                print(f"[warn] skipping missing checkpoint: {ckpt}")
    if not specs:
        raise SystemExit("[error] provide --ckpt or --ckpt-root with at least one existing checkpoint")

    seen: set[Path] = set()
    unique: list[CheckpointSpec] = []
    for spec in specs:
        path = spec.path.resolve()
        if path in seen:
            continue
        seen.add(path)
        unique.append(CheckpointSpec(label=spec.label, path=path))
    return unique


def encode_dataset(
    model,
    ds: ViewActionDataset,
    device: str,
    batch_size: int,
) -> tuple[torch.Tensor, list[str], list[str]]:
    z_items: list[torch.Tensor] = []
    shape_ids: list[str] = []
    categories: list[str] = []
    with torch.no_grad():
        for item in ds:
            z = encode_all_views(model, item["views"], batch_size=batch_size, device=device)
            z_items.append(F.normalize(z, dim=-1).cpu())
            shape_ids.append(str(item["shape_id"]))
            categories.append(str(item["category"]))
    if not z_items:
        raise SystemExit(f"[error] empty dataset for split={ds.split}")
    return torch.stack(z_items, dim=0), shape_ids, categories


def outgoing_by_source(edges: torch.Tensor, num_views: int) -> dict[int, list[int]]:
    return {
        src: torch.nonzero(edges[:, 0] == src, as_tuple=False).view(-1).tolist()
        for src in range(num_views)
    }


def first_wrong_edge(outgoing: dict[int, list[int]], edges: torch.Tensor, edge_idx: int) -> int:
    src = int(edges[edge_idx, 0].item())
    tgt = int(edges[edge_idx, 1].item())
    for cand in outgoing[src]:
        if int(edges[cand, 1].item()) != tgt:
            return int(cand)
    return int(edge_idx)


def candidate_pairs(
    mode: str,
    shape_idx: int,
    edge_idx: int,
    num_shapes: int,
    num_views: int,
    edges: torch.Tensor,
    action_id: torch.Tensor,
    outgoing: dict[int, list[int]],
    same_action_edges: dict[int, list[int]],
    cross_object_negatives: int,
) -> tuple[list[tuple[int, int]], int]:
    src = int(edges[edge_idx, 0].item())
    tgt = int(edges[edge_idx, 1].item())
    target_pair = (shape_idx, tgt)

    if mode == "same_object_all_views":
        pairs = [(shape_idx, view) for view in range(num_views) if view != src]
    elif mode == "same_source_outgoing":
        pairs = [(shape_idx, int(edges[e, 1].item())) for e in outgoing[src]]
    elif mode == "include_current_negative":
        pairs = [(shape_idx, src)]
        pairs.extend((shape_idx, int(edges[e, 1].item())) for e in outgoing[src])
    elif mode == "cross_object_same_action":
        aid = int(action_id[edge_idx].item())
        raw_pairs = [(shape_idx, tgt)]
        for cand_shape in range(num_shapes):
            if cand_shape == shape_idx:
                continue
            for cand_edge in same_action_edges[aid]:
                raw_pairs.append((cand_shape, int(edges[cand_edge, 1].item())))
                if cross_object_negatives and len(raw_pairs) >= cross_object_negatives + 1:
                    break
            if cross_object_negatives and len(raw_pairs) >= cross_object_negatives + 1:
                break
        pairs = list(dict.fromkeys(raw_pairs))
    else:
        raise ValueError(mode)

    if target_pair not in pairs:
        pairs.append(target_pair)
    target_index = pairs.index(target_pair)
    return pairs, target_index


def predict_variant(
    model,
    z0_cpu: torch.Tensor,
    edge_idx: int,
    wrong_edge_idx: int,
    variant: str,
    action_id: torch.Tensor,
    action_vec: torch.Tensor,
    device: str,
) -> torch.Tensor:
    z0 = z0_cpu.unsqueeze(0).to(device)
    if variant == "no_action":
        pred = model.predict_next(
            z0,
            action_id[edge_idx:edge_idx + 1].to(device),
            action_vec[edge_idx:edge_idx + 1].to(device),
            variant="no_action",
        )
    elif variant == "shuffled_action":
        pred = model.predict_next(
            z0,
            action_id[wrong_edge_idx:wrong_edge_idx + 1].to(device),
            action_vec[wrong_edge_idx:wrong_edge_idx + 1].to(device),
            variant="action",
        )
    elif variant == "action_only":
        pred = model.predict_next(
            z0,
            action_id[edge_idx:edge_idx + 1].to(device),
            action_vec[edge_idx:edge_idx + 1].to(device),
            variant="action_only",
        )
    else:
        pred = model.predict_next(
            z0,
            action_id[edge_idx:edge_idx + 1].to(device),
            action_vec[edge_idx:edge_idx + 1].to(device),
            variant="action",
        )
    return F.normalize(pred.detach().cpu(), dim=-1).view(-1)


def evaluate_checkpoint(
    spec: CheckpointSpec,
    args: argparse.Namespace,
    modes: list[str],
    variants: list[str],
    device: str,
) -> dict[str, dict[str, dict[str, float | int]]]:
    model = load_model(str(spec.path), args.cache_root, device=device)
    ds = ViewActionDataset(
        args.cache_root,
        split=args.split,
        mode="all_views",
        max_shapes=args.max_shapes,
        seed=args.seed,
    )
    pair_ds = ViewActionDataset(
        args.cache_root,
        split=args.split,
        mode="pair",
        max_shapes=args.max_shapes,
        seed=args.seed,
    )
    if len(pair_ds.edges) == 0:
        raise SystemExit("[error] empty view graph")

    all_z, _, _ = encode_dataset(model, ds, device=device, batch_size=args.encode_batch_size)
    num_shapes, num_views, _ = all_z.shape
    edges = torch.as_tensor(pair_ds.edges, dtype=torch.long)
    action_id = torch.as_tensor(pair_ds.action_id, dtype=torch.long)
    action_vec = torch.as_tensor(pair_ds.action_vec, dtype=torch.float32)
    outgoing = outgoing_by_source(edges, num_views)
    same_action_edges = {
        int(a): torch.nonzero(action_id == int(a), as_tuple=False).view(-1).tolist()
        for a in torch.unique(action_id).tolist()
    }

    meters: dict[str, dict[str, RetrievalMeter]] = {
        mode: {variant: RetrievalMeter() for variant in variants}
        for mode in modes
    }

    with torch.no_grad():
        for shape_idx in range(num_shapes):
            for edge_idx in range(edges.shape[0]):
                src = int(edges[edge_idx, 0].item())
                wrong_edge_idx = first_wrong_edge(outgoing, edges, int(edge_idx))
                pred_by_variant = {
                    variant: predict_variant(
                        model,
                        all_z[shape_idx, src],
                        int(edge_idx),
                        wrong_edge_idx,
                        variant,
                        action_id,
                        action_vec,
                        device,
                    )
                    for variant in variants
                }
                for mode in modes:
                    pairs, target_index = candidate_pairs(
                        mode=mode,
                        shape_idx=shape_idx,
                        edge_idx=int(edge_idx),
                        num_shapes=num_shapes,
                        num_views=num_views,
                        edges=edges,
                        action_id=action_id,
                        outgoing=outgoing,
                        same_action_edges=same_action_edges,
                        cross_object_negatives=args.cross_object_negatives,
                    )
                    shape_ix = torch.as_tensor([p[0] for p in pairs], dtype=torch.long)
                    view_ix = torch.as_tensor([p[1] for p in pairs], dtype=torch.long)
                    cand_z = all_z[shape_ix, view_ix]
                    target_score_index = int(target_index)
                    current_mask = (shape_ix == shape_idx) & (view_ix == src)
                    has_current = bool(current_mask.any().item())
                    target_pair = pairs[target_score_index]
                    for variant, pred in pred_by_variant.items():
                        sims = cand_z @ pred
                        target_score = sims[target_score_index]
                        rank = int((sims > target_score).sum().item()) + 1
                        selected = int(torch.argmax(sims).item())
                        selected_is_current = pairs[selected] == (shape_idx, src)
                        if pairs[selected] == target_pair:
                            selected_is_current = False
                        meters[mode][variant].update(rank, selected_is_current, has_current)

    del model
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return {
        mode: {variant: meters[mode][variant].summary() for variant in variants}
        for mode in modes
    }


def write_markdown(path: Path, results: dict[str, dict[str, dict[str, dict[str, float | int]]]]) -> None:
    lines = [
        "# Hard next-view retrieval",
        "",
        "| checkpoint | mode | inference_variant | top1 | top3 | top5 | MRR | current_selected | n |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ckpt_label, by_mode in results.items():
        for mode, by_variant in by_mode.items():
            for variant, row in by_variant.items():
                current = row.get("current_selected_rate", float("nan"))
                lines.append(
                    f"| {ckpt_label} | {mode} | {variant} | "
                    f"{float(row['top1']):.4f} | {float(row['top3']):.4f} | "
                    f"{float(row['top5']):.4f} | {float(row['mrr']):.4f} | "
                    f"{float(current):.4f} | {int(row['n'])} |"
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--ckpt-name", default="")
    ap.add_argument("--ckpt-root", default="")
    ap.add_argument("--checkpoint-variant", default="all",
                    help="Comma-separated checkpoint variants or all; used with --ckpt-root.")
    ap.add_argument("--checkpoint-name", default="ckpt_last.pth")
    ap.add_argument("--require-all-checkpoints", action="store_true")
    ap.add_argument("--split", default="test")
    ap.add_argument("--variant", default="all",
                    help="Comma-separated inference variants or all.")
    ap.add_argument("--candidate-mode", default="all",
                    help="Comma-separated hard candidate modes or all.")
    ap.add_argument("--max-shapes", type=int, default=200)
    ap.add_argument("--encode-batch-size", type=int, default=32)
    ap.add_argument("--cross-object-negatives", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", default="")
    args = ap.parse_args()

    device = resolve_device(args.device)
    variants = parse_csv_or_all(args.variant, INFERENCE_VARIANTS)
    modes = parse_csv_or_all(args.candidate_mode, HARD_MODES)
    checkpoints = resolve_checkpoints(args)

    results: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    for spec in checkpoints:
        print(f"[eval] checkpoint={spec.label} path={spec.path} device={device}")
        results[spec.label] = evaluate_checkpoint(spec, args, modes=modes, variants=variants, device=device)

    payload = {
        "metadata": {
            "cache_root": args.cache_root,
            "split": args.split,
            "max_shapes": args.max_shapes,
            "candidate_modes": modes,
            "inference_variants": variants,
            "checkpoints": [{"label": spec.label, "path": str(spec.path)} for spec in checkpoints],
        },
        "results": results,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    if args.out_md:
        write_markdown(Path(args.out_md), results)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
