from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nepa3d.data.modelnet40_index import list_npz
from nepa3d.tracks.patch_nepa.cqa.analysis.eval_primitive_answering_tokens import apply_control
from nepa3d.tracks.patch_nepa.cqa.data.cqa_codec import CQA_VOCAB_VERSION, query_name_to_id
from nepa3d.tracks.patch_nepa.cqa.data.dataset_cqa import (
    _choice,
    _choice_with_replacement_if_needed,
    _ordered_xyz_perm,
    _resolve_query_order,
    _select_optional_bank,
)
from nepa3d.tracks.patch_nepa.cqa.models.factory import load_cqa_model_from_ckpt


def _parse_controls(text: str) -> list[str]:
    vals = [x.strip() for x in str(text).split(",") if x.strip()]
    return vals if vals else ["correct"]


def _sample_paths(paths: Sequence[str], *, max_samples: int, seed: int, mode: str) -> list[str]:
    if int(max_samples) <= 0 or len(paths) <= int(max_samples):
        return list(paths)
    if str(mode) == "head":
        return list(paths[: int(max_samples)])
    if str(mode) != "random":
        raise KeyError(f"unknown sample_mode={mode}")
    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    take = rng.choice(len(paths), size=int(max_samples), replace=False)
    return [paths[int(i)] for i in take.tolist()]


def _load_paths(cache_root: str, split: str, *, manifest_json: str | None) -> list[str]:
    if not manifest_json:
        return list_npz(cache_root, split)
    payload = json.loads(Path(manifest_json).read_text())
    records = payload.get("records", [])
    out: list[str] = []
    for rec in records:
        if str(rec.get("split", "")) != str(split):
            continue
        rel = str(rec.get("src_relpath", "")).strip()
        if not rel:
            continue
        path = os.path.join(cache_root, rel)
        if os.path.exists(path):
            out.append(path)
    return sorted(set(out))


def _np_to_torch_f32(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32))


def _np_to_torch_i64(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.int64))


def _apply_query_order_probe(
    *,
    qry_xyz: np.ndarray,
    target: np.ndarray,
    rng: Any,
    query_order: str,
) -> tuple[np.ndarray, np.ndarray]:
    if int(qry_xyz.shape[0]) <= 1:
        return qry_xyz, target
    if query_order == "sampled":
        return qry_xyz, target
    if query_order == "shuffled":
        perm = rng.permutation(int(qry_xyz.shape[0])).astype(np.int64)
    elif query_order == "ordered_xyz":
        perm = _ordered_xyz_perm(qry_xyz)
    else:
        raise KeyError(f"unknown query_order={query_order}")
    return qry_xyz[perm], target[perm]


class FrozenProbeDataset(Dataset):
    def __init__(
        self,
        paths: Sequence[str],
        *,
        probe_target: str,
        codec_version: str,
        context_source: str = "surf",
        n_ctx: int = 2048,
        n_qry: int = 64,
        seed: int = 0,
        mode: str = "train",
        query_order: str | None = None,
    ) -> None:
        super().__init__()
        self.paths = list(paths)
        self.probe_target = str(probe_target)
        if self.probe_target not in {"curvature", "signed_normal"}:
            raise KeyError(f"unsupported probe_target={probe_target}")
        self.codec_version = str(codec_version or CQA_VOCAB_VERSION)
        self.query_type = int(query_name_to_id(self.codec_version)["mesh_normal_unsigned"])
        self.context_source = str(context_source)
        self.n_ctx = int(n_ctx)
        self.n_qry = int(n_qry)
        self.seed = int(seed)
        self.mode = str(mode)
        self.query_order = _resolve_query_order(query_order, mode=self.mode)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.paths[idx]
        rng = np.random if self.mode == "train" else np.random.RandomState(self.seed + idx)
        with np.load(path, allow_pickle=False) as npz:
            if self.context_source == "surf":
                ctx_all = np.asarray(npz["surf_xyz"], dtype=np.float32)
                ctx_bank_idx = None
            elif self.context_source == "pc_bank":
                ctx_all, ctx_bank_idx = _select_optional_bank(
                    np.asarray(npz["pc_ctx_bank_xyz"], dtype=np.float32),
                    rng,
                    self.mode,
                    fallback_seed=self.seed + idx,
                )
            else:
                raise ValueError(f"unknown context_source={self.context_source}")
            ctx_idx = _choice(int(ctx_all.shape[0]), self.n_ctx, rng)
            ctx_xyz = np.asarray(ctx_all[ctx_idx], dtype=np.float32)

            surf_xyz = np.asarray(npz["surf_xyz"], dtype=np.float32)
            q_pool = np.arange(int(surf_xyz.shape[0]), dtype=np.int64)
            if self.n_qry > 0:
                if int(q_pool.shape[0]) >= self.n_qry:
                    take = _choice(int(q_pool.shape[0]), self.n_qry, rng)
                else:
                    take = _choice_with_replacement_if_needed(int(q_pool.shape[0]), self.n_qry, rng)
                q_idx = q_pool[take]
            else:
                q_idx = q_pool
            qry_xyz = np.asarray(surf_xyz[q_idx], dtype=np.float32)

            if self.probe_target == "curvature":
                target = np.asarray(npz["mesh_surf_curv"], dtype=np.float32).reshape(-1, 1)[q_idx]
            else:
                target = np.asarray(npz["mesh_surf_n"], dtype=np.float32).reshape(-1, 3)[q_idx]
                target /= np.linalg.norm(target, axis=1, keepdims=True) + 1e-8
            qry_xyz, target = _apply_query_order_probe(
                qry_xyz=qry_xyz,
                target=target,
                rng=rng,
                query_order=self.query_order,
            )

            path_obj = Path(path)
            out: Dict[str, Any] = {
                "ctx_xyz": _np_to_torch_f32(ctx_xyz),
                "qry_xyz": _np_to_torch_f32(qry_xyz),
                "qry_type": torch.full((int(qry_xyz.shape[0]),), int(self.query_type), dtype=torch.long),
                "answer_code": _np_to_torch_i64(np.zeros((int(qry_xyz.shape[0]),), dtype=np.int64)),
                "target": _np_to_torch_f32(target),
                "probe_target": self.probe_target,
                "context_source": self.context_source,
                "cache_split": path_obj.parent.parent.name,
                "synset": path_obj.parent.name,
                "path": path,
                "context_bank_idx": None if ctx_bank_idx is None else int(ctx_bank_idx),
                "query_order": self.query_order,
                "vocab_version": self.codec_version,
            }
            return out


def probe_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["ctx_xyz"] = torch.stack([b["ctx_xyz"] for b in batch], dim=0)
    out["qry_xyz"] = torch.stack([b["qry_xyz"] for b in batch], dim=0)
    out["qry_type"] = torch.stack([b["qry_type"] for b in batch], dim=0)
    out["answer_code"] = torch.stack([b["answer_code"] for b in batch], dim=0)
    out["target"] = torch.stack([b["target"] for b in batch], dim=0)
    out["probe_target"] = batch[0]["probe_target"]
    out["context_source"] = [b["context_source"] for b in batch]
    out["cache_split"] = [b["cache_split"] for b in batch]
    out["synset"] = [b["synset"] for b in batch]
    out["path"] = [b["path"] for b in batch]
    out["context_bank_idx"] = [b.get("context_bank_idx") for b in batch]
    out["query_order"] = [b.get("query_order") for b in batch]
    out["vocab_version"] = str(batch[0].get("vocab_version", CQA_VOCAB_VERSION))
    return out


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(int(in_dim), int(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@dataclass(frozen=True)
class ProbeEvalSpec:
    name: str
    dataset: FrozenProbeDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("train_frozen_geometric_probe")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--cache_root", type=str, required=True)
    p.add_argument("--probe_target", type=str, required=True, choices=["curvature", "signed_normal"])
    p.add_argument("--manifest_json", type=str, default="")
    p.add_argument("--save_dir", type=str, default="runs/cqa_probe")
    p.add_argument("--run_name", type=str, default="probe_debug")
    p.add_argument("--out_json", type=str, default="")
    p.add_argument("--train_split", type=str, default="train_mesh")
    p.add_argument("--eval_split", type=str, default="eval")
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--n_ctx", type=int, default=2048)
    p.add_argument("--n_qry", type=int, default=64)
    p.add_argument("--train_query_order", type=str, default="shuffled")
    p.add_argument("--eval_query_order", type=str, default="sampled")
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_eval_samples", type=int, default=128)
    p.add_argument("--eval_sample_mode", type=str, default="random", choices=["random", "head"])
    p.add_argument("--controls", type=str, default="correct,no_context,wrong_shape_same_synset,wrong_shape_other_synset,shuffled_query")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _build_datasets(args: argparse.Namespace, *, codec_version: str) -> tuple[FrozenProbeDataset, list[ProbeEvalSpec]]:
    train_paths = _load_paths(str(args.cache_root), str(args.train_split), manifest_json=str(args.manifest_json or ""))
    eval_paths = _load_paths(str(args.cache_root), str(args.eval_split), manifest_json=str(args.manifest_json or ""))
    train_paths = _sample_paths(
        train_paths,
        max_samples=int(args.max_train_samples),
        seed=int(args.seed),
        mode=str(args.eval_sample_mode),
    )
    eval_paths = _sample_paths(
        eval_paths,
        max_samples=int(args.max_eval_samples),
        seed=int(args.seed) + 17,
        mode=str(args.eval_sample_mode),
    )
    train_ds = FrozenProbeDataset(
        train_paths,
        probe_target=str(args.probe_target),
        codec_version=str(codec_version),
        context_source="surf",
        n_ctx=int(args.n_ctx),
        n_qry=int(args.n_qry),
        seed=int(args.seed),
        mode="train",
        query_order=str(args.train_query_order),
    )
    same_ds = FrozenProbeDataset(
        eval_paths,
        probe_target=str(args.probe_target),
        codec_version=str(codec_version),
        context_source="surf",
        n_ctx=int(args.n_ctx),
        n_qry=int(args.n_qry),
        seed=int(args.seed),
        mode="eval",
        query_order=str(args.eval_query_order),
    )
    offdiag_ds = FrozenProbeDataset(
        eval_paths,
        probe_target=str(args.probe_target),
        codec_version=str(codec_version),
        context_source="pc_bank",
        n_ctx=int(args.n_ctx),
        n_qry=int(args.n_qry),
        seed=int(args.seed),
        mode="eval",
        query_order=str(args.eval_query_order),
    )
    eval_specs = [
        ProbeEvalSpec(name="same", dataset=same_ds),
        ProbeEvalSpec(name="offdiag", dataset=offdiag_ds),
    ]
    return train_ds, eval_specs


def _extract_answer_hidden(model: nn.Module, batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        out = model(
            ctx_xyz=batch["ctx_xyz"].to(device, non_blocking=True),
            qry_xyz=batch["qry_xyz"].to(device, non_blocking=True),
            qry_type=batch["qry_type"].to(device, non_blocking=True),
            answer_code=batch["answer_code"].to(device, non_blocking=True),
        )
    return out.answer_hidden.detach()


def _probe_loss_and_pred(probe_target: str, head: nn.Module, features: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred = head(features)
    if str(probe_target) == "curvature":
        loss = F.mse_loss(pred, target)
        return loss, pred
    if str(probe_target) == "signed_normal":
        pred_n = F.normalize(pred, dim=-1)
        tgt_n = F.normalize(target, dim=-1)
        loss = (1.0 - (pred_n * tgt_n).sum(dim=-1)).mean()
        return loss, pred_n
    raise KeyError(f"unknown probe_target={probe_target}")


def _collect_metrics(probe_target: str, pred: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
    if pred.size == 0 or target.size == 0:
        return {"count": 0}
    if str(probe_target) == "curvature":
        pred1 = pred.reshape(-1)
        tgt1 = target.reshape(-1)
        err = pred1 - tgt1
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(np.square(err))))
        if pred1.size > 1 and float(np.std(pred1)) > 1e-8 and float(np.std(tgt1)) > 1e-8:
            pearson = float(np.corrcoef(pred1, tgt1)[0, 1])
        else:
            pearson = 0.0
        mean_pred = float(np.mean(tgt1))
        mean_mae = float(np.mean(np.abs(tgt1 - mean_pred)))
        mean_rmse = float(np.sqrt(np.mean(np.square(tgt1 - mean_pred))))
        return {
            "count": int(pred1.size),
            "mae": mae,
            "rmse": rmse,
            "pearson_r": pearson,
            "mean_baseline_mae": mean_mae,
            "mean_baseline_rmse": mean_rmse,
        }
    pred_n = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
    tgt_n = target / (np.linalg.norm(target, axis=-1, keepdims=True) + 1e-8)
    cos = np.clip(np.sum(pred_n * tgt_n, axis=-1), -1.0, 1.0).reshape(-1)
    angle = np.degrees(np.arccos(cos))
    return {
        "count": int(cos.size),
        "mean_cos": float(np.mean(cos)),
        "angle_deg": float(np.mean(angle)),
    }


def evaluate_probe(
    model: nn.Module,
    head: nn.Module,
    spec: ProbeEvalSpec,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    controls: Sequence[str],
) -> Dict[str, Any]:
    loader = DataLoader(
        spec.dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=False,
        collate_fn=probe_collate_fn,
    )
    probe_target = str(spec.dataset.probe_target)
    out: Dict[str, Any] = {"dataset": spec.name, "probe_target": probe_target, "by_control": {}}
    for control in controls:
        preds: list[np.ndarray] = []
        tgts: list[np.ndarray] = []
        for batch in loader:
            ctl_batch = apply_control(batch, str(control))
            feats = _extract_answer_hidden(model, ctl_batch, device)
            target = ctl_batch["target"].to(device, non_blocking=True)
            pred = head(feats)
            if probe_target == "signed_normal":
                pred = F.normalize(pred, dim=-1)
            eff = ctl_batch.get("control_effective_mask")
            if eff is None:
                eff_q = torch.ones((int(target.shape[0]), int(target.shape[1])), device=target.device, dtype=torch.bool)
            else:
                eff_q = eff.to(device=target.device, dtype=torch.bool).unsqueeze(1).expand(-1, int(target.shape[1]))
            if int(eff_q.sum().item()) <= 0:
                continue
            pred_eff = pred[eff_q]
            tgt_eff = target[eff_q]
            if probe_target == "curvature":
                pred_eff = pred_eff.view(-1, 1)
                tgt_eff = tgt_eff.view(-1, 1)
            else:
                pred_eff = pred_eff.view(-1, 3)
                tgt_eff = tgt_eff.view(-1, 3)
            preds.append(pred_eff.detach().cpu().numpy())
            tgts.append(tgt_eff.detach().cpu().numpy())
        if preds:
            pred_np = np.concatenate(preds, axis=0)
            tgt_np = np.concatenate(tgts, axis=0)
        else:
            pred_np = np.zeros((0, 1), dtype=np.float32) if probe_target == "curvature" else np.zeros((0, 3), dtype=np.float32)
            tgt_np = np.zeros_like(pred_np)
        out["by_control"][str(control)] = _collect_metrics(probe_target, pred_np, tgt_np)

    correct = out["by_control"].get("correct", {})
    deltas: Dict[str, Any] = {}
    if probe_target == "curvature":
        base_mae = float(correct.get("mae", float("nan")))
        for control, metrics in out["by_control"].items():
            if control == "correct" or "mae" not in metrics:
                continue
            deltas[control] = {"delta_mae": float(metrics["mae"] - base_mae)}
    else:
        base_cos = float(correct.get("mean_cos", float("nan")))
        for control, metrics in out["by_control"].items():
            if control == "correct" or "mean_cos" not in metrics:
                continue
            deltas[control] = {"delta_mean_cos": float(metrics["mean_cos"] - base_cos)}
    out["deltas_vs_correct"] = deltas
    return out


def _evaluate_same_correct(
    model: nn.Module,
    head: nn.Module,
    spec: ProbeEvalSpec,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, Any]:
    result = evaluate_probe(
        model,
        head,
        spec,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        controls=["correct"],
    )
    return result["by_control"]["correct"]


def main() -> None:
    args = parse_args()
    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    model, _ckpt, model_args = load_cqa_model_from_ckpt(str(args.ckpt), device=device)
    factorization = str(model_args.get("answer_factorization", "ar"))
    if factorization == "ar":
        raise ValueError("frozen geometric probes currently support non-AR CQA checkpoints only")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    codec_version = str(model_args.get("codec_version", CQA_VOCAB_VERSION))
    train_ds, eval_specs = _build_datasets(args, codec_version=codec_version)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        collate_fn=probe_collate_fn,
    )

    d_model = int(getattr(model, "d_model"))
    out_dim = 1 if str(args.probe_target) == "curvature" else 3
    head = LinearProbe(d_model, out_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json) if str(args.out_json).strip() else (save_dir / "summary.json")

    controls = _parse_controls(args.controls)
    best_metric = -float("inf") if str(args.probe_target) == "signed_normal" else float("inf")
    best_step = 0
    best_ckpt_path = save_dir / "ckpt_best.pt"

    train_iter = iter(train_loader)
    pbar = tqdm(range(1, int(args.max_steps) + 1), desc="probe", leave=False)
    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        head.train()
        feats = _extract_answer_hidden(model, batch, device)
        target = batch["target"].to(device, non_blocking=True)
        loss, pred = _probe_loss_and_pred(str(args.probe_target), head, feats, target)
        del pred
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        pbar.set_postfix(loss=float(loss.detach().cpu()))

        should_eval = step == int(args.max_steps) or (int(args.eval_every) > 0 and step % int(args.eval_every) == 0)
        if not should_eval:
            continue
        head.eval()
        same_correct = _evaluate_same_correct(
            model,
            head,
            eval_specs[0],
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
        )
        if str(args.probe_target) == "curvature":
            metric = float(same_correct.get("mae", float("inf")))
            is_better = metric < float(best_metric)
        else:
            metric = float(same_correct.get("mean_cos", -float("inf")))
            is_better = metric > float(best_metric)
        if is_better:
            best_metric = metric
            best_step = int(step)
            torch.save(
                {
                    "head": head.state_dict(),
                    "args": vars(args),
                    "codec_version": codec_version,
                    "best_metric": best_metric,
                    "best_step": best_step,
                    "model_ckpt": str(args.ckpt),
                },
                best_ckpt_path,
            )

    if best_ckpt_path.exists():
        payload = torch.load(best_ckpt_path, map_location="cpu")
        head.load_state_dict(payload["head"], strict=True)
    head.eval()

    results = {
        "args": dict(vars(args)),
        "codec_version": codec_version,
        "model_ckpt": str(args.ckpt),
        "model_arch": str(model_args.get("model_arch", "prefixlm")),
        "answer_factorization": str(model_args.get("answer_factorization", "")),
        "query_interface_mode": str(model_args.get("query_interface_mode", "")),
        "best_metric": best_metric,
        "best_step": best_step,
        "controls": controls,
        "train_count": len(train_ds),
        "eval_count": {spec.name: len(spec.dataset) for spec in eval_specs},
        "results": {},
    }
    for spec in eval_specs:
        results["results"][spec.name] = evaluate_probe(
            model,
            head,
            spec,
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            controls=controls,
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2) + "\n")
    with open(save_dir / "args.json", "w") as f:
        json.dump(dict(vars(args)), f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
