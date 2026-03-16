from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nepa3d.tracks.patch_nepa.cqa.data.cqa_codec import (
    ANSWER_VOCAB_SIZE,
    QUERY_TYPE_NAMES,
    QUERY_TYPE_VOCAB_SIZE,
    answer_range_for_query_type,
    mask_logits_for_query_type,
)
from nepa3d.tracks.patch_nepa.cqa.data.dataset_cqa import (
    QUERY_SRC_CODE_TO_NAME,
    V2PrimitiveCQADataset,
    cqa_collate_fn,
)
from nepa3d.data.mixed_pretrain import load_mix_config
from nepa3d.data.modelnet40_index import list_npz
from nepa3d.tracks.patch_nepa.cqa.models.primitive_answering import PrimitiveAnsweringModel

CONTROL_MODES = (
    "correct",
    "no_context",
    "wrong_shape",
    "wrong_shape_same_synset",
    "wrong_shape_other_synset",
    "shuffled_context",
    "wrong_type",
    "shuffled_query",
)


@dataclass(frozen=True)
class EvalDatasetSpec:
    name: str
    split: str
    cache_root: str
    context_source: str
    eval_sample_mode: str
    dataset: V2PrimitiveCQADataset


def _parse_task_filter(text: str) -> set[str]:
    s = str(text).strip()
    if not s:
        return set()
    return {x.strip() for x in s.split(",") if x.strip()}


def _sample_eval_paths(paths: List[str], *, max_samples: int, seed: int, mode: str) -> List[str]:
    if int(max_samples) <= 0 or len(paths) <= int(max_samples):
        return list(paths)
    mode = str(mode)
    if mode == "head":
        return list(paths[: int(max_samples)])
    if mode == "random":
        rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
        take = rng.choice(len(paths), size=int(max_samples), replace=False)
        return [paths[int(i)] for i in take.tolist()]
    raise KeyError(f"unknown eval_sample_mode={mode}")


def _stable_int_seed(text: str) -> int:
    x = 0
    for ch in str(text):
        x = (x * 131 + ord(ch)) % 2147483647
    return int(x)


def load_cqa_model(ckpt_path: str, device: torch.device) -> tuple[PrimitiveAnsweringModel, Dict[str, Any], Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = dict(ckpt.get("args", {}))
    model = PrimitiveAnsweringModel(
        d_model=int(args.get("d_model", 384)),
        n_layers=int(args.get("n_layers", 12)),
        n_heads=int(args.get("n_heads", 6)),
        mlp_ratio=float(args.get("mlp_ratio", 4.0)),
        dropout=float(args.get("dropout", 0.0)),
        drop_path=float(args.get("drop_path", 0.0)),
        backbone_impl=str(args.get("backbone_impl", "nepa2d")),
        num_groups=int(args.get("num_groups", 64)),
        group_size=int(args.get("group_size", 32)),
        patch_center_mode=str(args.get("patch_center_mode", "fps")),
        patch_fps_random_start=bool(args.get("patch_fps_random_start", 1)),
        local_encoder=str(args.get("local_encoder", "pointmae_conv")),
        query_type_vocab=int(args.get("query_type_vocab", QUERY_TYPE_VOCAB_SIZE)),
        answer_vocab=int(args.get("answer_vocab", ANSWER_VOCAB_SIZE)),
        generator_depth=int(args.get("generator_depth", 2)),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model, ckpt, args


def build_eval_datasets(
    mix_config_path: str,
    *,
    seed: int,
    n_ctx: int,
    n_qry: int,
    max_samples_per_task: int,
    split_override: str | None,
    task_filter: set[str],
    eval_sample_mode: str,
) -> List[EvalDatasetSpec]:
    specs, _cfg = load_mix_config(mix_config_path)
    out: List[EvalDatasetSpec] = []
    for s in specs:
        task_name = str(s.extra.get("task_name", s.name))
        if task_filter and task_name not in task_filter and s.name not in task_filter:
            continue
        split = str(split_override or s.split)
        paths = list_npz(s.cache_root, split)
        if int(max_samples_per_task) > 0:
            paths = _sample_eval_paths(
                paths,
                max_samples=int(max_samples_per_task),
                seed=(int(seed) + _stable_int_seed(f"{task_name}:{split}")),
                mode=str(eval_sample_mode),
            )
        ds = V2PrimitiveCQADataset(
            paths,
            task_name=task_name,
            context_source=str(s.extra.get("context_source", "surf")),
            n_ctx=int(s.extra.get("n_ctx", n_ctx)),
            n_qry=int(s.extra.get("n_qry", n_qry)),
            seed=int(seed),
            mode="eval",
            query_src_filter=s.extra.get("query_src_filter", None),
            query_dist_min=s.extra.get("query_dist_min", None),
            query_dist_max=s.extra.get("query_dist_max", None),
        )
        out.append(
            EvalDatasetSpec(
                name=task_name,
                split=split,
                cache_root=str(s.cache_root),
                context_source=str(s.extra.get("context_source", "surf")),
                eval_sample_mode=str(eval_sample_mode),
                dataset=ds,
            )
        )
    return out


def _clone_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.clone()
        elif isinstance(v, list):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _swap_indices(batch: Dict[str, Any], predicate) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(batch["ctx_xyz"].shape[0])
    perm = list(range(batch_size))
    effective = torch.zeros((batch_size,), dtype=torch.bool)
    synset = list(batch.get("synset", []))
    path = list(batch.get("path", []))
    for i in range(batch_size):
        donor = None
        for offset in range(1, batch_size):
            j = (i + offset) % batch_size
            if predicate(i, j, synset, path):
                donor = j
                break
        if donor is not None:
            perm[i] = int(donor)
            effective[i] = True
    return torch.as_tensor(perm, dtype=torch.long), effective


def apply_control(batch: Dict[str, Any], control: str) -> Dict[str, Any]:
    control = str(control)
    if control not in CONTROL_MODES:
        raise KeyError(f"unknown control={control}")
    out = _clone_batch(batch)
    out["control_effective_mask"] = torch.ones((int(out["ctx_xyz"].shape[0]),), dtype=torch.bool)
    if control == "correct":
        return out
    if control == "no_context":
        out["ctx_xyz"] = torch.zeros_like(out["ctx_xyz"])
        return out
    if control == "wrong_shape":
        if int(out["ctx_xyz"].shape[0]) > 1:
            out["ctx_xyz"] = torch.roll(out["ctx_xyz"], shifts=1, dims=0)
        return out
    if control == "wrong_shape_same_synset":
        perm, eff = _swap_indices(
            out,
            lambda i, j, synset, path: len(synset) > 0 and synset[i] == synset[j] and path[i] != path[j],
        )
        out["ctx_xyz"] = out["ctx_xyz"][perm]
        out["control_effective_mask"] = eff
        return out
    if control == "wrong_shape_other_synset":
        perm, eff = _swap_indices(
            out,
            lambda i, j, synset, path: len(synset) > 0 and synset[i] != synset[j],
        )
        out["ctx_xyz"] = out["ctx_xyz"][perm]
        out["control_effective_mask"] = eff
        return out
    if control == "shuffled_context":
        if int(out["ctx_xyz"].shape[0]) > 1:
            out["ctx_xyz"] = torch.roll(out["ctx_xyz"], shifts=1, dims=0)
        out["ctx_xyz"] = torch.flip(out["ctx_xyz"], dims=[1])
        return out
    if control == "wrong_type":
        out["qry_type"] = (out["qry_type"] + 1) % int(QUERY_TYPE_VOCAB_SIZE)
        return out
    if control == "shuffled_query":
        out["qry_xyz"] = torch.flip(out["qry_xyz"], dims=[1])
        return out
    raise AssertionError("unreachable")


def evaluate_dataset(
    model: PrimitiveAnsweringModel,
    spec: EvalDatasetSpec,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    control: str,
) -> Dict[str, Any]:
    loader = DataLoader(
        spec.dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
        collate_fn=cqa_collate_fn,
        drop_last=False,
    )

    total_tokens = 0
    total_loss = 0.0
    total_acc = 0.0
    total_entropy = 0.0
    total_samples = 0
    total_effective_samples = 0
    pred_hist = torch.zeros((ANSWER_VOCAB_SIZE,), dtype=torch.long)
    target_hist = torch.zeros((ANSWER_VOCAB_SIZE,), dtype=torch.long)
    src_stats: Dict[int, Dict[str, Any]] = {}
    pbar = tqdm(loader, desc=f"{spec.name}:{control}", leave=False)
    with torch.inference_mode():
        for batch in pbar:
            batch = apply_control(batch, control)
            ctx_xyz = batch["ctx_xyz"].to(device=device, dtype=torch.float32, non_blocking=True)
            qry_xyz = batch["qry_xyz"].to(device=device, dtype=torch.float32, non_blocking=True)
            qry_type = batch["qry_type"].to(device=device, dtype=torch.long, non_blocking=True)
            answer_code = batch["answer_code"].to(device=device, dtype=torch.long, non_blocking=True)
            qry_src_code = batch["qry_src_code"].to(device=device, dtype=torch.long, non_blocking=True)
            control_effective_mask = batch["control_effective_mask"]

            out = model(ctx_xyz=ctx_xyz, qry_xyz=qry_xyz, qry_type=qry_type, answer_code=answer_code)
            masked_logits = mask_logits_for_query_type(out.logits, qry_type)
            flat_logits = masked_logits.reshape(-1, int(masked_logits.shape[-1]))
            flat_target = answer_code.reshape(-1)
            per_token_loss = torch.nn.functional.cross_entropy(flat_logits, flat_target, reduction="none").reshape_as(
                answer_code
            )
            loss = per_token_loss.sum()

            pred = masked_logits.argmax(dim=-1)
            correct_mask = (pred == answer_code)
            correct = correct_mask.sum()
            probs = masked_logits.softmax(dim=-1)
            per_token_entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            entropy = per_token_entropy.sum()

            n_tok = int(answer_code.numel())
            total_tokens += n_tok
            total_loss += float(loss.detach().cpu())
            total_acc += float(correct.detach().cpu())
            total_entropy += float(entropy.detach().cpu())
            total_samples += int(answer_code.shape[0])
            total_effective_samples += int(control_effective_mask.sum().item())
            pred_hist += torch.bincount(pred.reshape(-1).detach().cpu(), minlength=ANSWER_VOCAB_SIZE)
            target_hist += torch.bincount(answer_code.reshape(-1).detach().cpu(), minlength=ANSWER_VOCAB_SIZE)
            for code in torch.unique(qry_src_code).tolist():
                key = int(code)
                mask = qry_src_code == key
                n_tok_src = int(mask.sum().item())
                if n_tok_src <= 0:
                    continue
                if key not in src_stats:
                    src_stats[key] = {
                        "n_tokens": 0,
                        "loss_sum": 0.0,
                        "acc_sum": 0.0,
                        "entropy_sum": 0.0,
                        "pred_hist": torch.zeros((ANSWER_VOCAB_SIZE,), dtype=torch.long),
                        "target_hist": torch.zeros((ANSWER_VOCAB_SIZE,), dtype=torch.long),
                    }
                st = src_stats[key]
                st["n_tokens"] += n_tok_src
                st["loss_sum"] += float(per_token_loss[mask].sum().detach().cpu())
                st["acc_sum"] += float(correct_mask[mask].sum().detach().cpu())
                st["entropy_sum"] += float(per_token_entropy[mask].sum().detach().cpu())
                st["pred_hist"] += torch.bincount(pred[mask].reshape(-1).detach().cpu(), minlength=ANSWER_VOCAB_SIZE)
                st["target_hist"] += torch.bincount(
                    answer_code[mask].reshape(-1).detach().cpu(),
                    minlength=ANSWER_VOCAB_SIZE,
                )

            if total_tokens > 0:
                pbar.set_postfix(
                    ce=total_loss / float(total_tokens),
                    acc=total_acc / float(total_tokens),
                )

    if total_tokens <= 0:
        raise RuntimeError(f"empty eval dataset for task={spec.name} split={spec.split}")
    qtype = int(spec.dataset.task.query_type)
    lo, hi = answer_range_for_query_type(qtype)
    nonzero_hist = {int(i): int(v) for i, v in enumerate(pred_hist.tolist()) if int(v) > 0}
    nonzero_target_hist = {int(i): int(v) for i, v in enumerate(target_hist.tolist()) if int(v) > 0}
    pred_unique = int((pred_hist > 0).sum().item())
    target_unique = int((target_hist > 0).sum().item())
    pred_top1 = int(pred_hist.argmax().item())
    target_top1 = int(target_hist.argmax().item())
    by_query_src = {}
    for key in sorted(src_stats.keys()):
        st = src_stats[key]
        n_tok = int(st["n_tokens"])
        pred_h = st["pred_hist"]
        target_h = st["target_hist"]
        pred_top1_src = int(pred_h.argmax().item())
        target_top1_src = int(target_h.argmax().item())
        by_query_src[str(key)] = {
            "query_src_code": int(key),
            "query_src_name": str(QUERY_SRC_CODE_TO_NAME.get(int(key), f"code_{int(key)}")),
            "n_tokens": n_tok,
            "ce": float(st["loss_sum"]) / float(max(1, n_tok)),
            "token_acc": float(st["acc_sum"]) / float(max(1, n_tok)),
            "answer_entropy": float(st["entropy_sum"]) / float(max(1, n_tok)),
            "majority_baseline_acc": float(target_h.max().item()) / float(max(1, n_tok)),
            "pred_top1_code": pred_top1_src,
            "pred_top1_share": float(pred_h[pred_top1_src].item()) / float(max(1, n_tok)),
            "pred_unique_codes": int((pred_h > 0).sum().item()),
            "target_top1_code": target_top1_src,
            "target_top1_share": float(target_h[target_top1_src].item()) / float(max(1, n_tok)),
            "target_unique_codes": int((target_h > 0).sum().item()),
        }
    return {
        "task_name": spec.name,
        "split": spec.split,
        "cache_root": spec.cache_root,
        "context_source": spec.context_source,
        "eval_sample_mode": spec.eval_sample_mode,
        "control": control,
        "query_type_id": qtype,
        "query_type_name": QUERY_TYPE_NAMES[qtype],
        "answer_range": [int(lo), int(hi)],
        "n_samples": int(len(spec.dataset)),
        "effective_control_samples": int(total_effective_samples),
        "effective_control_sample_frac": float(total_effective_samples) / float(max(1, total_samples)),
        "n_tokens": int(total_tokens),
        "ce": total_loss / float(total_tokens),
        "token_acc": total_acc / float(total_tokens),
        "answer_entropy": total_entropy / float(total_tokens),
        "majority_baseline_acc": float(target_hist.max().item()) / float(max(1, total_tokens)),
        "pred_top1_code": pred_top1,
        "pred_top1_share": float(pred_hist[pred_top1].item()) / float(max(1, total_tokens)),
        "pred_unique_codes": pred_unique,
        "target_top1_code": target_top1,
        "target_top1_share": float(target_hist[target_top1].item()) / float(max(1, total_tokens)),
        "target_unique_codes": target_unique,
        "pred_hist_nonzero": nonzero_hist,
        "target_hist_nonzero": nonzero_target_hist,
        "query_src_breakdown": by_query_src,
    }


def run_token_eval(
    *,
    ckpt_path: str,
    mix_config_path: str,
    device: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    n_ctx: int,
    n_qry: int,
    max_samples_per_task: int,
    split_override: str | None,
    task_filter: set[str],
    control: str,
    eval_sample_mode: str,
) -> Dict[str, Any]:
    torch_device = torch.device(device)
    model, ckpt, train_args = load_cqa_model(ckpt_path, torch_device)
    if int(n_ctx) <= 0:
        n_ctx = int(train_args.get("n_ctx", 2048))
    if int(n_qry) <= 0:
        n_qry = int(train_args.get("n_qry", 64))
    eval_specs = build_eval_datasets(
        mix_config_path,
        seed=int(seed),
        n_ctx=int(n_ctx),
        n_qry=int(n_qry),
        max_samples_per_task=int(max_samples_per_task),
        split_override=split_override,
        task_filter=task_filter,
        eval_sample_mode=str(eval_sample_mode),
    )
    results = [
        evaluate_dataset(
            model,
            spec,
            device=torch_device,
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            control=control,
        )
        for spec in eval_specs
    ]
    overall_tokens = sum(int(r["n_tokens"]) for r in results)
    overall = {
        "ce": sum(float(r["ce"]) * int(r["n_tokens"]) for r in results) / float(max(1, overall_tokens)),
        "token_acc": sum(float(r["token_acc"]) * int(r["n_tokens"]) for r in results) / float(max(1, overall_tokens)),
        "answer_entropy": sum(float(r["answer_entropy"]) * int(r["n_tokens"]) for r in results) / float(max(1, overall_tokens)),
        "n_tokens": int(overall_tokens),
    }
    return {
        "ckpt": str(ckpt_path),
        "mix_config_path": str(mix_config_path),
        "control": str(control),
        "train_run_name": str(train_args.get("run_name", "")),
        "train_global_step": int(ckpt.get("global_step", -1)),
        "eval_sample_mode": str(eval_sample_mode),
        "results": results,
        "overall": overall,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("eval_primitive_answering_tokens")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--mix_config_path", type=str, required=True)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_ctx", type=int, default=-1, help="If <=0, reuse training n_ctx from the checkpoint.")
    p.add_argument("--n_qry", type=int, default=-1, help="If <=0, reuse training n_qry from the checkpoint.")
    p.add_argument("--max_samples_per_task", type=int, default=256)
    p.add_argument("--split_override", type=str, default="")
    p.add_argument("--task_filter", type=str, default="")
    p.add_argument("--eval_sample_mode", type=str, default="head", choices=["head", "random"])
    p.add_argument("--control", type=str, default="correct", choices=list(CONTROL_MODES))
    p.add_argument("--output_json", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_token_eval(
        ckpt_path=str(args.ckpt),
        mix_config_path=str(args.mix_config_path),
        device=str(args.device),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        n_ctx=int(args.n_ctx),
        n_qry=int(args.n_qry),
        max_samples_per_task=int(args.max_samples_per_task),
        split_override=(str(args.split_override).strip() or None),
        task_filter=_parse_task_filter(str(args.task_filter)),
        control=str(args.control),
        eval_sample_mode=str(args.eval_sample_mode),
    )
    print(json.dumps(summary, indent=2))
    if str(args.output_json).strip():
        out_path = Path(str(args.output_json))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
