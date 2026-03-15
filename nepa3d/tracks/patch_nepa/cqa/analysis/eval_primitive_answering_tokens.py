from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

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
from nepa3d.tracks.patch_nepa.cqa.data.dataset_cqa import V2PrimitiveCQADataset, cqa_collate_fn
from nepa3d.data.mixed_pretrain import load_mix_config
from nepa3d.data.modelnet40_index import list_npz
from nepa3d.tracks.patch_nepa.cqa.models.primitive_answering import PrimitiveAnsweringModel

CONTROL_MODES = ("correct", "no_context", "shuffled_context", "wrong_type", "shuffled_query")


@dataclass(frozen=True)
class EvalDatasetSpec:
    name: str
    split: str
    cache_root: str
    context_source: str
    dataset: V2PrimitiveCQADataset


def _parse_task_filter(text: str) -> set[str]:
    s = str(text).strip()
    if not s:
        return set()
    return {x.strip() for x in s.split(",") if x.strip()}


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
            paths = paths[: int(max_samples_per_task)]
        ds = V2PrimitiveCQADataset(
            paths,
            task_name=task_name,
            context_source=str(s.extra.get("context_source", "surf")),
            n_ctx=int(s.extra.get("n_ctx", n_ctx)),
            n_qry=int(s.extra.get("n_qry", n_qry)),
            seed=int(seed),
            mode="eval",
        )
        out.append(
            EvalDatasetSpec(
                name=task_name,
                split=split,
                cache_root=str(s.cache_root),
                context_source=str(s.extra.get("context_source", "surf")),
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


def apply_control(batch: Dict[str, Any], control: str) -> Dict[str, Any]:
    control = str(control)
    if control not in CONTROL_MODES:
        raise KeyError(f"unknown control={control}")
    out = _clone_batch(batch)
    if control == "correct":
        return out
    if control == "no_context":
        out["ctx_xyz"] = torch.zeros_like(out["ctx_xyz"])
        return out
    if control == "shuffled_context":
        if int(out["ctx_xyz"].shape[0]) > 1:
            out["ctx_xyz"] = torch.roll(out["ctx_xyz"], shifts=1, dims=0)
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
    pred_hist = torch.zeros((ANSWER_VOCAB_SIZE,), dtype=torch.long)
    pbar = tqdm(loader, desc=f"{spec.name}:{control}", leave=False)
    with torch.inference_mode():
        for batch in pbar:
            batch = apply_control(batch, control)
            ctx_xyz = batch["ctx_xyz"].to(device=device, dtype=torch.float32, non_blocking=True)
            qry_xyz = batch["qry_xyz"].to(device=device, dtype=torch.float32, non_blocking=True)
            qry_type = batch["qry_type"].to(device=device, dtype=torch.long, non_blocking=True)
            answer_code = batch["answer_code"].to(device=device, dtype=torch.long, non_blocking=True)

            out = model(ctx_xyz=ctx_xyz, qry_xyz=qry_xyz, qry_type=qry_type, answer_code=answer_code)
            masked_logits = mask_logits_for_query_type(out.logits, qry_type)
            flat_logits = masked_logits.reshape(-1, int(masked_logits.shape[-1]))
            flat_target = answer_code.reshape(-1)
            loss = torch.nn.functional.cross_entropy(flat_logits, flat_target, reduction="sum")

            pred = masked_logits.argmax(dim=-1)
            correct = (pred == answer_code).sum()
            probs = masked_logits.softmax(dim=-1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).sum()

            n_tok = int(answer_code.numel())
            total_tokens += n_tok
            total_loss += float(loss.detach().cpu())
            total_acc += float(correct.detach().cpu())
            total_entropy += float(entropy.detach().cpu())
            pred_hist += torch.bincount(pred.reshape(-1).detach().cpu(), minlength=ANSWER_VOCAB_SIZE)

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
    return {
        "task_name": spec.name,
        "split": spec.split,
        "cache_root": spec.cache_root,
        "context_source": spec.context_source,
        "control": control,
        "query_type_id": qtype,
        "query_type_name": QUERY_TYPE_NAMES[qtype],
        "answer_range": [int(lo), int(hi)],
        "n_samples": int(len(spec.dataset)),
        "n_tokens": int(total_tokens),
        "ce": total_loss / float(total_tokens),
        "token_acc": total_acc / float(total_tokens),
        "answer_entropy": total_entropy / float(total_tokens),
        "pred_hist_nonzero": nonzero_hist,
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
    )
    print(json.dumps(summary, indent=2))
    if str(args.output_json).strip():
        out_path = Path(str(args.output_json))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
