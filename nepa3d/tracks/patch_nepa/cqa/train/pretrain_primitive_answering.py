from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from nepa3d.tracks.patch_nepa.cqa.data.cqa_codec import (
    ANSWER_VOCAB_SIZE,
    CQA_VOCAB_VERSION,
    QUERY_TYPE_VOCAB_SIZE,
    mask_logits_for_query_type,
)
from nepa3d.tracks.patch_nepa.cqa.data.dataset_cqa import QUERY_ORDER_MODES, cqa_collate_fn
from nepa3d.tracks.patch_nepa.cqa.data.mixed_pretrain_cqa import build_mixed_pretrain_cqa
from nepa3d.tracks.patch_nepa.cqa.models.primitive_answering import PrimitiveAnsweringModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("pretrain_primitive_answering")
    p.add_argument("--mix_config_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="runs/cqa")
    p.add_argument("--run_name", type=str, default="cqa_debug")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--max_steps", type=int, default=-1, help="If >0, stop after this many optimizer steps.")
    p.add_argument("--lr_scheduler", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument("--warmup_steps", type=int, default=-1, help="If >=0, overrides warmup_ratio.")
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--save_every_steps", type=int, default=0)
    p.add_argument("--use_wandb", type=int, default=0, choices=[0, 1])
    p.add_argument("--wandb_project", type=str, default="patchnepa-cqa-pretrain")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_group", type=str, default="")
    p.add_argument("--wandb_tags", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_log_every", type=int, default=10)
    p.add_argument("--wandb_dir", type=str, default="")
    p.add_argument("--n_ctx", type=int, default=2048)
    p.add_argument("--n_qry", type=int, default=64)
    p.add_argument(
        "--query_order",
        type=str,
        default="shuffled",
        choices=list(QUERY_ORDER_MODES),
        help="Ordering applied to sampled queries before decoding.",
    )

    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--drop_path", type=float, default=0.0)
    p.add_argument("--backbone_impl", type=str, default="nepa2d", choices=["nepa2d", "pointmae", "legacy"])
    p.add_argument("--num_groups", type=int, default=64)
    p.add_argument("--group_size", type=int, default=32)
    p.add_argument("--patch_center_mode", type=str, default="fps", choices=["fps", "first"])
    p.add_argument("--patch_fps_random_start", type=int, default=1, choices=[0, 1])
    p.add_argument("--local_encoder", type=str, default="pointmae_conv", choices=["mlp", "pointmae_conv"])
    p.add_argument("--answer_vocab", type=int, default=ANSWER_VOCAB_SIZE)
    p.add_argument("--query_type_vocab", type=int, default=QUERY_TYPE_VOCAB_SIZE)
    p.add_argument("--generator_depth", type=int, default=2)
    p.add_argument(
        "--answer_factorization",
        type=str,
        default="ar",
        choices=["ar", "parallel"],
        help="Answer decoding factorization: causal AR or non-AR parallel baseline.",
    )
    return p.parse_args()


def _resolve_max_steps(args: argparse.Namespace, steps_per_epoch: int) -> int:
    if int(args.max_steps) > 0:
        return int(args.max_steps)
    return int(max(1, int(args.epochs) * max(1, int(steps_per_epoch))))


def _resolve_warmup_steps(args: argparse.Namespace, max_steps: int) -> int:
    if int(args.warmup_steps) >= 0:
        return int(args.warmup_steps)
    ratio = max(0.0, float(args.warmup_ratio))
    return int(round(float(max_steps) * ratio))


def _scheduler_scale(step: int, args: argparse.Namespace, max_steps: int, warmup_steps: int) -> float:
    if str(args.lr_scheduler) != "cosine":
        return 1.0
    s = min(max(0, int(step)), int(max_steps))
    if warmup_steps > 0 and s < warmup_steps:
        return float(s + 1) / float(max(1, warmup_steps))

    denom = max(1, int(max_steps) - max(0, int(warmup_steps)))
    t = float(s - max(0, int(warmup_steps))) / float(denom)
    t = min(max(t, 0.0), 1.0)
    min_scale = float(args.min_lr) / float(args.lr)
    cosv = 0.5 * (1.0 + math.cos(math.pi * t))
    return min_scale + (1.0 - min_scale) * cosv


def _masked_ce_metrics(logits: torch.Tensor, target: torch.Tensor, qry_type: torch.Tensor) -> dict[str, torch.Tensor]:
    masked_logits = mask_logits_for_query_type(logits, qry_type)
    flat_logits = masked_logits.reshape(-1, int(masked_logits.shape[-1]))
    flat_target = target.reshape(-1)
    loss = F.cross_entropy(flat_logits, flat_target)

    pred = masked_logits.argmax(dim=-1)
    token_acc = (pred == target).float().mean()

    probs = masked_logits.softmax(dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
    return {
        "loss": loss,
        "token_acc": token_acc,
        "entropy": entropy,
        "masked_logits": masked_logits,
    }


def _save_ckpt(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    args: argparse.Namespace,
    save_dir: Path,
    epoch: int,
    global_step: int,
    name: str,
) -> None:
    ckpt = {
        "model": accelerator.unwrap_model(model).state_dict(),
        "args": vars(args),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "vocab_version": CQA_VOCAB_VERSION,
    }
    torch.save(ckpt, save_dir / name)


def _init_wandb(args: argparse.Namespace, accelerator: Accelerator):
    if int(args.use_wandb) != 1:
        return None
    if not accelerator.is_main_process:
        return None
    try:
        import wandb  # type: ignore
    except Exception as e:
        accelerator.print(f"[wandb] disabled: import failed ({e})")
        return None
    tags = [t.strip() for t in str(args.wandb_tags).split(",") if t.strip()]
    run_name = str(args.wandb_run_name).strip() or str(args.run_name)
    group = str(args.wandb_group).strip() or None
    entity = str(args.wandb_entity).strip() or None
    mode = str(args.wandb_mode).strip()
    run_dir = str(args.wandb_dir).strip() or None
    if mode == "disabled":
        accelerator.print("[wandb] disabled by mode=disabled")
        return None
    try:
        if run_dir is not None:
            Path(run_dir).mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            project=str(args.wandb_project),
            entity=entity,
            name=run_name,
            group=group,
            tags=tags,
            mode=mode,
            dir=run_dir,
            config=vars(args),
        )
        accelerator.print(
            f"[wandb] enabled project={args.wandb_project} run={run_name} "
            f"group={group if group else '-'} mode={mode} dir={run_dir if run_dir else '-'}"
        )
        return run
    except Exception as e:
        accelerator.print(f"[wandb] disabled: init failed ({e})")
        return None


def main() -> None:
    args = parse_args()
    ddp = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp])
    set_seed(int(args.seed))

    dataset, sampler, info = build_mixed_pretrain_cqa(
        args.mix_config_path,
        n_ctx=int(args.n_ctx),
        n_qry=int(args.n_qry),
        mode="train",
        eval_seed=int(args.seed),
        query_order=str(args.query_order),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=cqa_collate_fn,
        drop_last=True,
    )

    model = PrimitiveAnsweringModel(
        d_model=int(args.d_model),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
        drop_path=float(args.drop_path),
        backbone_impl=str(args.backbone_impl),
        num_groups=int(args.num_groups),
        group_size=int(args.group_size),
        patch_center_mode=str(args.patch_center_mode),
        patch_fps_random_start=bool(args.patch_fps_random_start),
        local_encoder=str(args.local_encoder),
        query_type_vocab=int(args.query_type_vocab),
        answer_vocab=int(args.answer_vocab),
        generator_depth=int(args.generator_depth),
        answer_factorization=str(args.answer_factorization),
    )
    opt = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    model, opt, loader = accelerator.prepare(model, opt, loader)
    steps_per_epoch = int(len(loader))
    max_steps = _resolve_max_steps(args, steps_per_epoch)
    warmup_steps = max(0, _resolve_warmup_steps(args, max_steps))
    scheduler = None
    if str(args.lr_scheduler) == "cosine":
        scheduler = optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=lambda st: _scheduler_scale(int(st), args, max_steps, warmup_steps),
        )

    save_dir = Path(args.save_dir) / args.run_name
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        with open(save_dir / "mix_info.json", "w") as f:
            json.dump(info, f, indent=2)
        with open(save_dir / "vocab_spec.json", "w") as f:
            json.dump(
                {
                    "vocab_version": CQA_VOCAB_VERSION,
                    "answer_vocab_size": int(args.answer_vocab),
                    "query_type_vocab": int(args.query_type_vocab),
                },
                f,
                indent=2,
            )
        with open(save_dir / "train_runtime.json", "w") as f:
            json.dump(
                {
                    "steps_per_epoch": int(steps_per_epoch),
                    "effective_max_steps": int(max_steps),
                    "warmup_steps": int(warmup_steps),
                    "lr_scheduler": str(args.lr_scheduler),
                    "max_grad_norm": float(args.max_grad_norm),
                },
                f,
                indent=2,
            )
    wandb_run = _init_wandb(args, accelerator)
    wandb_log_every = max(1, int(args.wandb_log_every))

    global_step = 0
    last_epoch = 0
    stop = False
    for ep in range(int(args.epochs)):
        last_epoch = int(ep + 1)
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(ep)
        model.train()
        pbar = tqdm(loader, disable=not accelerator.is_main_process, desc=f"ep {ep:03d}")
        for batch in pbar:
            out = model(
                ctx_xyz=batch["ctx_xyz"],
                qry_xyz=batch["qry_xyz"],
                qry_type=batch["qry_type"],
                answer_code=batch["answer_code"],
            )
            metrics = _masked_ce_metrics(out.logits, batch["answer_code"], batch["qry_type"])
            loss = metrics["loss"]
            accelerator.backward(loss)
            if float(args.max_grad_norm) > 0:
                accelerator.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            opt.step()
            if scheduler is not None:
                scheduler.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1
            if accelerator.is_main_process:
                cur_lr = float(opt.param_groups[0]["lr"])
                pbar.set_postfix(
                    loss=float(loss.detach().cpu()),
                    acc=float(metrics["token_acc"].detach().cpu()),
                    ent=float(metrics["entropy"].detach().cpu()),
                    lr=cur_lr,
                    step=int(global_step),
                )
                if wandb_run is not None and (global_step % wandb_log_every == 0):
                    wandb_run.log(
                        {
                            "train/loss": float(loss.detach().cpu()),
                            "train/token_acc": float(metrics["token_acc"].detach().cpu()),
                            "train/answer_entropy": float(metrics["entropy"].detach().cpu()),
                            "train/lr": cur_lr,
                            "train/epoch": float(ep + 1),
                        },
                        step=int(global_step),
                    )

            if accelerator.is_main_process and int(args.save_every_steps) > 0 and (global_step % int(args.save_every_steps) == 0):
                _save_ckpt(
                    accelerator=accelerator,
                    model=model,
                    args=args,
                    save_dir=save_dir,
                    epoch=int(ep),
                    global_step=int(global_step),
                    name=f"ckpt_step{global_step:06d}.pt",
                )

            if int(global_step) >= int(max_steps):
                stop = True
                break

        if accelerator.is_main_process and ((ep + 1) % int(args.save_every) == 0 or (ep + 1) == int(args.epochs)):
            _save_ckpt(
                accelerator=accelerator,
                model=model,
                args=args,
                save_dir=save_dir,
                epoch=int(ep + 1),
                global_step=int(global_step),
                name=f"ckpt_ep{ep+1:04d}.pt",
            )
            _save_ckpt(
                accelerator=accelerator,
                model=model,
                args=args,
                save_dir=save_dir,
                epoch=int(ep + 1),
                global_step=int(global_step),
                name="ckpt_latest.pt",
            )
        if stop:
            break

    if accelerator.is_main_process:
        _save_ckpt(
            accelerator=accelerator,
            model=model,
            args=args,
            save_dir=save_dir,
            epoch=int(last_epoch),
            global_step=int(global_step),
            name="ckpt_final.pt",
        )
        _save_ckpt(
            accelerator=accelerator,
            model=model,
            args=args,
            save_dir=save_dir,
            epoch=int(last_epoch),
            global_step=int(global_step),
            name="ckpt_latest.pt",
        )
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
