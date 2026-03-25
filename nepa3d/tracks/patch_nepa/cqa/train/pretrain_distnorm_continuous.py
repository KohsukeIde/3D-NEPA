from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from nepa3d.tracks.patch_nepa.cqa.data.cqa_codec import ASK_DISTANCE, ASK_NORMAL, ASK_VISIBILITY
from nepa3d.tracks.patch_nepa.cqa.data.dataset_cqa_continuous import cqa_continuous_collate_fn
from nepa3d.tracks.patch_nepa.cqa.data.mixed_pretrain_cqa_continuous import build_mixed_pretrain_cqa_continuous
from nepa3d.tracks.patch_nepa.cqa.models.primitive_answering_distnorm_continuous import (
    PrimitiveAnsweringDistNormContinuousModel,
)
from nepa3d.utils.seed import set_seed


class TaskLossBalancer:
    def __init__(
        self,
        *,
        mode: str = "mean",
        ema_momentum: float = 0.99,
        ema_eps: float = 1e-6,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.mode = str(mode)
        self.ema_momentum = float(ema_momentum)
        self.ema_eps = float(ema_eps)
        self.weights = dict(weights or {})
        self.ema = {k: 1.0 for k in self.weights.keys()}

    def combine(self, loss_map: dict[str, torch.Tensor]) -> torch.Tensor:
        active = {k: v for k, v in loss_map.items() if v is not None}
        if not active:
            raise RuntimeError("empty active loss map")
        if self.mode == "mean":
            return torch.stack(list(active.values())).mean()
        if self.mode == "fixed":
            terms = []
            weights = []
            for key, loss in active.items():
                w = float(self.weights.get(key, 1.0))
                terms.append(loss * w)
                weights.append(w)
            return torch.stack(terms).sum() / float(max(sum(weights), self.ema_eps))
        if self.mode == "ema_norm":
            terms = []
            weights = []
            for key, loss in active.items():
                with torch.no_grad():
                    prev = float(self.ema.get(key, 1.0))
                    cur = float(loss.detach().cpu().item())
                    self.ema[key] = self.ema_momentum * prev + (1.0 - self.ema_momentum) * cur
                denom = float(max(self.ema.get(key, 1.0), self.ema_eps))
                w = float(self.weights.get(key, 1.0))
                terms.append((loss / denom) * w)
                weights.append(w)
            return torch.stack(terms).sum() / float(max(sum(weights), self.ema_eps))
        raise ValueError(f"unknown task loss balance mode: {self.mode}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("pretrain_distnorm_continuous")
    p.add_argument("--mix_config_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="runs/cqa")
    p.add_argument("--run_name", type=str, default="cqa_distnorm_continuous")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--lr_scheduler", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument("--warmup_steps", type=int, default=-1)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--save_every_steps", type=int, default=500)
    p.add_argument("--n_ctx", type=int, default=2048)
    p.add_argument("--n_qry", type=int, default=64)
    p.add_argument("--query_order", type=str, default="shuffled")
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
    p.add_argument("--query_type_vocab", type=int, default=6)
    p.add_argument("--generator_depth", type=int, default=2)
    p.add_argument("--distance_floor", type=float, default=0.0)
    p.add_argument("--task_loss_balance", type=str, default="mean", choices=["mean", "ema_norm", "fixed"])
    p.add_argument("--loss_ema_momentum", type=float, default=0.99)
    p.add_argument("--loss_ema_eps", type=float, default=1e-6)
    p.add_argument("--loss_weight_distance", type=float, default=1.0)
    p.add_argument("--loss_weight_ao", type=float, default=1.0)
    p.add_argument("--loss_weight_normal", type=float, default=1.0)
    return p.parse_args()


def _resolve_max_steps(args: argparse.Namespace, steps_per_epoch: int) -> int:
    if int(args.max_steps) > 0:
        return int(args.max_steps)
    return int(max(1, int(args.epochs) * max(1, int(steps_per_epoch))))


def _resolve_warmup_steps(args: argparse.Namespace, max_steps: int) -> int:
    if int(args.warmup_steps) >= 0:
        return int(args.warmup_steps)
    return int(round(float(max_steps) * max(0.0, float(args.warmup_ratio))))


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


def _save_ckpt(*, model: torch.nn.Module, args: argparse.Namespace, save_dir: Path, epoch: int, global_step: int, name: str) -> None:
    ckpt = {
        "arch": "cqa_distnorm_continuous",
        "model": model.state_dict(),
        "args": vars(args),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "target_mode": "continuous_distnorm",
    }
    torch.save(ckpt, save_dir / name)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = torch.device(str(args.device))

    dataset, sampler, info = build_mixed_pretrain_cqa_continuous(
        str(args.mix_config_path),
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
        collate_fn=cqa_continuous_collate_fn,
        drop_last=True,
    )

    model = PrimitiveAnsweringDistNormContinuousModel(
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
        generator_depth=int(args.generator_depth),
        distance_floor=float(args.distance_floor),
    ).to(device)
    opt = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    steps_per_epoch = int(len(loader))
    max_steps = _resolve_max_steps(args, steps_per_epoch)
    warmup_steps = max(0, _resolve_warmup_steps(args, max_steps))
    scheduler = None
    if str(args.lr_scheduler) == "cosine":
        scheduler = optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=lambda st: _scheduler_scale(int(st), args, max_steps, warmup_steps),
        )
    loss_balancer = TaskLossBalancer(
        mode=str(args.task_loss_balance),
        ema_momentum=float(args.loss_ema_momentum),
        ema_eps=float(args.loss_ema_eps),
        weights={
            "distance": float(args.loss_weight_distance),
            "ao": float(args.loss_weight_ao),
            "normal": float(args.loss_weight_normal),
        },
    )

    save_dir = Path(str(args.save_dir)) / str(args.run_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    (save_dir / "mix_info.json").write_text(json.dumps(info, indent=2) + "\n")

    global_step = 0
    stop = False
    for ep in range(int(args.epochs)):
        model.train()
        losses = []
        dist_maes = []
        ao_maes = []
        norm_coses = []
        for batch in loader:
            ctx_xyz = batch["ctx_xyz"].to(device=device, dtype=torch.float32, non_blocking=True)
            qry_xyz = batch["qry_xyz"].to(device=device, dtype=torch.float32, non_blocking=True)
            qry_type = batch["qry_type"].to(device=device, dtype=torch.long, non_blocking=True)
            target_vec = batch["target_vec"].to(device=device, dtype=torch.float32, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            pred = model(ctx_xyz, qry_xyz, qry_type).pred_answer

            task_loss_map: dict[str, torch.Tensor | None] = {
                "distance": None,
                "ao": None,
                "normal": None,
            }
            dist_mask = qry_type == int(ASK_DISTANCE)
            if bool(dist_mask.any()):
                dist_pred = pred[..., 0][dist_mask]
                dist_tgt = target_vec[..., 0][dist_mask]
                dist_loss = F.mse_loss(dist_pred, dist_tgt)
                task_loss_map["distance"] = dist_loss
                dist_maes.append(float((dist_pred.detach() - dist_tgt.detach()).abs().mean().cpu().item()))

            ao_mask = qry_type == int(ASK_VISIBILITY)
            if bool(ao_mask.any()):
                ao_pred = pred[..., 0][ao_mask]
                ao_tgt = target_vec[..., 0][ao_mask]
                ao_loss = F.mse_loss(ao_pred, ao_tgt)
                task_loss_map["ao"] = ao_loss
                ao_maes.append(float((ao_pred.detach() - ao_tgt.detach()).abs().mean().cpu().item()))

            norm_mask = qry_type == int(ASK_NORMAL)
            if bool(norm_mask.any()):
                norm_pred = pred[norm_mask]
                norm_tgt = target_vec[norm_mask]
                norm_cos = F.cosine_similarity(norm_pred, norm_tgt, dim=-1, eps=1e-8)
                norm_loss = (1.0 - norm_cos).mean()
                task_loss_map["normal"] = norm_loss
                norm_coses.append(float(norm_cos.detach().mean().cpu().item()))

            active_losses = {k: v for k, v in task_loss_map.items() if v is not None}
            if not active_losses:
                raise RuntimeError("empty task loss in distnorm continuous train batch")
            loss = loss_balancer.combine(active_losses)
            loss.backward()
            if float(args.max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            opt.step()
            if scheduler is not None:
                scheduler.step()

            loss_val = float(loss.detach().cpu().item())
            losses.append(loss_val)
            if global_step % 100 == 0:
                lr_now = float(opt.param_groups[0]["lr"])
                dist_mae_val = float(np.mean(dist_maes[-1:])) if dist_maes else float("nan")
                ao_mae_val = float(np.mean(ao_maes[-1:])) if ao_maes else float("nan")
                norm_cos_val = float(np.mean(norm_coses[-1:])) if norm_coses else float("nan")
                print(
                    f"ep={ep} step={global_step} loss={loss_val:.6f} "
                    f"dist_mae={dist_mae_val:.6f} ao_mae={ao_mae_val:.6f} "
                    f"norm_cos={norm_cos_val:.6f} lr={lr_now:.8f}"
                )

            global_step += 1
            if int(args.save_every_steps) > 0 and global_step % int(args.save_every_steps) == 0:
                _save_ckpt(model=model, args=args, save_dir=save_dir, epoch=ep, global_step=global_step, name=f"ckpt_step{global_step}.pt")
            if global_step >= max_steps:
                stop = True
                break
        ep_loss = float(np.mean(losses)) if losses else float("nan")
        ep_dist_mae = float(np.mean(dist_maes)) if dist_maes else float("nan")
        ep_ao_mae = float(np.mean(ao_maes)) if ao_maes else float("nan")
        ep_norm_cos = float(np.mean(norm_coses)) if norm_coses else float("nan")
        print(
            f"[epoch] ep={ep} loss_mean={ep_loss:.6f} dist_mae_mean={ep_dist_mae:.6f} "
            f"ao_mae_mean={ep_ao_mae:.6f} norm_cos_mean={ep_norm_cos:.6f}"
        )
        if (ep % max(1, int(args.save_every)) == 0) or stop or (ep == int(args.epochs) - 1):
            _save_ckpt(model=model, args=args, save_dir=save_dir, epoch=ep, global_step=global_step, name=f"ckpt_ep{ep:03d}.pt")
            _save_ckpt(model=model, args=args, save_dir=save_dir, epoch=ep, global_step=global_step, name="ckpt_final.pt")
        if stop:
            break


if __name__ == "__main__":
    main()
