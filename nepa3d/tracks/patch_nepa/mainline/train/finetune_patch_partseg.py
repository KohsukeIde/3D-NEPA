"""PatchNEPA direct part-segmentation fine-tuning on ShapeNetPart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch import optim
from torch.utils.data import DataLoader

from nepa3d.data.shapenetpart_dataset import SEG_CLASSES, ShapeNetPartAugConfig, ShapeNetPartDataset
from nepa3d.tracks.patch_nepa.mainline.models.patch_partseg import PatchTransformerNepaPartSeg
from nepa3d.tracks.patch_nepa.mainline.train.finetune_patch_cls import (
    _adapt_patchnepa_pretrain_to_patchnepa_classifier,
    _init_wandb,
    _patchnepa_kwargs_from_ckpt,
    _set_seed,
)


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--ckpt_use_ema", type=int, default=0, choices=[0, 1])
    p.add_argument("--save_dir", type=str, default="runs/patchpart")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--use_wandb", type=int, default=0, choices=[0, 1])
    p.add_argument("--wandb_project", type=str, default="patchnepa-shapenetpart")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_group", type=str, default="")
    p.add_argument("--wandb_tags", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    p.add_argument("--root", type=str, default="data/shapenetcore_partanno_segmentation_benchmark_v0_normal")
    p.add_argument("--train_split", type=str, default="trainval", choices=["train", "trainval"])
    p.add_argument("--test_split", type=str, default="test", choices=["test", "val"])
    p.add_argument("--n_point", type=int, default=2048)
    p.add_argument("--use_normals", type=int, default=0, choices=[0, 1])
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--deterministic_eval_sampling", type=int, default=1, choices=[0, 1])

    p.add_argument("--patch_embed", type=str, default="fps_knn")
    p.add_argument("--patch_local_encoder", type=str, default="pointmae_conv")
    p.add_argument("--num_groups", type=int, default=64)
    p.add_argument("--group_size", type=int, default=32)
    p.add_argument("--serial_order", type=str, default="morton")
    p.add_argument("--serial_bits", type=int, default=10)
    p.add_argument("--serial_shuffle_within_patch", type=int, default=0)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--drop_path_rate", type=float, default=0.0)
    p.add_argument("--qk_norm", type=int, default=1, choices=[0, 1])
    p.add_argument("--qk_norm_affine", type=int, default=0, choices=[0, 1])
    p.add_argument("--qk_norm_bias", type=int, default=0, choices=[0, 1])
    p.add_argument("--layerscale_value", type=float, default=1e-5)
    p.add_argument("--rope_theta", type=float, default=100.0)
    p.add_argument("--use_gated_mlp", type=int, default=0, choices=[0, 1])
    p.add_argument("--hidden_act", type=str, default="gelu", choices=["gelu", "silu"])
    p.add_argument("--backbone_mode", type=str, default="nepa2d", choices=["nepa2d", "vanilla", "pointmae"])
    p.add_argument("--use_ray_patch", type=int, default=0, choices=[0, 1])

    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--batch_mode", type=str, default="global", choices=["global", "per_proc"])
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=float, default=10.0)
    p.add_argument("--warmup_start_factor", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--head_dropout", type=float, default=0.5)
    p.add_argument("--label_dim", type=int, default=64)
    p.add_argument("--patchnepa_ft_mode", type=str, default="q_only", choices=["q_only"])
    p.add_argument("--patchnepa_freeze_patch_embed", type=int, default=1, choices=[0, 1])


def _build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))


def _seg_label_to_cat() -> Dict[int, str]:
    out: Dict[int, str] = {}
    for cat, labels in SEG_CLASSES.items():
        for label in labels:
            out[int(label)] = str(cat)
    return out


def evaluate_local(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_normals: bool,
    class_idx_to_cat: Dict[int, str],
) -> Dict[str, float]:
    model.eval()
    seg_label_to_cat = _seg_label_to_cat()
    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(50)]
    total_correct_class = [0 for _ in range(50)]
    shape_ious = {cat: [] for cat in SEG_CLASSES.keys()}
    loss_sum = 0.0
    loss_count = 0

    with torch.no_grad():
        for batch in loader:
            xyz = batch["xyz"].to(device)
            cls_label = batch["cls_label"].to(device)
            seg_label = batch["seg_label"].to(device)
            normals = batch.get("normal", None)
            if use_normals and normals is not None:
                normals = normals.to(device)

            logits = model(xyz, cls_label, normals=normals)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), seg_label.reshape(-1))
            loss_sum += float(loss.item())
            loss_count += 1

            logits_np = logits.detach().cpu().numpy()
            target_np = seg_label.detach().cpu().numpy()
            cls_np = cls_label.detach().cpu().numpy()
            cur_batch_size, n_point, _ = logits_np.shape
            pred_np = np.zeros((cur_batch_size, n_point), dtype=np.int64)

            for i in range(cur_batch_size):
                cat = class_idx_to_cat[int(cls_np[i])]
                valid_parts = np.asarray(SEG_CLASSES[cat], dtype=np.int64)
                pred_np[i, :] = valid_parts[np.argmax(logits_np[i][:, valid_parts], axis=1)]

            total_correct += int(np.sum(pred_np == target_np))
            total_seen += int(cur_batch_size * n_point)
            for label in range(50):
                total_seen_class[label] += int(np.sum(target_np == label))
                total_correct_class[label] += int(np.sum((pred_np == label) & (target_np == label)))

            for i in range(cur_batch_size):
                segp = pred_np[i, :]
                segl = target_np[i, :]
                cat = seg_label_to_cat[int(segl[0])]
                part_ious = []
                for label in SEG_CLASSES[cat]:
                    if (np.sum(segl == label) == 0) and (np.sum(segp == label) == 0):
                        part_ious.append(1.0)
                    else:
                        inter = np.sum((segl == label) & (segp == label))
                        union = np.sum((segl == label) | (segp == label))
                        part_ious.append(float(inter) / float(max(1, union)))
                shape_ious[cat].append(float(np.mean(part_ious)))

    all_shape_ious = [iou for values in shape_ious.values() for iou in values]
    class_avg_iou = float(np.mean([np.mean(v) for v in shape_ious.values() if len(v) > 0]))
    instance_avg_iou = float(np.mean(all_shape_ious)) if all_shape_ious else 0.0
    class_acc = []
    for seen, correct in zip(total_seen_class, total_correct_class):
        class_acc.append(float(correct) / float(max(1, seen)))

    return {
        "acc": float(total_correct) / float(max(1, total_seen)),
        "loss": float(loss_sum) / float(max(1, loss_count)),
        "class_avg_accuracy": float(np.mean(class_acc)),
        "class_avg_iou": class_avg_iou,
        "instance_avg_iou": instance_avg_iou,
    }


def main() -> None:
    parser = argparse.ArgumentParser("PatchNEPA ShapeNetPart fine-tune")
    add_args(parser)
    args = parser.parse_args()

    _set_seed(int(args.seed))
    # The PatchNEPA part-seg head does not exercise every adapted checkpoint
    # branch on every step, so DDP must tolerate temporarily unused parameters.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    save_dir = Path(args.save_dir)
    if args.run_name:
        save_dir = save_dir / args.run_name
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "checkpoints").mkdir(exist_ok=True)
        with open(save_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
    accelerator.wait_for_everyone()
    wandb_run = _init_wandb(args, save_dir, accelerator)

    use_normals = bool(int(args.use_normals))
    aug_cfg = ShapeNetPartAugConfig()
    train_set = ShapeNetPartDataset(
        args.root,
        n_point=int(args.n_point),
        split=str(args.train_split),
        normal_channel=use_normals,
        aug=True,
        aug_cfg=aug_cfg,
        seed=int(args.seed),
        deterministic_eval_sampling=False,
    )
    test_set = ShapeNetPartDataset(
        args.root,
        n_point=int(args.n_point),
        split=str(args.test_split),
        normal_channel=use_normals,
        aug=False,
        aug_cfg=aug_cfg,
        seed=int(args.seed) + 123,
        deterministic_eval_sampling=bool(int(args.deterministic_eval_sampling)),
    )

    world_size = max(1, int(accelerator.num_processes))
    eff_batch = int(args.batch)
    if str(args.batch_mode) == "global" and world_size > 1:
        if eff_batch % world_size != 0:
            raise ValueError(f"--batch {eff_batch} must be divisible by world_size={world_size} when --batch_mode=global")
        eff_batch = eff_batch // world_size

    train_loader = DataLoader(
        train_set,
        batch_size=eff_batch,
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=eff_batch,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
    )

    ckpt_obj: Optional[Dict[str, object]] = None
    pretrain_state: Optional[Dict[str, torch.Tensor]] = None
    pretrain_args: Dict[str, object] = {}
    if args.ckpt:
        ckpt_obj = torch.load(args.ckpt, map_location="cpu")
        state_source = "model"
        if isinstance(ckpt_obj, dict):
            use_ema = bool(int(args.ckpt_use_ema))
            if use_ema and isinstance(ckpt_obj.get("model_ema", None), dict):
                pretrain_state = ckpt_obj["model_ema"]  # type: ignore[index]
                state_source = "model_ema"
            else:
                pretrain_state = ckpt_obj.get("model", ckpt_obj)  # type: ignore[assignment]
            if isinstance(ckpt_obj.get("args", None), dict):
                pretrain_args = ckpt_obj["args"]  # type: ignore[index]
        else:
            pretrain_state = ckpt_obj  # type: ignore[assignment]
        if accelerator.is_main_process:
            print(f"[ckpt] source={state_source} ckpt_use_ema={int(args.ckpt_use_ema)}")

    nepa_kwargs = _patchnepa_kwargs_from_ckpt(pretrain_args, args, pretrain_state)
    model = PatchTransformerNepaPartSeg(
        num_parts=50,
        num_shape_classes=16,
        head_dropout=float(args.head_dropout),
        label_dim=int(args.label_dim),
        ft_sequence_mode=str(args.patchnepa_ft_mode),
        **nepa_kwargs,
    )

    if pretrain_state is not None:
        state_to_load, stats = _adapt_patchnepa_pretrain_to_patchnepa_classifier(pretrain_state, model.state_dict())
        missing, unexpected = model.load_state_dict(state_to_load, strict=False)
        if accelerator.is_main_process:
            print(
                "[ckpt-adapt] patchnepa-pretrain->patchnepa-partseg "
                f"mapped={stats['mapped']} direct={stats['direct']} load={stats['total_to_load']} "
                f"src={stats['src_total']} dst={stats['dst_total']}"
            )
            print(f"Loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}")

    pred_head = getattr(model.core, "pred_head", None)
    if pred_head is not None:
        for p in pred_head.parameters():
            p.requires_grad_(False)
        if accelerator.is_main_process:
            print("[finetune] patchnepa partseg: pred_head frozen/excluded from optimizer")
    if int(args.patchnepa_freeze_patch_embed) == 1:
        for p in model.core.patch_embed.parameters():
            p.requires_grad_(False)
        if accelerator.is_main_process:
            print("[finetune] patchnepa partseg: patch_embed frozen")

    optimizer = _build_optimizer(model, args)
    warmup_epochs = max(0.0, float(args.warmup_epochs))
    if warmup_epochs > 0:
        lr_scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=float(args.warmup_start_factor),
                    total_iters=int(round(warmup_epochs)),
                ),
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, int(args.epochs) - int(round(warmup_epochs))),
                    eta_min=0.0,
                ),
            ],
            milestones=[int(round(warmup_epochs))],
        )
    else:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)), eta_min=0.0)

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    best_metric = -1.0
    best_path = save_dir / "checkpoints" / "best.pt"

    if accelerator.is_main_process:
        print(
            f"PatchPartSeg: train={len(train_set)} test={len(test_set)} "
            f"n_point={args.n_point} use_normals={use_normals} world_size={world_size} "
            f"batch_mode={args.batch_mode} batch_arg={args.batch} batch_effective={eff_batch}\n"
            f"  root={args.root} ft_mode={args.patchnepa_ft_mode} freeze_patch_embed={int(args.patchnepa_freeze_patch_embed)}"
        )

    try:
        for epoch in range(int(args.epochs)):
            model.train()
            train_loss_sum = 0.0
            train_loss_count = 0
            for batch in train_loader:
                xyz = batch["xyz"].to(accelerator.device)
                cls_label = batch["cls_label"].to(accelerator.device)
                seg_label = batch["seg_label"].to(accelerator.device)
                normals = batch.get("normal", None)
                if use_normals and normals is not None:
                    normals = normals.to(accelerator.device)

                logits = model(xyz, cls_label, normals=normals)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), seg_label.reshape(-1))
                accelerator.backward(loss)
                if float(args.grad_clip) > 0:
                    accelerator.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                loss_mean = accelerator.reduce(loss.detach(), reduction="mean")
                if accelerator.is_main_process:
                    train_loss_sum += float(loss_mean.item())
                    train_loss_count += 1

            lr_scheduler.step()
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                model_local = accelerator.unwrap_model(model)
                test_metrics = evaluate_local(
                    model_local,
                    test_loader,
                    accelerator.device,
                    use_normals=use_normals,
                    class_idx_to_cat=test_set.class_idx_to_cat,
                )
                train_loss_avg = float(train_loss_sum / max(1, train_loss_count))
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"[ep {epoch+1:03d}/{args.epochs}] lr={lr_now:.2e} "
                    f"train_loss={train_loss_avg:.4f} "
                    f"test_acc={test_metrics['acc']:.4f} "
                    f"test_ins_miou={test_metrics['instance_avg_iou']:.4f} "
                    f"test_cls_miou={test_metrics['class_avg_iou']:.4f}"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/epoch": float(epoch + 1),
                            "train/lr": float(lr_now),
                            "train/loss": float(train_loss_avg),
                            "test/acc": float(test_metrics["acc"]),
                            "test/class_avg_accuracy": float(test_metrics["class_avg_accuracy"]),
                            "test/class_avg_iou": float(test_metrics["class_avg_iou"]),
                            "test/instance_avg_iou": float(test_metrics["instance_avg_iou"]),
                            "test/loss": float(test_metrics["loss"]),
                            "test/best_instance_avg_iou": float(max(best_metric, test_metrics["instance_avg_iou"])),
                        },
                        step=int(epoch + 1),
                    )
                if test_metrics["instance_avg_iou"] > best_metric:
                    best_metric = float(test_metrics["instance_avg_iou"])
                    torch.save({"model": accelerator.unwrap_model(model).state_dict(), "args": vars(args)}, best_path)
                    print(f"  saved best -> {best_path} (instance_avg_iou={best_metric:.4f})")
            accelerator.wait_for_everyone()
    finally:
        pass

    accelerator.wait_for_everyone()
    if best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(ckpt["model"], strict=True)
    accelerator.wait_for_everyone()

    test_metrics = None
    if accelerator.is_main_process:
        test_metrics = evaluate_local(
            accelerator.unwrap_model(model),
            test_loader,
            accelerator.device,
            use_normals=use_normals,
            class_idx_to_cat=test_set.class_idx_to_cat,
        )
        print(
            f"TEST acc={test_metrics['acc']:.4f} "
            f"class_avg_iou={test_metrics['class_avg_iou']:.4f} "
            f"instance_avg_iou={test_metrics['instance_avg_iou']:.4f} "
            f"loss={test_metrics['loss']:.4f}"
        )
        with open(save_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "final/acc": float(test_metrics["acc"]),
                    "final/class_avg_accuracy": float(test_metrics["class_avg_accuracy"]),
                    "final/class_avg_iou": float(test_metrics["class_avg_iou"]),
                    "final/instance_avg_iou": float(test_metrics["instance_avg_iou"]),
                    "final/loss": float(test_metrics["loss"]),
                },
                step=int(args.epochs + 1),
            )
            wandb_run.finish()

    accelerator.wait_for_everyone()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    if hasattr(accelerator, "end_training"):
        accelerator.end_training()


if __name__ == "__main__":
    main()
