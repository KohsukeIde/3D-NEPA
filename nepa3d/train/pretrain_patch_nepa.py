"""Patch-token NEPA pretraining entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from nepa3d.data.mixed_pretrain import build_mixed_pretrain
from nepa3d.models.patch_nepa import PatchTransformerNepa


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("pretrain_patch_nepa")

    # IO
    p.add_argument("--save_dir", type=str, default="runs_patch_nepa")
    p.add_argument("--run_name", type=str, default="debug")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--save_every", type=int, default=1)

    # Data
    p.add_argument("--mix_config_path", type=str, required=True)
    p.add_argument("--n_point", type=int, default=1024)
    p.add_argument("--n_ray", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--pt_xyz_key", type=str, default="pt_xyz_pool")
    p.add_argument("--pt_dist_key", type=str, default="pt_dist_pool")
    p.add_argument("--ablate_point_dist", type=int, default=0, choices=[0, 1])
    p.add_argument("--pt_sample_mode", type=str, default="random", choices=["random", "fps", "rfps", "grid"])
    p.add_argument("--pt_fps_key", type=str, default="auto")
    p.add_argument("--pt_rfps_m", type=int, default=4096)
    p.add_argument("--point_order_mode", type=str, default="morton", choices=["morton", "fps", "random"])

    # Model (patching)
    p.add_argument("--patch_embed", type=str, default="fps_knn", choices=["serial", "pointgpt", "fps_knn"])
    p.add_argument("--group_size", type=int, default=32)
    p.add_argument("--num_groups", type=int, default=64)
    p.add_argument("--serial_order", type=str, default="morton",
                   choices=["morton", "morton_trans", "z", "z-trans", "random", "identity"])
    p.add_argument("--serial_bits", type=int, default=10)
    p.add_argument("--serial_shuffle_within_patch", type=int, default=0, choices=[0, 1])
    p.add_argument("--morton_bits", type=int, default=10, help="Backward-compat alias; used when --serial_bits is omitted.")
    p.add_argument("--scale_morton", type=float, default=1.0, help="Compatibility arg (unused).")
    p.add_argument("--use_normals", type=int, default=0, choices=[0, 1])

    # Model (transformer)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--drop_path_rate", type=float, default=0.0)
    p.add_argument("--qk_norm", type=int, default=1)
    p.add_argument("--qk_norm_affine", type=int, default=1)
    p.add_argument("--qk_norm_bias", type=int, default=1)
    p.add_argument("--layerscale_value", type=float, default=0.0)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    p.add_argument("--use_gated_mlp", type=int, default=0)
    p.add_argument("--hidden_act", type=str, default="gelu")
    p.add_argument("--backbone_mode", type=str, default="nepa2d", choices=["nepa2d", "vanilla"])

    # Pos (vanilla only)
    p.add_argument("--pos_mode", type=str, default="center_mlp_once", choices=["center_mlp_once", "center_linear", "none"])
    p.add_argument("--pos_mlp_hidden_mult", type=float, default=2.0)
    p.add_argument("--pos_add_times", type=int, default=1)

    # Ray binding
    p.add_argument("--use_ray_patch", type=int, default=0)
    p.add_argument("--ray_hit_threshold", type=float, default=0.5)
    p.add_argument("--ray_miss_t", type=float, default=4.0)
    p.add_argument("--ray_pool_k_max", type=int, default=32)
    p.add_argument("--ray_pool_mode", type=str, default="mean", choices=["mean", "max"])
    p.add_argument("--ray_fuse", type=str, default="add", choices=["add", "concat"])

    # Train
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=0)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def save_ckpt(save_path: Path, model: torch.nn.Module, optimizer: optim.Optimizer, epoch: int, args: argparse.Namespace) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "args": vars(args),
    }
    torch.save(payload, str(save_path))


def _scheduler_scale(epoch: int, args: argparse.Namespace) -> float:
    if int(args.epochs) <= 1:
        return 1.0
    if epoch < int(args.warmup_epochs):
        return float(epoch + 1) / float(max(1, int(args.warmup_epochs)))
    t = float(epoch - int(args.warmup_epochs)) / float(max(1, int(args.epochs) - int(args.warmup_epochs)))
    cosine = 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * t))).item()
    min_scale = float(args.min_lr) / float(args.lr)
    return min_scale + (1.0 - min_scale) * cosine


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=int(args.grad_accum))
    set_seed(int(args.seed))

    serial_bits = int(args.serial_bits) if "serial_bits" in vars(args) else int(args.morton_bits)

    dataset, sampler, _info = build_mixed_pretrain(
        mix_config_path=args.mix_config_path,
        n_point=int(args.n_point),
        n_ray=int(args.n_ray),
        num_workers=int(args.num_workers),
        mode="train",
        return_raw=True,
        pt_xyz_key=str(args.pt_xyz_key),
        pt_dist_key=str(args.pt_dist_key),
        ablate_point_dist=bool(args.ablate_point_dist),
        pt_sample_mode=str(args.pt_sample_mode),
        pt_fps_key=str(args.pt_fps_key),
        pt_rfps_m=int(args.pt_rfps_m),
        point_order_mode=str(args.point_order_mode),
    )
    train_loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
    )

    model = PatchTransformerNepa(
        patch_embed=args.patch_embed,
        n_point=int(args.n_point),
        group_size=int(args.group_size),
        num_groups=args.num_groups,
        serial_order=str(args.serial_order),
        serial_bits=serial_bits,
        serial_shuffle_within_patch=int(args.serial_shuffle_within_patch),
        use_normals=bool(args.use_normals),
        d_model=int(args.d_model),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
        drop_path_rate=float(args.drop_path_rate),
        qk_norm=int(args.qk_norm),
        qk_norm_affine=int(args.qk_norm_affine),
        qk_norm_bias=int(args.qk_norm_bias),
        layerscale_value=float(args.layerscale_value),
        rope_theta=float(args.rope_theta),
        use_gated_mlp=int(args.use_gated_mlp),
        hidden_act=str(args.hidden_act),
        backbone_mode=str(args.backbone_mode),
        pos_mode=str(args.pos_mode),
        pos_mlp_hidden_mult=float(args.pos_mlp_hidden_mult),
        pos_add_times=int(args.pos_add_times),
        use_ray_patch=bool(args.use_ray_patch),
        ray_hit_threshold=float(args.ray_hit_threshold),
        ray_miss_t=float(args.ray_miss_t),
        ray_pool_k_max=int(args.ray_pool_k_max),
        ray_pool_mode=str(args.ray_pool_mode),
        ray_fuse=str(args.ray_fuse),
        use_bos=True,
    )

    optimizer = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: _scheduler_scale(ep, args))
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(ckpt.get("model", ckpt), strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1

    save_root = Path(args.save_dir) / args.run_name
    if accelerator.is_main_process:
        save_root.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(start_epoch, int(args.epochs)):
        model.train()
        sampler_obj = getattr(train_loader, "sampler", None)
        if sampler_obj is not None and hasattr(sampler_obj, "set_epoch"):
            sampler_obj.set_epoch(epoch)

        for batch in train_loader:
            with accelerator.accumulate(model):
                out = model(
                    pt_xyz=batch["pt_xyz"],
                    pt_n=None,
                    pt_dist=batch.get("pt_dist", None),
                    ray_o=batch.get("ray_o", None),
                    ray_d=batch.get("ray_d", None),
                    ray_t=batch.get("ray_t", None),
                    ray_hit=batch.get("ray_hit", None),
                    ray_available=batch.get("ray_available", None),
                    is_causal=True,
                )
                loss = PatchTransformerNepa.nepa_loss(out.z, out.z_hat)
                accelerator.backward(loss)
                if float(args.max_grad_norm) > 0:
                    accelerator.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.is_main_process and (global_step % 50 == 0):
                lr = optimizer.param_groups[0]["lr"]
                print(f"[epoch {epoch:03d} step {global_step:06d}] loss={loss.item():.8e} lr={lr:.2e}")
            global_step += 1

        scheduler.step()
        if accelerator.is_main_process and ((epoch + 1) % int(args.save_every) == 0):
            save_ckpt(save_root / f"ckpt_epoch_{epoch:04d}.pt", accelerator.unwrap_model(model), optimizer, epoch, args)
            save_ckpt(save_root / "ckpt_latest.pt", accelerator.unwrap_model(model), optimizer, epoch, args)

    if accelerator.is_main_process:
        print(f"Done. checkpoints: {save_root}")


if __name__ == "__main__":
    main()
