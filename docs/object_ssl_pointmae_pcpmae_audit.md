# Point-MAE / PCP-MAE Object SSL Audit

- phase: `final`
- git commit: `fd9da67bb59f04e034c8f98fa1e9f42244801318`
- root: `/mnt/urashima/users/minesawa/home-offload/ssl/3D-NEPA`
- Point-MAE root: `/mnt/urashima/users/minesawa/home-offload/ssl/3D-NEPA/Point-MAE`
- PCP-MAE root: `/mnt/urashima/users/minesawa/home-offload/ssl/3D-NEPA/PCP-MAE`
- Point-MAE python: `/home/minesawa/anaconda3/envs/scenemi/bin/python`
- PCP-MAE python: `/home/minesawa/anaconda3/envs/scenemi/bin/python`
- GPUs: `0,1,2,3`
- ScanObjectNN root: `/mnt/urashima/users/minesawa/home-offload/ssl/3D-NEPA/data/ScanObjectNN/h5_files`
- ShapeNetPart root: `/mnt/urashima/users/minesawa/home-offload/ssl/3D-NEPA/data/shapenetcore_partanno_segmentation_benchmark_v0_normal`

## Checkpoint Manifest

```json
pcpmae:
  obj_bg:
    path: /mnt/urashima/users/minesawa/home-offload/ssl/3D-NEPA/results/object_ssl_pointmae_pcpmae/checkpoints/pcpmae/obj_bg/ckpt-best-95.53.pth
    sha256: 9629b3e166e782a6da4671552bf16b91c7e7d72155ceab35a4aba613ae75f7c5
    url: https://drive.google.com/drive/folders/1He3bUfXJ36nwAcGbQE4I9tOUnxjEmfae?usp=drive_link
  obj_only:
    path: /mnt/urashima/users/minesawa/home-offload/ssl/3D-NEPA/results/object_ssl_pointmae_pcpmae/checkpoints/pcpmae/obj_only/ckpt-best-93.97-ckpt-300-finetune.pth
    sha256: c35925e8eba8fa4d904158fff64d93bc43f66dbfbac0d2f977d65d66c0a5f62f
    url: https://drive.google.com/drive/folders/1xuJlAwSYMwc0bTKvnzaoePggMrLqQw3r?usp=drive_link
  pb_t50_rs:
    path: /mnt/urashima/users/minesawa/home-offload/ssl/3D-NEPA/results/object_ssl_pointmae_pcpmae/checkpoints/pcpmae/pb_t50_rs/ckpt-best-90.35.pth
    sha256: 86fa26179907b9bfb0fcda5863a88992ada294ce45df4a19024c7003426e5c0d
    url: https://drive.google.com/drive/folders/1YWJrThywU6G4yoUn4-GvtnHH_bi_Uprp?usp=drive_link
  pretrain:
    path: /mnt/urashima/users/minesawa/home-offload/ssl/3D-NEPA/results/object_ssl_pointmae_pcpmae/checkpoints/pcpmae/pretrain/PCP-MAE-275.pth
    sha256: 56cdd124267a949a6aec811d6adb23565866954b4f64c25e9f62707ea1ba32ec
    url: https://drive.google.com/drive/folders/1smQMWBBEdMOXVAzIBs3xCBrcyQDg8_GS?usp=drive_link
pointmae:
  obj_bg:
    note: 'Mapped by checkpoint internal metric: 90.0172 acc checkpoint.'
    path: /home/minesawa/ssl/3D-NEPA/results/object_ssl_pointmae_pcpmae/checkpoints/pointmae/scan_objonly.pth
    sha256: d861f8b8caa612f29c5eb26ef8d2a613a36d57b35670efaa029f9d624b040925
    url: https://github.com/Pang-Yatian/Point-MAE/releases/download/main/scan_objonly.pth
  obj_only:
    note: 'Mapped by checkpoint internal metric: 88.2960 acc checkpoint.'
    path: /home/minesawa/ssl/3D-NEPA/results/object_ssl_pointmae_pcpmae/checkpoints/pointmae/scan_objbg.pth
    sha256: 150a4cb82fa91f21e3654c68d2d16f9755dec69e63bb99858197dcfe9e74f388
    url: https://github.com/Pang-Yatian/Point-MAE/releases/download/main/scan_objbg.pth
  part:
    path: /mnt/urashima/users/minesawa/home-offload/ssl/3D-NEPA/results/object_ssl_pointmae_pcpmae/checkpoints/pointmae/part_seg.pth
    sha256: 0a479fabb504d39462611943489de277aeb4513b3485ef194ce4eca896e95a9f
    url: https://github.com/Pang-Yatian/Point-MAE/releases/download/main/part_seg.pth
  pb_t50_rs:
    path: /mnt/urashima/users/minesawa/home-offload/ssl/3D-NEPA/results/object_ssl_pointmae_pcpmae/checkpoints/pointmae/scan_hardest.pth
    sha256: 1977b4dcbbafe45a61c42d5f676194fceb5cbc3aabb7ef3623b7f11f0a930a9f
    url: https://github.com/Pang-Yatian/Point-MAE/releases/download/main/scan_hardest.pth
  pretrain:
    path: /mnt/urashima/users/minesawa/home-offload/ssl/3D-NEPA/results/object_ssl_pointmae_pcpmae/checkpoints/pointmae/pretrain.pth
    sha256: 27ded932bb0a2625d5a8eb006df199b2578598c774aee6d86b985300b6a5fd20
    url: https://github.com/Pang-Yatian/Point-MAE/releases/download/main/pretrain.pth

```

## Checkpoint Notes

- Point-MAE official ScanObjectNN release filenames were mapped by checkpoint-internal metrics: `scan_objonly.pth` is used for `obj_bg` and `scan_objbg.pth` is used for `obj_only` in these diagnostics.

## Existing support/top-k utilities

- Reused protocol: `PointGPT/tools/eval_scanobjectnn_support_stress.py` structured/random/xyz-zero semantics.
- New adapters: `scripts/object_ssl/eval_scanobjectnn_mae.py`, `scripts/object_ssl/eval_shapenetpart_mae.py`.
- ShapeNetPart adapter computes support metrics on unique retained original point indices and aggregates repeated forward logits by original index.

## Scene-side audit note

- Static audit target: `/home/minesawa/ssl/concerto-shortcut-mvp/tools/concerto_projection_shortcut/eval_masking_battery.py`.
- Scene-side masking results distinguish `retained` from `full_nn` score spaces; cite this label explicitly when using scene Q3 rows.
