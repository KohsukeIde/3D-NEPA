# Point-MAE / PCP-MAE Object SSL Diagnostics Summary

These diagnostics test whether the object-side support and readout ambiguities persist beyond the PointGPT scaffold.
They do not by themselves prove a universal object-level 3D SSL failure.

ShapeNetPart support rows report metrics on unique retained original point indices.
When retained support is resampled to satisfy the fixed-size forward pass, logits are averaged back to each original retained point before mIoU/top-k computation.
`clean_subset_score` is the clean full-input prediction evaluated on the same retained point set, so `damage_pp` is a matched retained-subset delta for ShapeNetPart rows.

- git commit: `82a07015ea865349cdd6a537cfaca3128bce9086`
- result dir: `/mnt/urashima/users/minesawa/home-offload/ssl/3D-NEPA/results/object_ssl_pointmae_pcpmae`
- log dir: `/mnt/urashima/users/minesawa/home-offload/ssl/3D-NEPA/logs/object_ssl_pointmae_pcpmae`

## Checkpoints

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

## Clean Reproduction

| model | task | split | selection_protocol | metric_name | score |
|---|---|---|---|---|---|
| pcpmae | scanobjectnn | obj_bg | heldout_selection | top1 | 92.7711 |
| pcpmae | scanobjectnn | obj_bg | heldout_selection | top2_hit | 97.9346 |
| pcpmae | scanobjectnn | obj_bg | heldout_selection | top5_hit | 99.4837 |
| pcpmae | scanobjectnn | obj_bg | official_checkpoint | top1 | 94.4923 |
| pcpmae | scanobjectnn | obj_bg | official_checkpoint | top2_hit | 97.7625 |
| pcpmae | scanobjectnn | obj_bg | official_checkpoint | top5_hit | 99.6558 |
| pcpmae | scanobjectnn | obj_only | heldout_selection | top1 | 92.2547 |
| pcpmae | scanobjectnn | obj_only | heldout_selection | top2_hit | 97.2461 |
| pcpmae | scanobjectnn | obj_only | heldout_selection | top5_hit | 99.1394 |
| pcpmae | scanobjectnn | obj_only | official_checkpoint | top1 | 93.2874 |
| pcpmae | scanobjectnn | obj_only | official_checkpoint | top2_hit | 96.7298 |
| pcpmae | scanobjectnn | obj_only | official_checkpoint | top5_hit | 99.3115 |
| pcpmae | scanobjectnn | pb_t50_rs | heldout_selection | top1 | 88.8619 |
| pcpmae | scanobjectnn | pb_t50_rs | heldout_selection | top2_hit | 95.7321 |
| pcpmae | scanobjectnn | pb_t50_rs | heldout_selection | top5_hit | 98.8550 |
| pcpmae | scanobjectnn | pb_t50_rs | official_checkpoint | top1 | 89.5212 |
| pcpmae | scanobjectnn | pb_t50_rs | official_checkpoint | top2_hit | 96.2179 |
| pcpmae | scanobjectnn | pb_t50_rs | official_checkpoint | top5_hit | 99.0632 |
| pointmae | scanobjectnn | obj_bg | heldout_selection | top1 | 85.8864 |
| pointmae | scanobjectnn | obj_bg | heldout_selection | top2_hit | 94.4923 |
| pointmae | scanobjectnn | obj_bg | heldout_selection | top5_hit | 98.9673 |
| pointmae | scanobjectnn | obj_bg | official_checkpoint | top1 | 89.3287 |
| pointmae | scanobjectnn | obj_bg | official_checkpoint | top2_hit | 95.5250 |
| pointmae | scanobjectnn | obj_bg | official_checkpoint | top5_hit | 99.1394 |
| pointmae | scanobjectnn | obj_only | heldout_selection | top1 | 86.4028 |
| pointmae | scanobjectnn | obj_only | heldout_selection | top2_hit | 94.3201 |
| pointmae | scanobjectnn | obj_only | heldout_selection | top5_hit | 98.2788 |
| pointmae | scanobjectnn | obj_only | official_checkpoint | top1 | 86.7470 |
| pointmae | scanobjectnn | obj_only | official_checkpoint | top2_hit | 95.0086 |
| pointmae | scanobjectnn | obj_only | official_checkpoint | top5_hit | 98.1067 |
| pointmae | scanobjectnn | pb_t50_rs | heldout_selection | top1 | 83.1020 |
| pointmae | scanobjectnn | pb_t50_rs | heldout_selection | top2_hit | 92.0541 |
| pointmae | scanobjectnn | pb_t50_rs | heldout_selection | top5_hit | 97.5711 |
| pointmae | scanobjectnn | pb_t50_rs | official_checkpoint | top1 | 84.8022 |
| pointmae | scanobjectnn | pb_t50_rs | official_checkpoint | top2_hit | 92.6787 |
| pointmae | scanobjectnn | pb_t50_rs | official_checkpoint | top5_hit | 98.0569 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain | class_avg_miou | 84.3227 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain | instance_avg_miou | 85.9889 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain | point_top1 | 94.7296 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain | point_top2_hit | 99.2357 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain | point_top5_hit | 100.0000 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_ckpt300 | class_avg_miou | 84.2104 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_ckpt300 | instance_avg_miou | 85.8322 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_ckpt300 | point_top1 | 94.7102 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_ckpt300 | point_top2_hit | 99.2322 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_ckpt300 | point_top5_hit | 100.0000 |
| pointmae | shapenetpart | test | official_checkpoint | class_avg_miou | 84.1091 |
| pointmae | shapenetpart | test | official_checkpoint | instance_avg_miou | 86.0791 |
| pointmae | shapenetpart | test | official_checkpoint | point_top1 | 94.6506 |
| pointmae | shapenetpart | test | official_checkpoint | point_top2_hit | 99.2573 |
| pointmae | shapenetpart | test | official_checkpoint | point_top5_hit | 100.0000 |

## Q3 Support Perturbations

| model | task | split | condition | metric_name | score | clean_subset_score | damage_pp | mean_retained_unique_points | mean_repeated_forward_points |
|---|---|---|---|---|---|---|---|---|---|
| pcpmae | scanobjectnn | obj_bg | clean | Top-1 (%) | 92.7711 |  | 0.0000 |  |  |
| pcpmae | scanobjectnn | obj_bg | random_keep80 | Top-1 (%) | 91.3942 |  | 1.3769 |  |  |
| pcpmae | scanobjectnn | obj_bg | random_keep50 | Top-1 (%) | 82.6162 |  | 10.1549 |  |  |
| pcpmae | scanobjectnn | obj_bg | random_keep20 | Top-1 (%) | 30.8089 |  | 61.9621 |  |  |
| pcpmae | scanobjectnn | obj_bg | random_keep10 | Top-1 (%) | 13.4251 |  | 79.3460 |  |  |
| pcpmae | scanobjectnn | obj_bg | structured_keep80 | Top-1 (%) | 88.6403 |  | 4.1308 |  |  |
| pcpmae | scanobjectnn | obj_bg | structured_keep50 | Top-1 (%) | 64.7160 |  | 28.0551 |  |  |
| pcpmae | scanobjectnn | obj_bg | structured_keep20 | Top-1 (%) | 17.7281 |  | 75.0430 |  |  |
| pcpmae | scanobjectnn | obj_bg | structured_keep10 | Top-1 (%) | 11.1876 |  | 81.5835 |  |  |
| pcpmae | scanobjectnn | obj_bg | xyz_zero | Top-1 (%) | 9.2943 |  | 83.4768 |  |  |
| pcpmae | scanobjectnn | obj_bg | clean | Top-1 (%) | 94.4923 |  | 0.0000 |  |  |
| pcpmae | scanobjectnn | obj_bg | random_keep80 | Top-1 (%) | 94.3201 |  | 0.1721 |  |  |
| pcpmae | scanobjectnn | obj_bg | random_keep50 | Top-1 (%) | 87.7797 |  | 6.7126 |  |  |
| pcpmae | scanobjectnn | obj_bg | random_keep20 | Top-1 (%) | 21.8589 |  | 72.6334 |  |  |
| pcpmae | scanobjectnn | obj_bg | random_keep10 | Top-1 (%) | 17.2117 |  | 77.2806 |  |  |
| pcpmae | scanobjectnn | obj_bg | structured_keep80 | Top-1 (%) | 90.0172 |  | 4.4750 |  |  |
| pcpmae | scanobjectnn | obj_bg | structured_keep50 | Top-1 (%) | 67.9862 |  | 26.5060 |  |  |
| pcpmae | scanobjectnn | obj_bg | structured_keep20 | Top-1 (%) | 18.4165 |  | 76.0757 |  |  |
| pcpmae | scanobjectnn | obj_bg | structured_keep10 | Top-1 (%) | 14.4578 |  | 80.0344 |  |  |
| pcpmae | scanobjectnn | obj_bg | xyz_zero | Top-1 (%) | 9.2943 |  | 85.1979 |  |  |
| pcpmae | scanobjectnn | obj_only | clean | Top-1 (%) | 92.2547 |  | 0.0000 |  |  |
| pcpmae | scanobjectnn | obj_only | random_keep80 | Top-1 (%) | 91.0499 |  | 1.2048 |  |  |
| pcpmae | scanobjectnn | obj_only | random_keep50 | Top-1 (%) | 88.4682 |  | 3.7866 |  |  |
| pcpmae | scanobjectnn | obj_only | random_keep20 | Top-1 (%) | 46.6437 |  | 45.6110 |  |  |
| pcpmae | scanobjectnn | obj_only | random_keep10 | Top-1 (%) | 14.8021 |  | 77.4527 |  |  |
| pcpmae | scanobjectnn | obj_only | structured_keep80 | Top-1 (%) | 90.1893 |  | 2.0654 |  |  |
| pcpmae | scanobjectnn | obj_only | structured_keep50 | Top-1 (%) | 74.1824 |  | 18.0723 |  |  |
| pcpmae | scanobjectnn | obj_only | structured_keep20 | Top-1 (%) | 23.0637 |  | 69.1911 |  |  |
| pcpmae | scanobjectnn | obj_only | structured_keep10 | Top-1 (%) | 13.5972 |  | 78.6575 |  |  |
| pcpmae | scanobjectnn | obj_only | xyz_zero | Top-1 (%) | 9.2943 |  | 82.9604 |  |  |
| pcpmae | scanobjectnn | obj_only | clean | Top-1 (%) | 93.2874 |  | 0.0000 |  |  |
| pcpmae | scanobjectnn | obj_only | random_keep80 | Top-1 (%) | 92.5990 |  | 0.6885 |  |  |
| pcpmae | scanobjectnn | obj_only | random_keep50 | Top-1 (%) | 90.7057 |  | 2.5818 |  |  |
| pcpmae | scanobjectnn | obj_only | random_keep20 | Top-1 (%) | 60.5852 |  | 32.7022 |  |  |
| pcpmae | scanobjectnn | obj_only | random_keep10 | Top-1 (%) | 27.0224 |  | 66.2651 |  |  |
| pcpmae | scanobjectnn | obj_only | structured_keep80 | Top-1 (%) | 89.8451 |  | 3.4423 |  |  |
| pcpmae | scanobjectnn | obj_only | structured_keep50 | Top-1 (%) | 74.3546 |  | 18.9329 |  |  |
| pcpmae | scanobjectnn | obj_only | structured_keep20 | Top-1 (%) | 28.9157 |  | 64.3718 |  |  |
| pcpmae | scanobjectnn | obj_only | structured_keep10 | Top-1 (%) | 12.5645 |  | 80.7229 |  |  |
| pcpmae | scanobjectnn | obj_only | xyz_zero | Top-1 (%) | 9.2943 |  | 83.9931 |  |  |
| pcpmae | scanobjectnn | pb_t50_rs | clean | Top-1 (%) | 88.8619 |  | 0.0000 |  |  |
| pcpmae | scanobjectnn | pb_t50_rs | random_keep80 | Top-1 (%) | 88.5496 |  | 0.3123 |  |  |
| pcpmae | scanobjectnn | pb_t50_rs | random_keep50 | Top-1 (%) | 82.6162 |  | 6.2457 |  |  |
| pcpmae | scanobjectnn | pb_t50_rs | random_keep20 | Top-1 (%) | 29.9098 |  | 58.9521 |  |  |
| pcpmae | scanobjectnn | pb_t50_rs | random_keep10 | Top-1 (%) | 15.1978 |  | 73.6641 |  |  |
| pcpmae | scanobjectnn | pb_t50_rs | structured_keep80 | Top-1 (%) | 84.8369 |  | 4.0250 |  |  |
| pcpmae | scanobjectnn | pb_t50_rs | structured_keep50 | Top-1 (%) | 64.7120 |  | 24.1499 |  |  |
| pcpmae | scanobjectnn | pb_t50_rs | structured_keep20 | Top-1 (%) | 16.2734 |  | 72.5885 |  |  |
| pcpmae | scanobjectnn | pb_t50_rs | structured_keep10 | Top-1 (%) | 10.4788 |  | 78.3831 |  |  |
| pcpmae | scanobjectnn | pb_t50_rs | xyz_zero | Top-1 (%) | 9.3685 |  | 79.4934 |  |  |
| pcpmae | scanobjectnn | pb_t50_rs | clean | Top-1 (%) | 89.5212 |  | 0.0000 |  |  |
| pcpmae | scanobjectnn | pb_t50_rs | random_keep80 | Top-1 (%) | 88.7231 |  | 0.7981 |  |  |
| pcpmae | scanobjectnn | pb_t50_rs | random_keep50 | Top-1 (%) | 83.4490 |  | 6.0722 |  |  |
| pcpmae | scanobjectnn | pb_t50_rs | random_keep20 | Top-1 (%) | 26.9951 |  | 62.5260 |  |  |
| pcpmae | scanobjectnn | pb_t50_rs | random_keep10 | Top-1 (%) | 14.3997 |  | 75.1214 |  |  |
| pcpmae | scanobjectnn | pb_t50_rs | structured_keep80 | Top-1 (%) | 86.1555 |  | 3.3657 |  |  |
| pcpmae | scanobjectnn | pb_t50_rs | structured_keep50 | Top-1 (%) | 64.2262 |  | 25.2949 |  |  |
| pcpmae | scanobjectnn | pb_t50_rs | structured_keep20 | Top-1 (%) | 12.6648 |  | 76.8563 |  |  |
| pcpmae | scanobjectnn | pb_t50_rs | structured_keep10 | Top-1 (%) | 9.8196 |  | 79.7016 |  |  |
| pcpmae | scanobjectnn | pb_t50_rs | xyz_zero | Top-1 (%) | 9.3685 |  | 80.1527 |  |  |
| pointmae | scanobjectnn | obj_bg | clean | Top-1 (%) | 85.8864 |  | 0.0000 |  |  |
| pointmae | scanobjectnn | obj_bg | random_keep80 | Top-1 (%) | 85.7143 |  | 0.1721 |  |  |
| pointmae | scanobjectnn | obj_bg | random_keep50 | Top-1 (%) | 85.3701 |  | 0.5163 |  |  |
| pointmae | scanobjectnn | obj_bg | random_keep20 | Top-1 (%) | 71.6007 |  | 14.2857 |  |  |
| pointmae | scanobjectnn | obj_bg | random_keep10 | Top-1 (%) | 37.1773 |  | 48.7091 |  |  |
| pointmae | scanobjectnn | obj_bg | structured_keep80 | Top-1 (%) | 82.2719 |  | 3.6145 |  |  |
| pointmae | scanobjectnn | obj_bg | structured_keep50 | Top-1 (%) | 68.1583 |  | 17.7281 |  |  |
| pointmae | scanobjectnn | obj_bg | structured_keep20 | Top-1 (%) | 29.0878 |  | 56.7986 |  |  |
| pointmae | scanobjectnn | obj_bg | structured_keep10 | Top-1 (%) | 15.1463 |  | 70.7401 |  |  |
| pointmae | scanobjectnn | obj_bg | xyz_zero | Top-1 (%) | 7.2289 |  | 78.6575 |  |  |
| pointmae | scanobjectnn | obj_bg | clean | Top-1 (%) | 89.3287 |  | 0.0000 |  |  |
| pointmae | scanobjectnn | obj_bg | random_keep80 | Top-1 (%) | 89.3287 |  | 0.0000 |  |  |
| pointmae | scanobjectnn | obj_bg | random_keep50 | Top-1 (%) | 87.9518 |  | 1.3769 |  |  |
| pointmae | scanobjectnn | obj_bg | random_keep20 | Top-1 (%) | 73.8382 |  | 15.4905 |  |  |
| pointmae | scanobjectnn | obj_bg | random_keep10 | Top-1 (%) | 44.7504 |  | 44.5783 |  |  |
| pointmae | scanobjectnn | obj_bg | structured_keep80 | Top-1 (%) | 87.4355 |  | 1.8933 |  |  |
| pointmae | scanobjectnn | obj_bg | structured_keep50 | Top-1 (%) | 72.6334 |  | 16.6954 |  |  |
| pointmae | scanobjectnn | obj_bg | structured_keep20 | Top-1 (%) | 31.6695 |  | 57.6592 |  |  |
| pointmae | scanobjectnn | obj_bg | structured_keep10 | Top-1 (%) | 16.8675 |  | 72.4613 |  |  |
| pointmae | scanobjectnn | obj_bg | xyz_zero | Top-1 (%) | 7.2289 |  | 82.0998 |  |  |
| pointmae | scanobjectnn | obj_only | clean | Top-1 (%) | 86.4028 |  | 0.0000 |  |  |
| pointmae | scanobjectnn | obj_only | random_keep80 | Top-1 (%) | 86.5749 |  | -0.1721 |  |  |
| pointmae | scanobjectnn | obj_only | random_keep50 | Top-1 (%) | 85.5422 |  | 0.8606 |  |  |
| pointmae | scanobjectnn | obj_only | random_keep20 | Top-1 (%) | 70.2238 |  | 16.1790 |  |  |
| pointmae | scanobjectnn | obj_only | random_keep10 | Top-1 (%) | 39.7590 |  | 46.6437 |  |  |
| pointmae | scanobjectnn | obj_only | structured_keep80 | Top-1 (%) | 83.4768 |  | 2.9260 |  |  |
| pointmae | scanobjectnn | obj_only | structured_keep50 | Top-1 (%) | 67.4699 |  | 18.9329 |  |  |
| pointmae | scanobjectnn | obj_only | structured_keep20 | Top-1 (%) | 32.1859 |  | 54.2169 |  |  |
| pointmae | scanobjectnn | obj_only | structured_keep10 | Top-1 (%) | 15.3184 |  | 71.0843 |  |  |
| pointmae | scanobjectnn | obj_only | xyz_zero | Top-1 (%) | 7.2289 |  | 79.1738 |  |  |
| pointmae | scanobjectnn | obj_only | clean | Top-1 (%) | 86.7470 |  | 0.0000 |  |  |
| pointmae | scanobjectnn | obj_only | random_keep80 | Top-1 (%) | 86.5749 |  | 0.1721 |  |  |
| pointmae | scanobjectnn | obj_only | random_keep50 | Top-1 (%) | 86.2306 |  | 0.5164 |  |  |
| pointmae | scanobjectnn | obj_only | random_keep20 | Top-1 (%) | 75.2151 |  | 11.5318 |  |  |
| pointmae | scanobjectnn | obj_only | random_keep10 | Top-1 (%) | 44.2341 |  | 42.5129 |  |  |
| pointmae | scanobjectnn | obj_only | structured_keep80 | Top-1 (%) | 84.8537 |  | 1.8933 |  |  |
| pointmae | scanobjectnn | obj_only | structured_keep50 | Top-1 (%) | 71.4286 |  | 15.3184 |  |  |
| pointmae | scanobjectnn | obj_only | structured_keep20 | Top-1 (%) | 33.9071 |  | 52.8399 |  |  |
| pointmae | scanobjectnn | obj_only | structured_keep10 | Top-1 (%) | 15.1463 |  | 71.6007 |  |  |
| pointmae | scanobjectnn | obj_only | xyz_zero | Top-1 (%) | 7.2289 |  | 79.5181 |  |  |
| pointmae | scanobjectnn | pb_t50_rs | clean | Top-1 (%) | 83.1020 |  | 0.0000 |  |  |
| pointmae | scanobjectnn | pb_t50_rs | random_keep80 | Top-1 (%) | 82.7897 |  | 0.3123 |  |  |
| pointmae | scanobjectnn | pb_t50_rs | random_keep50 | Top-1 (%) | 81.6794 |  | 1.4226 |  |  |
| pointmae | scanobjectnn | pb_t50_rs | random_keep20 | Top-1 (%) | 64.4344 |  | 18.6676 |  |  |
| pointmae | scanobjectnn | pb_t50_rs | random_keep10 | Top-1 (%) | 21.2006 |  | 61.9015 |  |  |
| pointmae | scanobjectnn | pb_t50_rs | structured_keep80 | Top-1 (%) | 81.0201 |  | 2.0819 |  |  |
| pointmae | scanobjectnn | pb_t50_rs | structured_keep50 | Top-1 (%) | 67.8348 |  | 15.2672 |  |  |
| pointmae | scanobjectnn | pb_t50_rs | structured_keep20 | Top-1 (%) | 29.6322 |  | 53.4698 |  |  |
| pointmae | scanobjectnn | pb_t50_rs | structured_keep10 | Top-1 (%) | 13.4976 |  | 69.6044 |  |  |
| pointmae | scanobjectnn | pb_t50_rs | xyz_zero | Top-1 (%) | 7.0784 |  | 76.0236 |  |  |
| pointmae | scanobjectnn | pb_t50_rs | clean | Top-1 (%) | 84.8022 |  | 0.0000 |  |  |
| pointmae | scanobjectnn | pb_t50_rs | random_keep80 | Top-1 (%) | 84.5593 |  | 0.2429 |  |  |
| pointmae | scanobjectnn | pb_t50_rs | random_keep50 | Top-1 (%) | 83.6919 |  | 1.1103 |  |  |
| pointmae | scanobjectnn | pb_t50_rs | random_keep20 | Top-1 (%) | 68.5635 |  | 16.2387 |  |  |
| pointmae | scanobjectnn | pb_t50_rs | random_keep10 | Top-1 (%) | 27.4115 |  | 57.3907 |  |  |
| pointmae | scanobjectnn | pb_t50_rs | structured_keep80 | Top-1 (%) | 81.3671 |  | 3.4351 |  |  |
| pointmae | scanobjectnn | pb_t50_rs | structured_keep50 | Top-1 (%) | 67.9389 |  | 16.8633 |  |  |
| pointmae | scanobjectnn | pb_t50_rs | structured_keep20 | Top-1 (%) | 27.8279 |  | 56.9743 |  |  |
| pointmae | scanobjectnn | pb_t50_rs | structured_keep10 | Top-1 (%) | 12.4219 |  | 72.3803 |  |  |
| pointmae | scanobjectnn | pb_t50_rs | xyz_zero | Top-1 (%) | 7.0784 |  | 77.7238 |  |  |
| pcpmae | shapenetpart | test | clean | Instance mIoU (%) | 85.9889 | 85.9889 | 0.0000 | 2048.0000 | 0.0000 |
| pcpmae | shapenetpart | test | random_keep80 | Instance mIoU (%) | 86.0559 | 86.0980 | 0.0421 | 1638.0000 | 410.0000 |
| pcpmae | shapenetpart | test | random_keep50 | Instance mIoU (%) | 85.8114 | 86.2190 | 0.4076 | 1024.0000 | 1024.0000 |
| pcpmae | shapenetpart | test | random_keep20 | Instance mIoU (%) | 82.1574 | 86.7235 | 4.5661 | 410.0000 | 1638.0000 |
| pcpmae | shapenetpart | test | random_keep10 | Instance mIoU (%) | 68.2893 | 87.2877 | 18.9984 | 205.0000 | 1843.0000 |
| pcpmae | shapenetpart | test | structured_keep80 | Instance mIoU (%) | 84.9438 | 86.2881 | 1.3444 | 1638.0000 | 410.0000 |
| pcpmae | shapenetpart | test | structured_keep50 | Instance mIoU (%) | 79.0994 | 86.7817 | 7.6824 | 1024.0000 | 1024.0000 |
| pcpmae | shapenetpart | test | structured_keep20 | Instance mIoU (%) | 55.3745 | 89.4929 | 34.1184 | 410.0000 | 1638.0000 |
| pcpmae | shapenetpart | test | structured_keep10 | Instance mIoU (%) | 44.3529 | 91.3150 | 46.9621 | 205.0000 | 1843.0000 |
| pcpmae | shapenetpart | test | largest_part_removed | Instance mIoU (%) | 46.0976 | 65.1813 | 19.0837 | 775.2749 | 1272.7251 |
| pcpmae | shapenetpart | test | part_keep20_per_part | Instance mIoU (%) | 82.1125 | 86.3716 | 4.2591 | 409.6343 | 1638.3657 |
| pcpmae | shapenetpart | test | xyz_zero | Instance mIoU (%) | 32.7590 | 85.9889 | 53.2299 | 2048.0000 | 0.0000 |
| pcpmae | shapenetpart | test | clean | Instance mIoU (%) | 85.8322 | 85.8322 | 0.0000 | 2048.0000 | 0.0000 |
| pcpmae | shapenetpart | test | random_keep80 | Instance mIoU (%) | 85.9733 | 85.9174 | -0.0559 | 1638.0000 | 410.0000 |
| pcpmae | shapenetpart | test | random_keep50 | Instance mIoU (%) | 85.8245 | 86.1201 | 0.2956 | 1024.0000 | 1024.0000 |
| pcpmae | shapenetpart | test | random_keep20 | Instance mIoU (%) | 81.9350 | 86.7124 | 4.7774 | 410.0000 | 1638.0000 |
| pcpmae | shapenetpart | test | random_keep10 | Instance mIoU (%) | 67.8075 | 87.3595 | 19.5520 | 205.0000 | 1843.0000 |
| pcpmae | shapenetpart | test | structured_keep80 | Instance mIoU (%) | 84.6288 | 86.0452 | 1.4163 | 1638.0000 | 410.0000 |
| pcpmae | shapenetpart | test | structured_keep50 | Instance mIoU (%) | 77.8057 | 86.6333 | 8.8276 | 1024.0000 | 1024.0000 |
| pcpmae | shapenetpart | test | structured_keep20 | Instance mIoU (%) | 54.0379 | 89.2615 | 35.2236 | 410.0000 | 1638.0000 |
| pcpmae | shapenetpart | test | structured_keep10 | Instance mIoU (%) | 41.8619 | 91.2811 | 49.4192 | 205.0000 | 1843.0000 |
| pcpmae | shapenetpart | test | largest_part_removed | Instance mIoU (%) | 44.3507 | 64.7302 | 20.3795 | 776.2418 | 1271.7582 |
| pcpmae | shapenetpart | test | part_keep20_per_part | Instance mIoU (%) | 81.9100 | 86.2420 | 4.3320 | 409.6068 | 1638.3932 |
| pcpmae | shapenetpart | test | xyz_zero | Instance mIoU (%) | 33.2066 | 85.8322 | 52.6256 | 2048.0000 | 0.0000 |
| pointmae | shapenetpart | test | clean | Instance mIoU (%) | 86.0791 | 86.0791 | 0.0000 | 2048.0000 | 0.0000 |
| pointmae | shapenetpart | test | random_keep80 | Instance mIoU (%) | 86.1441 | 86.1594 | 0.0154 | 1638.0000 | 410.0000 |
| pointmae | shapenetpart | test | random_keep50 | Instance mIoU (%) | 85.9258 | 86.3607 | 0.4349 | 1024.0000 | 1024.0000 |
| pointmae | shapenetpart | test | random_keep20 | Instance mIoU (%) | 82.2458 | 86.9233 | 4.6775 | 410.0000 | 1638.0000 |
| pointmae | shapenetpart | test | random_keep10 | Instance mIoU (%) | 69.1295 | 87.2361 | 18.1066 | 205.0000 | 1843.0000 |
| pointmae | shapenetpart | test | structured_keep80 | Instance mIoU (%) | 85.2046 | 86.3984 | 1.1938 | 1638.0000 | 410.0000 |
| pointmae | shapenetpart | test | structured_keep50 | Instance mIoU (%) | 78.9483 | 86.7395 | 7.7912 | 1024.0000 | 1024.0000 |
| pointmae | shapenetpart | test | structured_keep20 | Instance mIoU (%) | 54.3569 | 89.5506 | 35.1938 | 410.0000 | 1638.0000 |
| pointmae | shapenetpart | test | structured_keep10 | Instance mIoU (%) | 41.1261 | 91.3828 | 50.2566 | 205.0000 | 1843.0000 |
| pointmae | shapenetpart | test | largest_part_removed | Instance mIoU (%) | 45.6727 | 64.9442 | 19.2716 | 774.9175 | 1273.0825 |
| pointmae | shapenetpart | test | part_keep20_per_part | Instance mIoU (%) | 81.9024 | 86.3709 | 4.4685 | 409.6211 | 1638.3789 |
| pointmae | shapenetpart | test | xyz_zero | Instance mIoU (%) | 31.7246 | 86.0791 | 54.3545 | 2048.0000 | 0.0000 |

## Q4 Candidate Sets

| model | task | split | selection_protocol | top1 | top2_hit | top5_hit | oracle2_score | oracle5_score |
|---|---|---|---|---|---|---|---|---|
| pcpmae | scanobjectnn | obj_bg | heldout_selection | 92.7711 | 97.9346 | 99.4837 | 97.9346 | 99.4837 |
| pcpmae | scanobjectnn | obj_bg | official_checkpoint | 94.4923 | 97.7625 | 99.6558 | 97.7625 | 99.6558 |
| pcpmae | scanobjectnn | obj_only | heldout_selection | 92.2547 | 97.2461 | 99.1394 | 97.2461 | 99.1394 |
| pcpmae | scanobjectnn | obj_only | official_checkpoint | 93.2874 | 96.7298 | 99.3115 | 96.7298 | 99.3115 |
| pcpmae | scanobjectnn | pb_t50_rs | heldout_selection | 88.8619 | 95.7321 | 98.8550 | 95.7321 | 98.8550 |
| pcpmae | scanobjectnn | pb_t50_rs | official_checkpoint | 89.5212 | 96.2179 | 99.0632 | 96.2179 | 99.0632 |
| pointmae | scanobjectnn | obj_bg | heldout_selection | 85.8864 | 94.4923 | 98.9673 | 94.4923 | 98.9673 |
| pointmae | scanobjectnn | obj_bg | official_checkpoint | 89.3287 | 95.5250 | 99.1394 | 95.5250 | 99.1394 |
| pointmae | scanobjectnn | obj_only | heldout_selection | 86.4028 | 94.3201 | 98.2788 | 94.3201 | 98.2788 |
| pointmae | scanobjectnn | obj_only | official_checkpoint | 86.7470 | 95.0086 | 98.1067 | 95.0086 | 98.1067 |
| pointmae | scanobjectnn | pb_t50_rs | heldout_selection | 83.1020 | 92.0541 | 97.5711 | 92.0541 | 97.5711 |
| pointmae | scanobjectnn | pb_t50_rs | official_checkpoint | 84.8022 | 92.6787 | 98.0569 | 92.6787 | 98.0569 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain | 94.7296 | 99.2357 | 100.0000 | 95.6547 | 100.0000 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_ckpt300 | 94.7102 | 99.2322 | 100.0000 | 95.5729 | 100.0000 |
| pointmae | shapenetpart | test | official_checkpoint | 94.6506 | 99.2573 | 100.0000 | 95.8730 | 100.0000 |

## Q4 Selection Protocol

| model | task | split | selection_protocol | top1 | top2_hit | top5_hit |
|---|---|---|---|---|---|---|
| pcpmae | scanobjectnn | obj_bg | heldout_selection | 92.7711 | 97.9346 | 99.4837 |
| pcpmae | scanobjectnn | obj_only | heldout_selection | 92.2547 | 97.2461 | 99.1394 |
| pcpmae | scanobjectnn | pb_t50_rs | heldout_selection | 88.8619 | 95.7321 | 98.8550 |
| pointmae | scanobjectnn | obj_bg | heldout_selection | 85.8864 | 94.4923 | 98.9673 |
| pointmae | scanobjectnn | obj_only | heldout_selection | 86.4028 | 94.3201 | 98.2788 |
| pointmae | scanobjectnn | pb_t50_rs | heldout_selection | 83.1020 | 92.0541 | 97.5711 |

## Grouping Diagnostics

- raw dir: `/mnt/urashima/users/minesawa/home-offload/ssl/3D-NEPA/results/object_ssl_pointmae_pcpmae/raw_grouping`

| model | task | split | selection_protocol | grouping_mode | condition | metric_name | score | clean_subset_score | damage_pp | mean_retained_unique_points | mean_repeated_forward_points |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pcpmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | fps_knn | clean | Top-1 (%) | 95.1807 |  | 0.0000 |  |  |
| pcpmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | fps_knn | random_keep20 | Top-1 (%) | 21.1704 |  | 74.0103 |  |  |
| pcpmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | fps_knn | structured_keep20 | Top-1 (%) | 17.7281 |  | 77.4527 |  |  |
| pcpmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | fps_knn | xyz_zero | Top-1 (%) | 9.2943 |  | 85.8864 |  |  |
| pcpmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_center_knn | clean | Top-1 (%) | 93.4596 |  | 0.0000 |  |  |
| pcpmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_center_knn | random_keep20 | Top-1 (%) | 22.3752 |  | 71.0843 |  |  |
| pcpmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_center_knn | structured_keep20 | Top-1 (%) | 17.7281 |  | 75.7315 |  |  |
| pcpmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_center_knn | xyz_zero | Top-1 (%) | 9.2943 |  | 84.1652 |  |  |
| pcpmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_group | clean | Top-1 (%) | 11.3597 |  | 0.0000 |  |  |
| pcpmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_group | random_keep20 | Top-1 (%) | 11.1876 |  | 0.1721 |  |  |
| pcpmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_group | structured_keep20 | Top-1 (%) | 9.8107 |  | 1.5491 |  |  |
| pcpmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_group | xyz_zero | Top-1 (%) | 9.2943 |  | 2.0654 |  |  |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | fps_knn | clean | Instance mIoU (%) | 85.9109 | 85.9109 | 0.0000 | 2048.0000 | 0.0000 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | fps_knn | random_keep20 | Instance mIoU (%) | 82.2624 | 86.7144 | 4.4520 | 410.0000 | 1638.0000 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | fps_knn | structured_keep20 | Instance mIoU (%) | 55.0888 | 89.3869 | 34.2982 | 410.0000 | 1638.0000 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | fps_knn | largest_part_removed | Instance mIoU (%) | 45.9525 | 65.0835 | 19.1311 | 776.1312 | 1271.8688 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | fps_knn | xyz_zero | Instance mIoU (%) | 32.7501 | 85.9109 | 53.1608 | 2048.0000 | 0.0000 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | random_center_knn | clean | Instance mIoU (%) | 84.0365 | 84.0365 | 0.0000 | 2048.0000 | 0.0000 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | random_center_knn | random_keep20 | Instance mIoU (%) | 78.9081 | 84.9286 | 6.0205 | 410.0000 | 1638.0000 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | random_center_knn | structured_keep20 | Instance mIoU (%) | 53.2739 | 87.9918 | 34.7179 | 410.0000 | 1638.0000 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | random_center_knn | largest_part_removed | Instance mIoU (%) | 44.1462 | 63.7838 | 19.6376 | 775.0557 | 1272.9443 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | random_center_knn | xyz_zero | Instance mIoU (%) | 32.6972 | 84.0365 | 51.3393 | 2048.0000 | 0.0000 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | random_group | clean | Instance mIoU (%) | 55.3504 | 55.3504 | 0.0000 | 2048.0000 | 0.0000 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | random_group | random_keep20 | Instance mIoU (%) | 55.8869 | 56.3364 | 0.4495 | 410.0000 | 1638.0000 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | random_group | structured_keep20 | Instance mIoU (%) | 55.5982 | 63.7916 | 8.1933 | 410.0000 | 1638.0000 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | random_group | largest_part_removed | Instance mIoU (%) | 38.6009 | 42.5978 | 3.9968 | 775.0571 | 1272.9429 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | random_group | xyz_zero | Instance mIoU (%) | 32.7030 | 55.3504 | 22.6474 | 2048.0000 | 0.0000 |
| pointmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | fps_knn | clean | Top-1 (%) | 89.5009 |  | 0.0000 |  |  |
| pointmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | fps_knn | random_keep20 | Top-1 (%) | 72.2892 |  | 17.2117 |  |  |
| pointmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | fps_knn | structured_keep20 | Top-1 (%) | 30.8089 |  | 58.6919 |  |  |
| pointmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | fps_knn | xyz_zero | Top-1 (%) | 7.2289 |  | 82.2719 |  |  |
| pointmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_center_knn | clean | Top-1 (%) | 86.5749 |  | 0.0000 |  |  |
| pointmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_center_knn | random_keep20 | Top-1 (%) | 71.6007 |  | 14.9742 |  |  |
| pointmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_center_knn | structured_keep20 | Top-1 (%) | 28.5714 |  | 58.0034 |  |  |
| pointmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_center_knn | xyz_zero | Top-1 (%) | 7.2289 |  | 79.3460 |  |  |
| pointmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_group | clean | Top-1 (%) | 16.6954 |  | 0.0000 |  |  |
| pointmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_group | random_keep20 | Top-1 (%) | 16.1790 |  | 0.5164 |  |  |
| pointmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_group | structured_keep20 | Top-1 (%) | 13.4251 |  | 3.2702 |  |  |
| pointmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_group | xyz_zero | Top-1 (%) | 7.2289 |  | 9.4664 |  |  |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | fps_knn | clean | Instance mIoU (%) | 86.0380 | 86.0380 | 0.0000 | 2048.0000 | 0.0000 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | fps_knn | random_keep20 | Instance mIoU (%) | 82.0172 | 86.8155 | 4.7982 | 410.0000 | 1638.0000 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | fps_knn | structured_keep20 | Instance mIoU (%) | 54.1739 | 89.6003 | 35.4264 | 410.0000 | 1638.0000 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | fps_knn | largest_part_removed | Instance mIoU (%) | 45.3668 | 64.9801 | 19.6133 | 775.4680 | 1272.5320 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | fps_knn | xyz_zero | Instance mIoU (%) | 31.6966 | 86.0380 | 54.3413 | 2048.0000 | 0.0000 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | random_center_knn | clean | Instance mIoU (%) | 84.3750 | 84.3750 | 0.0000 | 2048.0000 | 0.0000 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | random_center_knn | random_keep20 | Instance mIoU (%) | 79.0622 | 85.2315 | 6.1693 | 410.0000 | 1638.0000 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | random_center_knn | structured_keep20 | Instance mIoU (%) | 53.1102 | 88.0284 | 34.9182 | 410.0000 | 1638.0000 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | random_center_knn | largest_part_removed | Instance mIoU (%) | 44.1234 | 63.8613 | 19.7379 | 775.0505 | 1272.9495 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | random_center_knn | xyz_zero | Instance mIoU (%) | 31.6634 | 84.3750 | 52.7116 | 2048.0000 | 0.0000 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | random_group | clean | Instance mIoU (%) | 56.4945 | 56.4945 | 0.0000 | 2048.0000 | 0.0000 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | random_group | random_keep20 | Instance mIoU (%) | 57.0634 | 57.3153 | 0.2519 | 410.0000 | 1638.0000 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | random_group | structured_keep20 | Instance mIoU (%) | 62.6296 | 64.7241 | 2.0945 | 410.0000 | 1638.0000 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | random_group | largest_part_removed | Instance mIoU (%) | 46.9831 | 52.2042 | 5.2211 | 774.9332 | 1273.0668 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | random_group | xyz_zero | Instance mIoU (%) | 31.7023 | 56.4945 | 24.7922 | 2048.0000 | 0.0000 |

## Scene-Side Support Audit

- Static audit target: `/home/minesawa/ssl/concerto-shortcut-mvp/tools/concerto_projection_shortcut/eval_masking_battery.py`.
- The scene-side masking battery filters point-level tensors with explicit masks and records separate `retained` and `full_nn` score spaces.
- Existing scene result files such as `/home/minesawa/ssl/concerto-shortcut-mvp/tools/concerto_projection_shortcut/results_support_severity_concerto_decoder.csv` and `/home/minesawa/ssl/concerto-shortcut-mvp/tools/concerto_projection_shortcut/results_utonia_scannet_support_stress_featurezero_audit/utonia_scannet_support_stress.md` should therefore be cited with their score-space label. `retained` is retained-support scoring; `full_nn` is full-scene nearest-retained propagation.
- This audit is separate from the object-side Point-MAE/PCP-MAE rerun and does not add new scene numbers.

## Data Paths

- ScanObjectNN: `/mnt/urashima/users/minesawa/home-offload/ssl/3D-NEPA/data/ScanObjectNN/h5_files`
- ShapeNetPart: `/mnt/urashima/users/minesawa/home-offload/ssl/3D-NEPA/data/shapenetcore_partanno_segmentation_benchmark_v0_normal`

## Scripts And Commands

- Main orchestrator: `scripts/object_ssl/pointmae_pcpmae_chain.py`
- ScanObjectNN adapter: `scripts/object_ssl/eval_scanobjectnn_mae.py`
- ShapeNetPart adapter: `scripts/object_ssl/eval_shapenetpart_mae.py`
- Held-out split maker: `scripts/object_ssl/make_scanobjectnn_heldout_split.py`
- Logs with exact commands: `/mnt/urashima/users/minesawa/home-offload/ssl/3D-NEPA/logs/object_ssl_pointmae_pcpmae`

## Blockers

- `selection=reporting` checkpoint-trajectory diagnostic was not emitted. The official fine-tuning code retained only `ckpt-best.pth` and `ckpt-last.pth`, not a full epoch checkpoint trajectory; selecting on the target test split from only these retained checkpoints would be an under-specified approximation rather than the planned diagnostic.

