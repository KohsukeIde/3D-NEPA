# Point-MAE / PCP-MAE Eval-Time Grouping Diagnostics

Checkpoint and readout are fixed. These rows perturb grouping only at inference time; no retraining is used.

- git commit: `7b7b788b372fc32c8a1a3be6a88682b25bf1095f`
- raw dir: `/mnt/urashima/users/minesawa/home-offload/ssl/3D-NEPA/results/object_ssl_pointmae_pcpmae/raw_grouping`

| model | task | split | selection_protocol | grouping_mode | condition | metric_name | score | damage_pp |
|---|---|---|---|---|---|---|---|---|
| pcpmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | fps_knn | clean | Top-1 (%) | 95.1807 | 0.0000 |
| pcpmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | fps_knn | random_keep20 | Top-1 (%) | 21.1704 | 74.0103 |
| pcpmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | fps_knn | structured_keep20 | Top-1 (%) | 17.7281 | 77.4527 |
| pcpmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | fps_knn | xyz_zero | Top-1 (%) | 9.2943 | 85.8864 |
| pcpmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_center_knn | clean | Top-1 (%) | 93.4596 | 0.0000 |
| pcpmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_center_knn | random_keep20 | Top-1 (%) | 22.3752 | 71.0843 |
| pcpmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_center_knn | structured_keep20 | Top-1 (%) | 17.7281 | 75.7315 |
| pcpmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_center_knn | xyz_zero | Top-1 (%) | 9.2943 | 84.1652 |
| pcpmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_group | clean | Top-1 (%) | 11.3597 | 0.0000 |
| pcpmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_group | random_keep20 | Top-1 (%) | 11.1876 | 0.1721 |
| pcpmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_group | structured_keep20 | Top-1 (%) | 9.8107 | 1.5491 |
| pcpmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_group | xyz_zero | Top-1 (%) | 9.2943 | 2.0654 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | fps_knn | clean | Instance mIoU (%) | 85.9175 | 0.0000 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | fps_knn | random_keep20 | Instance mIoU (%) | 82.3315 | 3.5860 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | fps_knn | structured_keep20 | Instance mIoU (%) | 55.2348 | 30.6827 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | fps_knn | largest_part_removed | Instance mIoU (%) | 46.2114 | 39.7061 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | fps_knn | xyz_zero | Instance mIoU (%) | 32.7488 | 53.1687 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | random_center_knn | clean | Instance mIoU (%) | 84.0348 | 0.0000 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | random_center_knn | random_keep20 | Instance mIoU (%) | 78.9476 | 5.0872 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | random_center_knn | structured_keep20 | Instance mIoU (%) | 53.3994 | 30.6354 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | random_center_knn | largest_part_removed | Instance mIoU (%) | 44.5848 | 39.4500 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | random_center_knn | xyz_zero | Instance mIoU (%) | 32.7157 | 51.3191 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | random_group | clean | Instance mIoU (%) | 55.5382 | 0.0000 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | random_group | random_keep20 | Instance mIoU (%) | 55.8912 | -0.3530 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | random_group | structured_keep20 | Instance mIoU (%) | 56.0278 | -0.4895 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | random_group | largest_part_removed | Instance mIoU (%) | 38.7735 | 16.7647 |
| pcpmae | shapenetpart | test | finetuned_from_public_pretrain_eval_time_grouping | random_group | xyz_zero | Instance mIoU (%) | 32.7050 | 22.8333 |
| pointmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | fps_knn | clean | Top-1 (%) | 89.5009 | 0.0000 |
| pointmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | fps_knn | random_keep20 | Top-1 (%) | 72.2892 | 17.2117 |
| pointmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | fps_knn | structured_keep20 | Top-1 (%) | 30.8089 | 58.6919 |
| pointmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | fps_knn | xyz_zero | Top-1 (%) | 7.2289 | 82.2719 |
| pointmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_center_knn | clean | Top-1 (%) | 86.5749 | 0.0000 |
| pointmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_center_knn | random_keep20 | Top-1 (%) | 71.6007 | 14.9742 |
| pointmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_center_knn | structured_keep20 | Top-1 (%) | 28.5714 | 58.0034 |
| pointmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_center_knn | xyz_zero | Top-1 (%) | 7.2289 | 79.3460 |
| pointmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_group | clean | Top-1 (%) | 16.6954 | 0.0000 |
| pointmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_group | random_keep20 | Top-1 (%) | 16.1790 | 0.5164 |
| pointmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_group | structured_keep20 | Top-1 (%) | 13.4251 | 3.2702 |
| pointmae | scanobjectnn | obj_bg | official_checkpoint_eval_time_grouping | random_group | xyz_zero | Top-1 (%) | 7.2289 | 9.4664 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | fps_knn | clean | Instance mIoU (%) | 86.0743 | 0.0000 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | fps_knn | random_keep20 | Instance mIoU (%) | 82.3334 | 3.7409 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | fps_knn | structured_keep20 | Instance mIoU (%) | 53.7922 | 32.2821 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | fps_knn | largest_part_removed | Instance mIoU (%) | 45.6462 | 40.4281 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | fps_knn | xyz_zero | Instance mIoU (%) | 31.6725 | 54.4018 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | random_center_knn | clean | Instance mIoU (%) | 84.2147 | 0.0000 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | random_center_knn | random_keep20 | Instance mIoU (%) | 78.9091 | 5.3055 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | random_center_knn | structured_keep20 | Instance mIoU (%) | 53.4100 | 30.8047 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | random_center_knn | largest_part_removed | Instance mIoU (%) | 43.6942 | 40.5205 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | random_center_knn | xyz_zero | Instance mIoU (%) | 31.6893 | 52.5254 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | random_group | clean | Instance mIoU (%) | 56.5401 | 0.0000 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | random_group | random_keep20 | Instance mIoU (%) | 57.1889 | -0.6488 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | random_group | structured_keep20 | Instance mIoU (%) | 61.6441 | -5.1040 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | random_group | largest_part_removed | Instance mIoU (%) | 46.6623 | 9.8778 |
| pointmae | shapenetpart | test | official_checkpoint_eval_time_grouping | random_group | xyz_zero | Instance mIoU (%) | 31.7176 | 24.8225 |
