# Point-MAE / PCP-MAE Eval-Time Grouping Diagnostics

Checkpoint and readout are fixed. These rows perturb grouping only at inference time; no retraining is used.

- git commit: `fd9da67bb59f04e034c8f98fa1e9f42244801318`
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
