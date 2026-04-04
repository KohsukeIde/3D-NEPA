# Itachi Geo-Teacher Active Results

Last updated: 2026-04-03 (JST)

This page records `itachi`-local geo-teacher runs as soon as they produce a
usable artifact.

It is intentionally separate from the paper-facing benchmark pages:

- this file may include pilot or provisional runs,
- each entry must state whether it is provisional or canonical,
- only canonically aligned runs should later be copied into headline tables.

Current compare note:

- `300 epoch` improves Route-B same-context geometry readouts,
- but does not materially improve Route-A utility over the `100 epoch` pilot,
- so the current interim compare budget is frozen at `100 pretrain epochs`
  while the next arm comparisons are assembled.

Current active compare arm:

- `geo_teacher_distance_only_100ep_itachi_main`
  - status:
    - pretrain running
    - `100` epochs
    - `4 GPU` DDP
    - `udf_distance` only
  - downstream chain:
    - queued and waiting for pretrain completion
    - planned readouts:
      - `ScanObjectNN`
      - `ShapeNetPart`
      - single-task Route-B (`udf_distance`)
      - `completion`
      - `curvature`

## Current checkpoint

- pretrain checkpoint:
  - `runs/cqa_itachi/geo_teacher_distnorm_unsigned_100ep_itachi_main/ckpt_final.pt`
- pretrain regime:
  - `100` epochs
  - `udf_distance + mesh_normal_unsigned`
  - `packed + multihead + per_task`
- status:
  - pilot
  - not the final `300`-epoch external-comparison recipe

## Current full-budget checkpoint

- pretrain checkpoint:
  - `runs/cqa_itachi/geo_teacher_distnorm_unsigned_300ep_itachi_main/ckpt_final.pt`
- pretrain regime:
  - `300` epochs
  - `udf_distance + mesh_normal_unsigned`
  - `packed + multihead + per_task`
  - `4 GPU` DDP
  - `global_step = 105300`
- status:
  - completed
  - this is the current full-budget local result for the `distance + normal_unsigned` arm

## Route A: ScanObjectNN utility

### Current full-budget (`300 epoch` pretrain) read

- `obj_bg`
  - completed
  - canonical read:
    - `test_acc = 0.8451`
  - delta vs `100 epoch` pilot:
    - `+0.0000`
  - current interpretation:
    - effectively unchanged from the `100 epoch` pilot
- `obj_only`
  - completed
  - canonical read:
    - `test_acc = 0.8485`
  - delta vs `100 epoch` pilot:
    - `+0.0017`
  - current interpretation:
    - slight gain over the `100 epoch` pilot
    - still below `C035 obj_only = 0.8520`
- `pb_t50_rs`
  - completed
  - canonical read:
    - `test_acc = 0.8046`
  - delta vs `100 epoch` pilot:
    - `+0.0014`
  - current interpretation:
    - slight gain over the `100 epoch` pilot
    - still below corrected `Point-MAE pb_t50_rs = 0.8459`

### Current canonical rerun status

- `obj_bg`
  - completed
  - route:
    - direct `scan_h5`
    - `4 GPU` DDP
    - eval `fps` aligned in H5 route via on-the-fly deterministic FPS
  - canonical read:
    - `test_acc = 0.8451`
    - `val_best_acc = 0.8434`
  - current interpretation:
    - effectively tied with `C035`
    - above `C034`
    - still below corrected `Point-MAE obj_bg = 0.9019`
- `obj_only`
  - completed
  - route:
    - direct `scan_h5`
    - `4 GPU` DDP
    - eval `fps` aligned in H5 route via on-the-fly deterministic FPS
  - canonical read:
    - `test_acc = 0.8468`
    - `val_best_acc = 0.8606`
  - current interpretation:
    - below `C035 obj_only = 0.8520`
    - below corrected `Point-MAE obj_only = 0.8795`
- `pb_t50_rs`
  - completed
  - route:
    - direct `scan_h5`
    - `4 GPU` DDP
    - eval `fps` aligned in H5 route via on-the-fly deterministic FPS
  - canonical read:
    - `test_acc = 0.8033`
    - `val_best_acc = 0.8029`
  - current interpretation:
    - above `C034 pb_t50_rs = 0.7679`
    - above `C035 pb_t50_rs = 0.7710`
    - still below corrected `Point-MAE pb_t50_rs = 0.8459`

## Route A: ShapeNetPart utility

- `ShapeNetPart` (`300 epoch` pretrain)
  - completed
  - canonical read:
    - `TEST acc = 0.9419`
    - `TEST class_avg_iou = 0.8261`
    - `TEST instance_avg_iou = 0.8499`
  - delta vs `100 epoch` pilot:
    - `class_avg_iou: -0.0052`
    - `instance_avg_iou: -0.0017`
  - current interpretation:
    - slightly worse than the `100 epoch` pilot
    - still below `Point-MAE 86.1` and `PCP-MAE 84.9 Cls.mIoU`

- `ShapeNetPart`
  - completed
  - route:
    - direct `shapenetcore_partanno_segmentation_benchmark_v0_normal`
    - `4 GPU` DDP
    - `300` epoch fine-tune
  - canonical read:
    - `TEST acc = 0.9434`
    - `TEST class_avg_iou = 0.8313`
    - `TEST instance_avg_iou = 0.8516`
    - `best test instance_avg_iou = 0.8530`
  - metric alignment note:
    - `Point-MAE` / `PCP-MAE` segmentation code reports both class-average and instance-average shape IoU
    - the README headline comparison is safest when read as `class_avg_iou` (`Cls.mIoU`)
    - therefore the primary comparison metric here should be `class_avg_iou = 0.8313`
    - `instance_avg_iou = 0.8516` should be kept as an auxiliary metric, not the headline parity number
  - current interpretation:
    - utility readout is strong enough to keep the geo-teacher line alive for Route A
    - current aligned `Cls.mIoU` is below `Point-MAE 86.1` and below `PCP-MAE 84.9`
    - but this is still a `100 epoch` pretrain pilot, not the final `300 epoch` pretrain comparison recipe

### Provisional historical local result

- run:
  - `geo_teacher_distnorm_unsigned_100ep_itachi_main__ft_obj_bg_300ep`
- note:
  - an earlier local run completed before H5-eval FPS parity was fixed
  - it used `pt_sample_mode_eval=fps` in config but actually fell back to
    random subset on direct H5 input
  - keep as diagnostic only; do not treat as canonical
- provisional read:
  - `obj_bg test_acc = 0.8520`

### Historical comparison anchors

- budget note:
  - `C034/C035` are best read as `10k-step` ABCI CQA runs, not full-budget `100/300`-epoch pretrains
  - current `itachi` pilot is `100` epochs on `45047` shapes with `steps_per_epoch=175`, so `17500` optimizer steps
  - this means the current pilot is already about `1.75x` the `C034/C035` optimization budget
- `C034` strict `DISTANCE + NORMAL_UNSIGNED` utility:
  - `obj_bg = 0.8399`
  - `obj_only = 0.8503`
  - `pb_t50_rs = 0.7679`
- `C035` raw-target 4-task utility:
  - `obj_bg = 0.8451`
  - `obj_only = 0.8520`
  - `pb_t50_rs = 0.7710`

## Route B: Analysis / capability

### Current full-budget (`300 epoch` pretrain) read

- multitype same/offdiag:
  - same-context correct:
    - `udf_distance token_acc = 0.5650`
    - `mesh_normal_unsigned token_acc = 0.7500`
  - offdiag correct:
    - `udf_distance token_acc = 0.1460`
    - `mesh_normal_unsigned token_acc = 0.4350`
  - delta vs `100 epoch` pilot:
    - strong same-context gain
    - modest offdiag gain

- completion:
  - same-context:
    - `MAE = 0.00635`
    - `IoU@0.05 = 0.9651`
  - offdiag:
    - `MAE = 0.1178`
    - `IoU@0.05 = 0.4785`
  - delta vs `100 epoch` pilot:
    - same-context improved clearly
    - offdiag improved slightly

- curvature probe:
  - same-context:
    - `MAE = 0.1202`
    - `pearson_r = 0.6795`
  - offdiag:
    - `MAE = 0.2041`
    - `pearson_r = 0.1506`
  - delta vs `100 epoch` pilot:
    - same-context improved
    - offdiag worsened

### Current automatic chain status

- the repaired automatic Route-B stage now completes through:
  - multitype same/offdiag suite
  - `udf_distance` completion same/offdiag
- current remaining tail:
  - a fresh `curvature` probe rerun is still in progress
- current state:
  - multitype/completion artifacts are present and usable
  - the post-train chain itself is still marked `running` because the final curvature rerun has not exited yet

### Multitype same/offdiag

- artifact:
  - `results/cqa_multitype_itachi/geo_teacher_distnorm_unsigned_100ep_itachi_main__multitype_same_offdiag.json`
- status:
  - usable
  - produced after the repaired post-train chain

Current pilot, correct-control rows:

- same-context:
  - `udf_distance token_acc = 0.3377`
  - `mesh_normal_unsigned token_acc = 0.6692`
- offdiag (`surf -> pc_bank`):
  - `udf_distance token_acc = 0.1278`
  - `mesh_normal_unsigned token_acc = 0.4014`

Control sanity checks:

- same-context `no_context`:
  - `udf_distance = 0.0171`
  - `mesh_normal_unsigned = 0.0522`
- offdiag `no_context`:
  - `udf_distance = 0.0169`
  - `mesh_normal_unsigned = 0.0512`

Current read:

- same-context typed answering is clearly real
- offdiag transfer survives, but with a large drop from same-context
- controls remain appropriately weak, so the signal is not a trivial leakage artifact

### Completion

- artifacts:
  - `results/cqa_completion_itachi/geo_teacher_distnorm_unsigned_100ep_itachi_main__completion_same.json`
  - `results/cqa_completion_itachi/geo_teacher_distnorm_unsigned_100ep_itachi_main__completion_offdiag.json`
- status:
  - usable
  - `udf_distance` only

Current pilot:

- same-context completion (`surf`)
  - `MAE mean = 0.009291`
  - `RMSE mean = 0.018080`
  - `IoU@0.01 = 0.5299`
  - `IoU@0.02 = 0.7365`
  - `IoU@0.05 = 0.8941`
- offdiag completion (`pc_bank`)
  - `MAE mean = 0.118532`
  - `RMSE mean = 0.201526`
  - `IoU@0.01 = 0.3312`
  - `IoU@0.02 = 0.4322`
  - `IoU@0.05 = 0.4554`

Current read:

- same-context completion is strong
- offdiag completion is substantially weaker but still clearly above collapse

### Curvature probe

- artifact:
  - `results/cqa_probe_itachi/geo_teacher_distnorm_unsigned_100ep_itachi_main__curvature_probe.json`
- status:
  - previous probe artifact is usable
  - a fresh rerun is currently in progress as the final tail of the repaired post-train chain

Current available read:

- same-context correct:
  - `MAE = 0.1262`
  - `RMSE = 0.1741`
  - `pearson_r = 0.6216`
- offdiag correct:
  - `MAE = 0.1902`
  - `RMSE = 0.2331`
  - `pearson_r = 0.1957`

Historical anchors:

- `C034`:
  - same-context correct:
    - `MAE = 0.1482`
    - `RMSE = 0.1988`
    - `pearson_r = 0.4821`
  - offdiag correct:
    - `MAE = 0.2000`
- `C035`:
  - same-context correct:
    - `MAE = 0.1517`
    - `RMSE = 0.1999`
    - `pearson_r = 0.4514`
  - offdiag correct:
    - `MAE = 0.1888`

Current read:

- same-context curvature is stronger than both `C034` and `C035`
- offdiag curvature is still mixed:
  - slightly better than `C034` on MAE
  - close to `C035`
  - still not a clean offdiag breakthrough

## Active queue

- `ShapeNetPart` local data:
  - ready at
    `data/shapenetcore_partanno_segmentation_benchmark_v0_normal`
- `300 epoch` full chain:
  - queued in tmux session
    `geo_teacher_distnorm_unsigned_300ep_itachi_main_chain`
  - current state:
    - `300 epoch` pretrain completed
    - corrected downstream rerun completed
    - the first downstream attempt was invalid:
      - it accidentally reused the `100 epoch` checkpoint
      - symptom:
        - `obj_bg` and `obj_only` `test_metrics.json` were exactly identical to the `100 epoch` run
      - cause:
        - stale `CKPT_PATH` leaked into the downstream launcher
    - invalid `300ep` downstream artifacts were archived on `2026-04-04`
    - corrected downstream used:
      - `ckpt=/home/minesawa/ssl/3D-NEPA/runs/cqa_itachi/geo_teacher_distnorm_unsigned_300ep_itachi_main/ckpt_final.pt`
  - planned flow:
    - completed

## Update rule

When a local run finishes and produces a stable artifact:

1. append the raw metric here with a `provisional` or `canonical` tag
2. compare it against the nearest historical anchor (`C034`, `C035`, etc.)
3. copy only canonical results into the benchmark-facing tables
