# PatchNEPA Execution Backlog

Last updated: 2026-03-11

## 1. Purpose

This file is the canonical operational backlog for the PatchNEPA line during
the local-only period.

Use this page when asking:

- what is still unresolved,
- what should run next on the local GPU machine,
- what the success/fail rule is for each queued experiment,
- where the canonical output must be recorded after completion.

This file is the source of truth for execution order and gating.

## 2. Operational Contract

- primary execution surface: local Linux GPU machine with the same repo checkout
- PBS/QF launchers are not the operational default during this phase
- canonical scientific conclusions still live in:
  - `storyline_query_to_patch_v2_active.md`
  - `hypothesis_matrix_active.md`
  - `restart_plan_patchnepa_data_v2_20260303.md`
  - `runlog_patch_nepa_202602.md`
- queue/runtime state lives in:
  - `scripts/local/patchnepa_local_queue.tsv`
  - `scripts/local/patchnepa_local_queue_runner.sh`
  - `scripts/local/patchnepa_local_status.sh`
- runtime completion is **not** canonical completion
- a row is only canonized after the result is written back into
  `restart_plan_patchnepa_data_v2_20260303.md` or
  `runlog_patch_nepa_202602.md`, and then summarized in
  `storyline_query_to_patch_v2_active.md` if it changes the current read

## 3. Status and Cost

Status values:

- `queued`
- `running`
- `done-unbackfilled`
- `canonized`
- `blocked`
- `deferred`
- `manual`

Local budget classes:

- `A`: no-GPU analysis / log aggregation / W&B summary extraction / plots
- `B`: 1-GPU short smoke or mini-eval, target `<= 2h`
- `C`: 1-GPU overnight, target `<= 12h`
- `D`: multi-day or multi-GPU local run

Policy:

- default queue target is `A-C`
- `D` requires a successful lower-cost gate first
- full300-class local reruns are banned unless they directly decide the paper

## 4. Canonicalization Workflow

- source priority:
  - result JSON
  - W&B summary
  - structured log root
- do not use top-level stdout/stderr as the only retained result
- queue state progression:
  - `queued -> running -> done-unbackfilled -> canonized`
- if a row fails:
  - keep the failed runtime record in `logs/local_queue/.../state.tsv`
  - do not promote the result into docs until the failure mode itself is worth
    recording

## 5. Backlog Table

| ID | Priority | Question | Class | Status | Local Cost | Gating | Canonical Input | Canonical Output | Decision Rule | Canonize Target | Notes |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| `L000A` | `5` | Can a visibility-first world-package rebuild (`mesh_qry_vis_sig`, `mesh_qry_ao`, `pc_ctx_bank_xyz`, `udf_surf_probe_*`) complete cleanly and keep short-run reconstruction healthy? | `D` | `queued` | `multi-stage local rebuild + 2 GPU short screen` | none | `data/ShapeNetCore.v2` + `scripts/local/patchnepa_visocc_branch.sh` + `nepa3d/configs/shapenet_unpaired_mix_v2_tokens_drop1_pc33_mesh33_udf33_visocc{,_base}.yaml` | `logs/local_patchnepa_visocc/patchnepa_visocc_l000ab_20260311/decision.json` | `visocc` stays alive only if short pretrain keeps `recon_lift_q >= 0` and `recon_lift_a > 0`; otherwise close the branch as an exploratory negative result. | `restart_plan -> runlog -> hypothesis_matrix if branch starts` | This row now owns the schema-breaking world-package rebuild plus the baseline/visocc short pretrain screen. |
| `L000B` | `6` | Under one fixed mini-CPAC protocol, does the world-package `visocc` line beat the same-lineage baseline enough to justify a follow-up `answer_only` arm? | `B` | `queued` | `<=2h once L000A ckpts exist` | `L000A` short pretrain ckpts | `results/local_patchnepa_cpac/l000b_visocc_{base,pc33,answeronly}.json` + `logs/local_patchnepa_visocc/patchnepa_visocc_l000ab_20260311/decision.json` | Promote only if `visocc` improves mini-CPAC by either `iou@0.01 >= baseline + 0.005` or `rmse <= baseline - 0.002` without catastrophic regression (`iou@0.01 < baseline - 0.003` or `rmse > baseline + 0.003`). | `restart_plan -> runlog -> storyline only if the branch materially changes the read` | Executed by the same local visibility pipeline after the short pretrain arms complete. |
| `L001` | `10` | Does the `g2` gain carry over to mini-CPAC on the current best mixed-source ckpt? | `B` | `queued` | `<=2h` | none | `runs/patchnepa_tokens/patchnepa_recong2_full300_20260306_072643/pt_pc33mesh33udf33_reconch_g2_e300/ckpt_final.pt` + `data/shapenet_unpaired_cache_v2_pc33_mesh33_udf33` | `results/local_patchnepa_cpac/l001_g2_pc33_cmp.json` | Compare against `g0 composite` mini-CPAC (`iou@0.01=0.0948`, `rmse=0.09929`). If either metric improves, mark `g2 CPAC favorable`; if both worsen, mark `FT-only gain`. | `restart_plan -> runlog -> storyline if favorable` | Same `PC context -> UDF query`, `64/64` mini-CPAC protocol as the canonical g0 rerun. |
| `L002` | `20` | For CPAC, is `mesh50udf50` or `pc33mesh33udf33` the better `g2` source ckpt under the same eval protocol? | `B` | `queued` | `<=2h` | none | `runs/patchnepa_tokens/patchnepa_recong2_full300_20260306_072643/pt_mesh50udf50_reconch_g2_e300/ckpt_final.pt` + `data/shapenet_unpaired_cache_v2_pc33_mesh33_udf33` | `results/local_patchnepa_cpac/l002_g2_mesh50udf50_cmp.json` | Use the same `iou@0.01` and `rmse` criteria as `L001`; then rank `mesh50udf50` vs `pc33mesh33udf33` under one fixed CPAC protocol. | `restart_plan -> runlog -> storyline if source ranking changes` | Eval root stays fixed to `pc33_mesh33_udf33` so only the pretrain source changes. |
| `L003` | `30` | Can strict `pointgpt_ctx_only` run cleanly enough to give a real short-readout? | `B` | `queued` | `<=2h` | none | `nepa3d/configs/shapenet_unpaired_mix_v2_tokens_drop1_pc100.yaml` | `runs/local_patchnepa/l003_pgptctxonly_pc100_s2000/ckpt_final.pt` | If the run finishes without graph/DDP failure and yields `recon_lift_q >= 0`, promote to a longer `10k` parity test; otherwise mark the parity branch `blocked`. | `restart_plan -> runlog` | Single-GPU local rerun is intentional because the earlier full run failed before scientific readout. |
| `L004` | `40` | Under `g2`, does `answer_only` keep a CPAC advantage without destroying reconstruction-side lift? | `C` | `queued` | `<=12h` | none | `nepa3d/configs/shapenet_unpaired_mix_v2_tokens_drop1_pc33_mesh33_udf33.yaml` | `runs/local_patchnepa/l004_g2_{cmp,ans}_pc33_s2000` + `results/local_patchnepa_cpac/l004_g2_{cmp,ans}_pc33.json` | `answer_only` only graduates if it keeps `recon_lift_q >= 0` and beats composite on mini-CPAC by `iou@0.01 >= composite + 0.005`. | `restart_plan -> runlog -> storyline if promoted` | Queue manifest splits this into two runnable arms: `composite` and `answer_only`. |
| `L005` | `50` | Does true `fps_then_sample` parity matter once `point_all > npoints` is real rather than a permutation-only proxy? | `B` | `blocked` | `<=2h` | real `point_all > npoints` input path required | current `scanobjectnn_*_v3_nonorm` cache is only `2048` points | target path not yet staged | Require at least two seeds with the same sign of delta before taking the effect seriously; single-run parity is not evidence. | `restart_plan -> runlog if unstaged input path is solved` | Do not rerun this on the current `2048`-point cache. |
| `L006` | `60` | Does adding rotation at FT materially improve `g2 obj_only` credibility? | `C` | `deferred` | `<=12h` | finish `L001-L004` first | `runs/patchnepa_tokens/patchnepa_recong2_full300_20260306_072643/pt_pc33mesh33udf33_reconch_g2_e300/ckpt_final.pt` + strict `obj_only` FT recipe | target run dir under `runs/local_patchnepa_ft/` | Promote only if `obj_only` gains `>= +1.0pt`; smaller movement stays a secondary recipe note. | `benchmark + restart_plan + runlog if executed` | Use direct `finetune_patch_cls.py` CLI if custom rotation knobs are needed; the shared wrapper only exposes `aug_preset`. |
| `L007` | `70` | Can the current failure mode be formalized into a one-page paper-ready diagnostic pack? | `A` | `manual` | `<=2h` | none | canonical cosine/recon metrics already stored in docs + W&B summaries | `docs/patch_nepa` figure/table spec and metric sheet | Done when `cos_tgt`, `cos_prev`, `gap`, `copy_win`, `recon_lift_q/a`, and `target_std_mean/min` are unified into one stable page. | `storyline + paper notes` | This is analysis work, not a queue-first GPU run. |
| `L008` | `80` | Is any new cosine-family escape attempt still worth spending local budget on? | `A` | `deferred` | `<=2h` | only reopen if a genuinely new target family appears | current docs already fix the decision boundary: minor cosine-only tweaks are stop | none | Keep out of the main queue unless a new objective family changes the causal picture. | `hypothesis_matrix only if reopened` | This row exists to preserve the negative-result boundary. |

## 6. Queue Mapping

- machine-readable queue manifest:
  - `scripts/local/patchnepa_local_queue.tsv`
- maintained local runner:
  - `scripts/local/patchnepa_local_queue_runner.sh`
- status view:
  - `scripts/local/patchnepa_local_status.sh`

Queue policy:

- `enabled=1` rows are runnable now
- `enabled=0` rows remain visible but are not launched by the runner
- current enabled set is intentionally limited to the visibility-first branch:
  - `L000A`
- `L001-L004` stay queued but gated behind `L000A/L000B`
