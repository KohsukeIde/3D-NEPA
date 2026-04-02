# ABCI Collaborator Entrypoints

This folder provides the thinnest maintained ABCI submit wrappers for the
current PatchNEPA line.

Use these when a collaborator asks "which `sh` should I run?" and does not
need the full internal script tree.

## Boundary

- this folder is the source of truth for collaborator-facing ABCI entrypoints
- local workstation launchers do not belong here; keep those in
  `scripts/local/`
- internal workers and detailed PBS helpers still belong in
  `scripts/pretrain/`, `scripts/finetune/`, `scripts/eval/`, and
  `scripts/analysis/`
- operational boundary notes live in `nepa3d/docs/operations/README.md`

## Current Defaults

- current pretrain line: PatchNEPA v2 reconstruction `g2`
- objective: `recon_chamfer`
- recon loss mode: `composite`
- generator depth: `2`
- no `rfps_cached`
- no ray path in the current token-path mainline (`N_RAY=0`)
- current mixed-source default: `pc33mesh33udf33`

Current ScanObjectNN benchmark headline status:

- `obj_bg=pending`
- `obj_only=pending`
- `pb_t50_rs=pending`

Reason:

- maintained FT policy switched on 2026-03-14 to official ScanObjectNN
  `test-as-val` parity (`val=test` style).
- previously cited `0.8485 / 0.8589 / 0.8140` remains useful as a historical
  file-split FT result, but is no longer the canonical benchmark headline.

Reference:

- `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
- `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
- `nepa3d/docs/operations/README.md`

## Entrypoints

- `submit_patchnepa_current_pretrain.sh`
  - submit current PatchNEPA token-path pretrain on ABCI
  - default mix: `pc33mesh33udf33`
  - override with `MIX_VARIANT=pc100|mesh50udf50|pc33mesh33udf33`
- `submit_patchnepa_current_ft.sh`
  - submit the current ScanObjectNN variant finetune set
  - default ckpt: current best mixed-source `g2` checkpoint
  - default FT validation policy: official `test-as-val`
  - override with `CKPT=/abs/path/to/ckpt_final.pt`
- `submit_patchnepa_current_cpac.sh`
  - submit the current mini-CPAC (`PC context -> UDF query`) evaluation
  - default ckpt: current best mixed-source `g2` checkpoint
- `submit_patchnepa_current_cqa_pretrain.sh`
  - submit the experimental explicit-query CQA branch on ABCI
  - historical default tasks: `udf_distance + mesh_normal_unsigned`
  - current default cache: `data/shapenet_unpaired_cache_v2_20260311_worldvis_drop1`
- `submit_patchnepa_geo_teacher_compare_pretrain.sh`
  - submit the matched `100`-epoch geo-teacher compare on ABCI
  - default config: `shapenet_geo_teacher_packed_distnorm_unsigned_v1.yaml`
  - default protocol: `packed + multihead + per_task + no_q`

## Recommended Reading

1. `nepa3d/docs/patch_nepa/collaborator_reading_guide_active.md`
2. `nepa3d/docs/patch_nepa/README.md`
3. `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
4. `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
5. `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`
6. `nepa3d/docs/patch_nepa/spec_cqa_vocab.md`
7. `nepa3d/docs/patch_nepa/experiment_route_ab_matrix_202604.md`

## Examples

```bash
# current default pretrain
bash scripts/abci/submit_patchnepa_current_pretrain.sh

# current default finetune (uses default g2 ckpt)
bash scripts/abci/submit_patchnepa_current_ft.sh

# current default mini-CPAC
bash scripts/abci/submit_patchnepa_current_cpac.sh

# experimental explicit-query CQA pretrain
bash scripts/abci/submit_patchnepa_current_cqa_pretrain.sh

# matched 100-epoch geo-teacher compare
bash scripts/abci/submit_patchnepa_geo_teacher_compare_pretrain.sh
```

```bash
# mesh50udf50 pretrain branch
MIX_VARIANT=mesh50udf50 bash scripts/abci/submit_patchnepa_current_pretrain.sh

# finetune a custom checkpoint
CKPT=/abs/path/to/ckpt_final.pt bash scripts/abci/submit_patchnepa_current_ft.sh
```
