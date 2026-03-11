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

The current best benchmark headline is:

- `obj_bg=0.8485`
- `obj_only=0.8589`
- `pb_t50_rs=0.8140`

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
  - override with `CKPT=/abs/path/to/ckpt_final.pt`
- `submit_patchnepa_current_cpac.sh`
  - submit the current mini-CPAC (`PC context -> UDF query`) evaluation
  - default ckpt: current best mixed-source `g2` checkpoint

## Recommended Reading

1. `nepa3d/docs/patch_nepa/README.md`
2. `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
3. `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
4. `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`

## Examples

```bash
# current default pretrain
bash scripts/abci/submit_patchnepa_current_pretrain.sh

# current default finetune (uses default g2 ckpt)
bash scripts/abci/submit_patchnepa_current_ft.sh

# current default mini-CPAC
bash scripts/abci/submit_patchnepa_current_cpac.sh
```

```bash
# mesh50udf50 pretrain branch
MIX_VARIANT=mesh50udf50 bash scripts/abci/submit_patchnepa_current_pretrain.sh

# finetune a custom checkpoint
CKPT=/abs/path/to/ckpt_final.pt bash scripts/abci/submit_patchnepa_current_ft.sh
```
