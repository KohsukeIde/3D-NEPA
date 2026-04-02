# PatchNEPA Itachi Notes

This subdirectory contains workstation-specific notes for `itachi`.

Use this area only for machine-dependent operational notes, such as:

- local data/cache roots under `/mnt/urashima/users/minesawa/3D-NEPA-data`
- local launcher policy under `scripts/local/patchnepa_data/`
- local geo-teacher pretrain launcher policy under
  `scripts/local/patchnepa_geo_teacher/`
- local geo-teacher post-train / downstream launcher policy under
  `scripts/local/patchnepa_geo_teacher/`
- local timeout / defer rules for long-tail preprocessing jobs
- local benchmark / recovery notes that are not ABCI-facing

Do not move paper claims, benchmark interpretation, or protocol source-of-truth
docs here. Those remain in the parent `patch_nepa/` directory.

## Current notes

- `local_data_ops_202604.md`
  - local-only data preparation boundary and launcher placement
- `local_geo_teacher_pretrain_ops_202604.md`
  - local-only multi-GPU geo-teacher pretrain boundary and launch policy
- `local_geo_teacher_posttrain_ops_202604.md`
  - local-only downstream chain after the current geo-teacher pretrain
