# World v3 Schema

This file freezes the raw world-package contract used after the March 2026
dataset-freeze sprint.

The goal is not to freeze the paper theme. The goal is to freeze the **raw
cache contract** so downstream experiments can be compared on one stable data
definition.

For the paper-facing dataset and protocol semantics layered above this raw
contract, read `dataset_geo_teacher_v1_spec.md`.

## Carriers

- `surf_xyz`
  - shared surface carrier
- `udf_qry_xyz`
  - volume / off-surface query carrier
- `pc_ctx_bank_xyz`
  - observation-context carrier

## Core Raw Fields

### Metadata / provenance

- `world_v3`
- `synset`
- `model_id`
- `mesh_source_path`
- `norm_center`
- `norm_scale`
- `bbox_min`
- `bbox_max`
- `surface_area`
- `volume`
- `is_watertight`
- `is_winding_consistent`
- `vertex_count`
- `face_count`

### Shared surface carrier

- `surf_xyz`
- `surf_face_idx`
- `surf_bary`

### Point-cloud observation context

- `pc_ctx_bank_xyz`
- `pc_ctx_bank_n`
- `pc_ctx_bank_density`
- `pc_ctx_bank_view_dir`
- `pc_xyz`
- `pc_n`
- `pc_density`

### Surface-aligned mesh answers

- `mesh_surf_n`
- `mesh_surf_curv`
- `mesh_surf_vis_sig`
- `mesh_surf_viscount`
- `mesh_surf_ao`

### Surface-aligned UDF answers

- `udf_surf_t_in`
- `udf_surf_t_out`
- `udf_surf_hit_out`
- `udf_surf_thickness`
- `udf_surf_clear_front`
- `udf_surf_clear_back`
- `udf_probe_deltas`
- `udf_surf_probe_front`
- `udf_surf_probe_back`
- `udf_surf_probe_thickness`

### Volume-query UDF answers

- `udf_qry_xyz`
- `udf_qry_dist`
- `udf_qry_grad`
- `udf_qry_src_code`
- `udf_qry_cp_xyz`
- `udf_qry_cp_n`

## Auxiliary Query Packs

These remain available for diagnostics / legacy compatibility but are not the
schema center.

- `mesh_qry_xyz`
- `mesh_qry_face_idx`
- `mesh_qry_bary`
- `mesh_qry_n`
- `mesh_qry_curv`
- `mesh_qry_vis_sig`
- `mesh_qry_viscount`
- `mesh_qry_ao`
- `pc_qry_xyz`
- `pc_qry_n`
- `pc_qry_density`
- `ray_o`
- `ray_d`
- `ray_t`
- `ray_hit`
- `ray_n`
- `ray_hit_xyz`
- `ray_face_idx`

## Quality / validity summary fields

These are shape-level summary fields added so audits do not need to rescan the
full raw arrays every time.

- `visibility_fallback_used`
  - `1` when exact triangle-ray visibility is unavailable and the local
    normal-based fallback is used
- `mesh_surf_vis_allzero_rate`
- `mesh_qry_vis_allzero_rate`
- `udf_surf_max_t`
- `udf_surf_hit_out_rate`
- `udf_clear_front_valid_rate`
- `udf_clear_back_valid_rate`
- `udf_probe_valid_rate`

## Policy

- Raw NPZs store **atomic observables**, not packed `ans_feat`.
- Discretization / codebooks / token order are **not** part of this contract.
- Unpairedness is implemented by split / manifest, not by deleting paired raw
  information from the NPZ.
- Future experiments may change tasks, codecs, or models, but they should not
  silently change the semantics of the fields above without bumping the schema.
