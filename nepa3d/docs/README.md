# NEPA3D Docs Hub

Last updated: 2026-05-12

## Current Snapshot

- paper-facing direction:
  - derived geometric teacher pretraining for point-context encoders
  - short name: geo-teacher
- historical mainline kept for provenance:
  - PatchNEPA v2 reconstruction `recong2` full300
  - `recon_chamfer`, `composite`, generator depth `2`
- current route decision:
  - Route A if matched 100-epoch geo-teacher pretraining is favorable on ScanObjectNN and ShapeNetPart transfer; Route B if the strongest signal remains direct geometry readouts, controls, and completion.
- current benchmark status:
  - ScanObjectNN PatchNEPA headline remains pending under official test-as-val; historical file-split FT numbers are internal only.
  - `obj_bg=pending`
  - `obj_only=pending`
  - `pb_t50_rs=pending`
- machine-readable state source:
  - `nepa3d/docs/current_state.json`

## Start Here

- Current LLM brief:
  - `nepa3d/docs/patch_nepa/current_llm_brief_active.md`
- Claude compact context:
  - `nepa3d/docs/patch_nepa/claude_context_active.md`
- Paper-facing direction:
  - `nepa3d/docs/patch_nepa/paper_direction_geo_teacher_202604.md`
- Dataset / split semantics:
  - `nepa3d/docs/patch_nepa/dataset_geo_teacher_v1_spec.md`
- Route A/B decision matrix:
  - `nepa3d/docs/patch_nepa/experiment_route_ab_matrix_202604.md`
- Collaborator reading guide:
  - `nepa3d/docs/patch_nepa/collaborator_reading_guide_active.md`
- LLM retrieval index:
  - `nepa3d/docs/llm_retrieval_index.md`
- Results index:
  - `nepa3d/docs/results_index.md`

## Physical Layout Contract

Root files are the control plane. Only these should live directly under
`nepa3d/docs/`:

- `README.md`
  - human entrypoint and physical tree map
- `current_state.json`
  - machine-readable current state
- `llm_retrieval_index.md`
  - authoritative LLM retrieval contract
- `results_index.md`
  - result and benchmark navigation surface

Everything else should live in a folder. Parallel folders are grouped by role:

- `_meta/`
  - docs governance, inventories, cleanup state, insight register, code/config
    ownership
- `patch_nepa/`
  - current PatchNEPA / geo-teacher paper-facing method, protocol, evidence,
    and PatchNEPA provenance
- `classification/`
  - NEPA / PatchNEPA classification task ledgers
- `completion/`
  - UCPR / CPAC / completion-domain ledgers
- `operations/`
  - execution-surface boundaries and local-vs-ABCI policy
- `query_nepa/`
  - historical QueryNEPA ledgers; not current truth by default
- `history/`
  - cross-era narratives and broad provenance
- `archive/`
  - frozen low-priority docs excluded from default retrieval

New loose root files are not allowed unless they are part of the control plane.
New scientific docs should go to the nearest topic folder; new docs about docs
should go to `_meta/`.

## Current Evidence Boundaries

- Canonical ScanObjectNN benchmark page:
  - `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
- File-split demotion rule:
  - `nepa3d/docs/patch_nepa/scanobjectnn_ft_policy_audit_active.md`
- Historical PatchNEPA trajectory:
  - `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
- CQA / geo-teacher hypothesis surface:
  - `nepa3d/docs/patch_nepa/hypothesis_matrix_geo_teacher_v1.md`
- Older PatchNEPA hypothesis surface:
  - `nepa3d/docs/patch_nepa/hypothesis_matrix_active.md`
  - historical/mixed March surface; read after the geo-teacher brief when the
    question needs reconstruction-era context.
- Itachi-local result ledger:
  - `nepa3d/docs/patch_nepa/itachi/results_geo_teacher_itachi_active.md`
  - local evidence only unless copied into canonical benchmark pages.
- PointGPT / pointNEPA sidecar ledger:
  - `pointnepa/docs/results_scanobjectnn_active.md`
  - active comparison context, not a PatchNEPA headline page.

## Default Retrieval Policy

- For current paper / method questions, read:
  - `nepa3d/docs/patch_nepa/current_llm_brief_active.md`
  - `nepa3d/docs/patch_nepa/claude_context_active.md`
  - `nepa3d/docs/patch_nepa/paper_direction_geo_teacher_202604.md`
  - `nepa3d/docs/patch_nepa/dataset_geo_teacher_v1_spec.md`
  - `nepa3d/docs/patch_nepa/experiment_route_ab_matrix_202604.md`
- For benchmark status, add:
  - `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
  - `nepa3d/docs/patch_nepa/scanobjectnn_ft_policy_audit_active.md`
- For local execution / next-run questions, add:
  - `nepa3d/docs/patch_nepa/execution_backlog_active.md`
  - `nepa3d/docs/operations/README.md`
- For code-layout / ownership questions, add:
  - `nepa3d/docs/_meta/code_inventory_active.md`
  - `nepa3d/docs/_meta/config_inventory_active.md`
- Read raw runlogs only when exact job provenance is required.
- Do not retrieve `nepa3d/docs/archive/` by default.

## Operations And Code

- Operations boundary:
  - `nepa3d/docs/operations/README.md`
- ABCI collaborator entrypoints:
  - `scripts/abci/README.md`
- Local workstation entrypoints:
  - `scripts/local/README.md`
- Code organization inventory:
  - `nepa3d/docs/_meta/code_inventory_active.md`
- Config ownership inventory:
  - `nepa3d/docs/_meta/config_inventory_active.md`
- Track split index:
  - `nepa3d/docs/patch_nepa/nepa_tracks_index.md`
  - historical/mixed track map; current code ownership is in
    `nepa3d/docs/_meta/code_inventory_active.md`.

## Folder Map

- `nepa3d/docs/_meta/`: docs governance, inventories, insight register, and
  code/config ownership maps
- `nepa3d/docs/patch_nepa/`: current PatchNEPA / geo-teacher synthesis,
  protocol, local evidence, and provenance ledgers
- `nepa3d/docs/classification/`: NEPA / PatchNEPA ScanObjectNN and ModelNet
  classification ledgers
- `pointnepa/docs/`: PointGPT / pointNEPA sidecar ledgers outside the
  `nepa3d/docs/` tree
- `nepa3d/docs/completion/`: historical UCPR/CPAC/completion ledgers
- `nepa3d/docs/query_nepa/`: QueryNEPA historical ledgers and runlog
- `nepa3d/docs/history/`: cross-era historical narratives
- `nepa3d/docs/operations/`: execution-surface boundary
- `nepa3d/docs/archive/`: frozen low-priority docs excluded from default retrieval

## Update Policy

- Top-level docs contract:
  - `nepa3d/docs/README.md`, `nepa3d/docs/llm_retrieval_index.md`, and
    `nepa3d/docs/results_index.md` must reflect `nepa3d/docs/current_state.json`.
  - Update all four when paper direction, benchmark status, retrieval order,
    or collaborator entrypoints change.
- Validation:
  - run `python3 scripts/analysis/check_top_level_docs_sync.py` after updates.
  - commit-time enforcement:
    - install tracked hooks with `bash scripts/analysis/install_git_hooks.sh`
    - this activates `.githooks/pre-commit`, which blocks commits when the
      top-level docs are out of sync.
- Put headline ScanObjectNN benchmark numbers only in
  `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`.
- Put paper-facing protocol decisions in:
  - `nepa3d/docs/patch_nepa/paper_direction_geo_teacher_202604.md`
  - `nepa3d/docs/patch_nepa/dataset_geo_teacher_v1_spec.md`
  - `nepa3d/docs/patch_nepa/experiment_route_ab_matrix_202604.md`
- Put local-only Itachi facts under `nepa3d/docs/patch_nepa/itachi/`.
- Keep raw runlogs append-only provenance.

## Notes

- Historical file-split FT numbers can explain why a branch was promising, but
  they are not current benchmark headlines.
- Itachi 100/300-epoch results are useful local evidence, but benchmark-facing
  claims must be copied into canonical benchmark docs first.
- Existing log helpers under `scripts/logs/` are:
  - `scripts/logs/cleanup_stale_pids.sh`
  - `scripts/logs/show_pipeline_status.sh`
