# Code Inventory

Last updated: 2026-03-30

## Purpose

This file is the canonical code-organization boundary for `nepa3d/`.

Use it to answer:

- where active track-specific code should live,
- which top-level packages are still active shared infrastructure,
- which paths are compatibility shims only,
- how to read old runlogs without confusing historical execution paths with
  current ownership.

## Canonical Boundary

Current code should be read in four layers:

1. `track`
2. `shared`
3. `compat`
4. `ops`

If a file does not have a clear layer, it should be moved, demoted to a shim,
or explicitly documented as historical.

## Current Inventory

| path | layer | status | current role | note |
|---|---|---|---|---|
| `nepa3d/tracks/query_nepa/` | `track` | `keep` | canonical QueryNEPA-specific code | legacy line kept for reproducibility |
| `nepa3d/tracks/patch_nepa/mainline/` | `track` | `keep` | canonical PatchNEPA mainline code | active patch-level line |
| `nepa3d/tracks/patch_nepa/tokens/` | `track` | `keep` | canonical PatchNEPA token-path code | current v2 token branch |
| `nepa3d/tracks/patch_nepa/cqa/` | `track` | `keep` | canonical explicit-query CQA code | additive branch |
| `nepa3d/tracks/kplane/` | `track` | `keep` | canonical KPlane baseline code | baseline line |
| `nepa3d/core/models/` | `shared` | `keep` | shared model building blocks | imported by multiple tracks |
| `nepa3d/data/` | `shared` | `keep` | shared datasets, cache tooling, and preprocessing | active infrastructure, not a legacy-only area |
| `nepa3d/backends/` | `shared` | `keep` | shared geometry backends | active infrastructure |
| `nepa3d/token/` | `shared` | `keep` | shared tokenization / ordering utilities | active infrastructure |
| `nepa3d/utils/` | `shared` | `keep` | shared utility layer | active infrastructure |
| `nepa3d/configs/` | `shared` + `compat` | `review` | launcher-facing config pool | existing top-level YAMLs remain valid until wrappers migrate |
| `nepa3d/models/` | `compat` | `keep` | backward-compatible import shims | do not add new concrete implementations here |
| `nepa3d/train/` | `compat` | `keep` | backward-compatible `python -m ...` entrypoints | keep thin wrappers only |
| `nepa3d/analysis/` | `compat` | `keep` | backward-compatible analysis entrypoints | keep thin wrappers only |
| `scripts/local/` | `ops` | `keep` | maintained workstation entrypoints | source of truth for local execution |
| `scripts/abci/` | `ops` | `keep` | maintained collaborator-facing ABCI entrypoints | source of truth for curated ABCI usage |
| `scripts/pretrain/`, `scripts/finetune/`, `scripts/analysis/`, `scripts/eval/` | `ops` | `keep` | worker / wrapper layer | broad internal surface behind curated entrypoints |
| `scripts/legacy/` | `ops` | `archive`-style | historical launchers | provenance only unless explicitly revived |

## Interpretation Rules

### 1. Track-specific code

If code is specific to one research line, its canonical home is under
`nepa3d/tracks/*`.

Examples:

- QueryNEPA model logic: `nepa3d/tracks/query_nepa/models/`
- PatchNEPA token pretrain: `nepa3d/tracks/patch_nepa/tokens/train/`
- PatchNEPA CQA models/data/analysis: `nepa3d/tracks/patch_nepa/cqa/`

### 2. Shared active infrastructure

If code is reused across more than one track, it intentionally stays outside
`tracks/`.

Examples:

- cache preprocessing in `nepa3d/data/`
- geometry backends in `nepa3d/backends/`
- tokenizer utilities in `nepa3d/token/`
- reusable blocks in `nepa3d/core/models/`

### 3. Compatibility layer

`nepa3d/models/`, `nepa3d/train/`, and `nepa3d/analysis/` should be treated as
compatibility surfaces, not as the place to add new logic.

Allowed use:

- old import paths
- old `python -m ...` entrypoints
- thin re-export wrappers for existing scripts

Disallowed default:

- new track-specific implementations
- new long-lived ownership of active logic

### 4. Config policy

Config handling is intentionally conservative:

- when a track already has its own `configs/` directory, new track-specific
  YAML should go there first
- existing `nepa3d/configs/*.yaml` files remain valid while launchers and docs
  still reference them
- do not move an existing top-level YAML unless the corresponding wrappers and
  docs are updated in the same pass

## Historical-Path Policy

Docs must distinguish between:

- `canonical current ownership`
- `historical execution path`

Rule:

- folder guides, inventories, and collaborator entrypoints should name current
  canonical paths
- runlogs, branch memos, and historical ledgers may keep the exact path that
  was executed at the time

This means an old runlog entry like `nepa3d/train/pretrain_patch_nepa.py` is
not wrong. It is a provenance record. It should not be read as the current
ownership location of the implementation.

## Common Path Translations

Use these translations when reading older docs:

| historical path | interpret as current canonical home |
|---|---|
| `nepa3d/models/query_nepa.py` | `nepa3d/tracks/query_nepa/models/query_nepa.py` |
| `nepa3d/train/pretrain.py` | `nepa3d/tracks/query_nepa/train/pretrain.py` |
| `nepa3d/models/patch_nepa.py` | `nepa3d/tracks/patch_nepa/mainline/models/patch_nepa.py` |
| `nepa3d/train/pretrain_patch_nepa.py` | `nepa3d/tracks/patch_nepa/mainline/train/pretrain_patch_nepa.py` |
| `nepa3d/train/pretrain_patch_nepa_tokens.py` | `nepa3d/tracks/patch_nepa/tokens/train/pretrain_patch_nepa_tokens.py` |
| `nepa3d/models/primitive_answering.py` | `nepa3d/tracks/patch_nepa/cqa/models/primitive_answering.py` |
| `nepa3d/analysis/audit_cqa_targets.py` | `nepa3d/tracks/patch_nepa/cqa/analysis/audit_cqa_targets.py` |
| `nepa3d/models/kplane.py` | `nepa3d/tracks/kplane/models/kplane.py` |
| `nepa3d/train/pretrain_kplane.py` | `nepa3d/tracks/kplane/train/pretrain_kplane.py` |

## Current Rules For New Work

1. Add new track-specific Python only under `nepa3d/tracks/*`.
2. Add new shared Python only under `nepa3d/data/`, `backends/`, `token/`,
   `utils/`, or `core/models/` when it is genuinely cross-track.
3. Keep `nepa3d/models/`, `train/`, and `analysis/` thin.
4. Prefer `scripts/local/` and `scripts/abci/` as human-facing entrypoints.
5. Keep active guide docs aligned to canonical paths, while preserving raw
   historical paths in ledgers.

## Immediate Follow-Through

The next conservative cleanup steps are:

1. inventory top-level `nepa3d/configs/` by owner track and migration risk
2. repoint maintained wrappers to `nepa3d.tracks.*` where that reduces
   ambiguity without breaking old flows
3. keep shrinking top-level shim files until they are unambiguously thin
