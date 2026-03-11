# Operations Boundary

This folder separates execution-surface notes from scientific result ledgers.

Use it to answer:

- which scripts are the primary local entrypoints?
- which scripts are the primary ABCI entrypoints?
- where should new operational notes live so they do not pollute benchmark or
  ablation docs?

## Source Of Truth

- workstation / local GPU execution:
  - `scripts/local/README.md`
- collaborator-facing ABCI execution:
  - `scripts/abci/README.md`
- supporting sanity / compatibility scripts:
  - `scripts/sanity/README.md`

## Boundary Rules

- local-only launchers, queue manifests, and workstation pipelines belong
  under `scripts/local/`
- ABCI-facing wrappers belong under `scripts/abci/`
- `scripts/sanity/` is not a primary launch surface; use it only for sanity
  jobs, screening, environment checks, or compatibility shims
- scientific conclusions, benchmark tables, and ablation interpretations stay
  in the task docs under `nepa3d/docs/patch_nepa/`,
  `nepa3d/docs/classification/`, `nepa3d/docs/completion/`, and related
  folders

## Documentation Rule

Do not mix machine-local execution notes into benchmark ledgers or storyline
docs unless the operational detail changes the scientific interpretation.

When operational guidance is needed:

1. document the execution surface here
2. link to the maintained script README
3. keep result interpretation in the science docs
