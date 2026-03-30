# Code Tracks

Track-specific code lives here so active work can be inspected without mixing
legacy QueryNEPA, PatchNEPA variants, and the additive CQA branch.

- `query_nepa/`: legacy token-level QueryNEPA line
- `patch_nepa/mainline/`: patch-level PatchNEPA base implementation
- `patch_nepa/tokens/`: active v2 token-stream PatchNEPA training
- `patch_nepa/cqa/`: additive explicit-query CQA branch
- `kplane/`: tri-plane / k-plane baseline line

Shared model building blocks live under `nepa3d/core/models/`.

Outside this directory, the important split is:

- shared active infra: `nepa3d/data/`, `nepa3d/backends/`, `nepa3d/token/`,
  `nepa3d/utils/`, `nepa3d/core/models/`
- mostly compatibility shims: `nepa3d/models/`, `nepa3d/train/`,
  `nepa3d/analysis/`

Use `tracks/*` as the canonical home for track-specific implementations and
entrypoints.
