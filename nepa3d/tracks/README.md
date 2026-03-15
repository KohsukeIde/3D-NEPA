# Code Tracks

Track-specific code lives here so active work can be inspected without mixing
legacy QueryNEPA, PatchNEPA variants, and the additive CQA branch.

- `query_nepa/`: legacy token-level QueryNEPA line
- `patch_nepa/mainline/`: patch-level PatchNEPA base implementation
- `patch_nepa/tokens/`: active v2 token-stream PatchNEPA training
- `patch_nepa/cqa/`: additive explicit-query CQA branch

Shared model building blocks live under `nepa3d/core/models/`.

Legacy import paths under `nepa3d.models`, `nepa3d.data`, `nepa3d.train`, and
`nepa3d.analysis` are kept as compatibility shims.
