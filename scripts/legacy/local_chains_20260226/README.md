# Local Chain Archive (2026-02-26)

All remaining `*_local.sh` scripts were moved from active folders to this archive.

Intent:

- keep active `scripts/` focused on reproducible scheduler entrypoints
  (`submit_*`, `*_qf.sh`, pipeline submitters).
- preserve older ad-hoc local orchestration scripts without deleting them.

Moved categories:

- `analysis/*local.sh`
- `finetune/*local.sh`
- `pretrain/*local.sh`

Selection rule:

- filename ends with `_local.sh`
- original path not already under `scripts/legacy/`
