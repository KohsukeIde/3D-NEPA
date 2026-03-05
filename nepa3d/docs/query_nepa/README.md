# Query-NEPA Docs

This folder keeps the historical Query-NEPA line. It remains important for
traceability and comparison, but it is not the active implementation line.

## Reading Order

1. `runlog_202602.md`
   - Historical job-level ledger for Query-NEPA.
2. `pretrain_abcd_1024_multinode_active.md`
   - Main historical ledger for the large A/B/C/D study.
3. `pretrain_abcd_1024_variant_reval_active.md`
   - Variant-split re-evaluation notes and protocol caveats.

## Role

- Use this folder when asking:
  - what Query-NEPA actually achieved,
  - which early claims were invalidated by later protocol audits,
  - which settings were later ported to Patch-NEPA.
- Do not use this folder as the primary source for current Patch-NEPA claims.

## Main Boundary

- Query-NEPA results that depend on mixed ScanObjectNN cache lines
  (for example `scanobjectnn_main_split_v2`) are historical diagnostics, not
  current protocol-safe headline evidence.
- Patch-NEPA should reference Query-NEPA through explicit comparison docs in
  `nepa3d/docs/patch_nepa/`, not by merging raw ledgers.
