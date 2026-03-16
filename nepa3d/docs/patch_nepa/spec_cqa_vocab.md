# CQA Vocabulary Spec

Last updated: 2026-03-13

This file fixes the explicit-query CQA answer vocabulary used by the additive
primitive-answering branch.

## Version

- `vocab_version = "cqa_v1"`
- source of truth: `nepa3d/data/cqa_codec.py`

## Query Types

- `0`: `ASK_NORMAL`
- `1`: `ASK_VISIBILITY`
- `2`: `ASK_CURVATURE`
- `3`: `ASK_THICKNESS`
- `4`: `ASK_CLEARANCE`
- `5`: `ASK_DISTANCE`

`query_type_vocab = 6`

## Shared Answer Vocab

- `0..127`: normal direction bins
- `128..383`: visibility signature codes
- `384..447`: curvature bins
- `448..511`: thickness bins
- `512..575`: clearance bins
- `576..639`: distance bins

`answer_vocab_size = 640`

## Rules

- these ranges are fixed in code and do not drift per config
- checkpoints and eval outputs should record `vocab_version`
- surface-aligned tasks use `surf_xyz` as the query carrier
- only `ASK_DISTANCE` uses `udf_qry_xyz`
- current `ASK_CLEARANCE` implementation is **front-clearance only**
  (`udf_surf_clear_front`); it is not yet a symmetric front/back clearance task
