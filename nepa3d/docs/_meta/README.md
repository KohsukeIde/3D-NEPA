# Docs Meta

Last updated: 2026-05-12

## Purpose

This folder contains docs about the documentation system itself and other
cross-cutting inventories.

It is not a scientific result folder. Do not put experiment runlogs, benchmark
tables, or paper narrative drafts here.

## Files

- `docs_inventory_active.md`
  - full inventory of files under `nepa3d/docs/`
- `docs_cleanup_plan_active.md`
  - cleanup policy, merge/archive rules, and next structural cleanup targets
- `insight_register_active.md`
  - compact experiment-family findings before opening raw ledgers
- `code_inventory_active.md`
  - code ownership and boundary map
- `config_inventory_active.md`
  - config ownership and migration-risk map

## Placement Rule

Use `_meta/` when the doc answers one of these questions:

- How is the docs tree organized?
- Which doc is canonical or stale?
- What did a family of experiments teach us?
- Where does code/config ownership live?
- What should be merged, archived, or kept?

Use a topic folder instead when the doc is about a method, result, dataset,
runlog, task domain, or operation.
