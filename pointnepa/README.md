# pointNEPA / PointGPT Sidecar

This directory is the repo-local wrapper workspace for PointGPT / pointNEPA
experiments, launchers, and tracked result summaries.

It intentionally depends on the sibling `PointGPT/` tree:

- `PointGPT/` remains the external or modified upstream dependency.
- `pointnepa/` contains NEPA-repo wrapper docs, scripts, and summarized
  result artifacts.
- PatchNEPA / CQA / geo-teacher docs remain under `nepa3d/docs/`.
- Point-MAE / PCP-MAE object-SSL diagnostics remain outside this split except
  when cited as comparison context from pointNEPA result pages.

## Layout

- `pointnepa/docs/`
  - PointGPT / pointNEPA result ledgers and diagnostics summaries.
- `pointnepa/scripts/local/`
  - maintained local launchers for PointGPT / pointNEPA runs.
- `pointnepa/scripts/sanity/`
  - QF, smoke, and compatibility launchers for PointGPT / pointNEPA runs.
- `pointnepa/results/`
  - tracked PointGPT / pointNEPA result summaries, CSVs, JSON summaries, and
    compact diagnostics artifacts.

## Entry Points

- ScanObjectNN sidecar ledger:
  - `pointnepa/docs/results_scanobjectnn_active.md`
- ModelNet40 PointGPT-style protocol ledger:
  - `pointnepa/docs/results_modelnet40_active.md`
- Unique-retained diagnostics:
  - `pointnepa/docs/pointgpt_unique_retained_diagnostics_summary.md`
- Local scripts guide:
  - `pointnepa/scripts/local/README.md`
- Sanity scripts guide:
  - `pointnepa/scripts/sanity/README.md`

## Runtime Contract

Scripts resolve the repo root as `WORKDIR` and default the upstream dependency
to:

```bash
POINTGPT_DIR="${WORKDIR}/PointGPT"
```

Override `POINTGPT_DIR` only when testing against another PointGPT checkout.

## Evidence Boundary

PointGPT / pointNEPA numbers are active sidecar comparison context. They are
not PatchNEPA headline benchmark values unless a result is explicitly copied
into the canonical PatchNEPA benchmark docs under `nepa3d/docs/patch_nepa/`.
