#!/usr/bin/env python3
"""Summarize PF3 mask/order pre-flight decisions.

This script focuses on pretext diagnostics. Downstream PB-T50-RS numbers are often
in separate fine-tune logs; pass --results-csv if you have a CSV with columns:
run_id,pb_t50_rs,obj_bg,obj_only,shapenetpart_inst_miou.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path | None):
    if path is None or not path.exists():
        return {}
    with path.open() as f:
        return {r["run_id"]: r for r in csv.DictReader(f)}


def to_float(x):
    if x is None or x == "" or x == "None":
        return None
    try:
        return float(x)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--manifest", default="posetnepa_mask_order_preflight/generated/manifest.json")
    ap.add_argument("--diag-csv", default="posetnepa_mask_order_preflight/generated/pretext_diag.csv")
    ap.add_argument("--results-csv", default="")
    ap.add_argument("--out-md", default="posetnepa_mask_order_preflight/generated/pf3_summary.md")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    manifest = json.loads((repo / args.manifest).read_text())
    diag = read_csv(repo / args.diag_csv)
    results = read_csv(repo / args.results_csv) if args.results_csv else {}

    # Identify baseline rows.
    runs = manifest["runs"]
    base_m0 = next((e for e in runs if e["order"] == "simplified_morton" and abs(float(e["mask_ratio"]) - 0.0) < 1e-9), None)
    base_m7 = next((e for e in runs if e["order"] == "simplified_morton" and abs(float(e["mask_ratio"]) - 0.7) < 1e-9), None)

    def dval(run_id, key):
        return to_float(diag.get(run_id, {}).get(key))
    def rval(run_id, key):
        return to_float(results.get(run_id, {}).get(key))

    lines = []
    lines.append("# PF3 mask/order pre-flight summary\n")
    lines.append("## Kill Test 0 — mask-off naive AR copy shortcut\n")
    if base_m0:
        rid = base_m0["run_id"]
        cw = dval(rid, "copy_win")
        gap = dval(rid, "gap")
        lines.append(f"Baseline `{rid}`: copy_win={cw}, gap={gap}.\n")
        if cw is None or gap is None:
            lines.append("- Status: incomplete; diagnostic not found in logs.\n")
        elif cw >= 0.60 or gap <= 0.02:
            lines.append("- Status: GO for copy-shortcut motivation.\n")
        elif cw >= 0.35:
            lines.append("- Status: partial; copy is supporting, not primary.\n")
        else:
            lines.append("- Status: no strong copy shortcut; use filtration/coherence, not copy, as primary motivation.\n")
    else:
        lines.append("No mask-off simplified_morton baseline found in manifest.\n")

    lines.append("\n## Kill Test 1 — order effect under fixed grouping\n")
    for mask in sorted(set(float(e["mask_ratio"]) for e in runs)):
        base = next((e for e in runs if e["order"] == "simplified_morton" and abs(float(e["mask_ratio"]) - mask) < 1e-9), None)
        if not base:
            continue
        base_rid = base["run_id"]
        base_pb = rval(base_rid, "pb_t50_rs")
        base_cw = dval(base_rid, "copy_win")
        lines.append(f"\n### mask={mask}\n")
        lines.append("| run | order | PB-T50-RS | ΔPB | copy_win | Δcopy_win | gap |\n")
        lines.append("|---|---|---:|---:|---:|---:|---:|\n")
        for e in runs:
            if abs(float(e["mask_ratio"]) - mask) > 1e-9:
                continue
            rid = e["run_id"]
            pb = rval(rid, "pb_t50_rs")
            cw = dval(rid, "copy_win")
            gap = dval(rid, "gap")
            dpb = None if pb is None or base_pb is None else pb - base_pb
            dcw = None if cw is None or base_cw is None else cw - base_cw
            def fmt(x): return "" if x is None else f"{x:.4f}"
            lines.append(f"| {rid} | {e['order']} | {fmt(pb)} | {fmt(dpb)} | {fmt(cw)} | {fmt(dcw)} | {fmt(gap)} |\n")

    if results:
        lines.append("\n## Downstream result table\n")
        lines.append("| run | order | mask | obj_bg | obj_only | PB_T50_RS |\n")
        lines.append("|---|---|---:|---:|---:|---:|\n")
        for e in runs:
            rid = e["run_id"]
            row = results.get(rid, {})
            def raw(key): return row.get(key, "")
            lines.append(
                f"| {rid} | {e['order']} | {e['mask_ratio']} | "
                f"{raw('obj_bg')} | {raw('obj_only')} | {raw('pb_t50_rs')} |\n"
            )

    lines.append("\n## Decision guardrails\n")
    lines.append("- Do not treat a Stage-1 single-run gap below 0.5 percentage points as evidence.\n")
    lines.append("- Treat a PB_T50_RS gain of 1.0 point or more as enough to justify Stage 2/3, not as a final paper claim.\n")
    lines.append("- Use `fixed_random` as the deterministic random-order control. A bad stochastic `random` result alone is not proof of PosetNEPA; prefer geometry orders versus `simplified_morton`, `morton`, `fixed_random`, and `axis_x` for the core claim.\n")
    lines.append("- If mask-off has no copy signal, keep PosetNEPA alive only under the filtration/coherence framing.\n")

    lines.append("\n## Interpretation checklist\n")
    lines.append("- If geometry order improves PB-T50-RS by >= +1.0% at mask=0.7, PosetNEPA has a strong masking-complementarity story.\n")
    lines.append("- If geometry order improves only at mask=0.0, PosetNEPA is mainly an alternative to masking; this is weaker unless diagnostics are very clear.\n")
    lines.append("- If geometry order changes copy_win but not accuracy, frame as a diagnostic/shortcut-reduction result, not an accuracy headline.\n")
    lines.append("- If order effects are within ±0.5% and diagnostics do not move, do not commit to PosetNEPA-lite; implement full frontier loss or pivot.\n")

    out = repo / args.out_md
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines))
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
