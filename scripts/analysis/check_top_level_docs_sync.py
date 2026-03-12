#!/usr/bin/env python3
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = REPO_ROOT / "nepa3d/docs/current_state.json"


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def require_substrings(path: Path, substrings: list[str]) -> list[str]:
    text = load_text(path)
    missing = [s for s in substrings if s not in text]
    return [f"{path}: missing substring: {s}" for s in missing]


def main() -> int:
    if not STATE_PATH.exists():
        print(f"[error] missing state file: {STATE_PATH}", file=sys.stderr)
        return 2

    state = json.loads(load_text(STATE_PATH))
    patch = state["patchnepa"]
    headline = patch["headline"]
    last_updated = state["last_updated"]

    common_metrics = [
        f"`obj_bg={headline['obj_bg']}`",
        f"`obj_only={headline['obj_only']}`",
        f"`pb_t50_rs={headline['pb_t50_rs']}`",
    ]

    checks: dict[Path, list[str]] = {
        REPO_ROOT / "nepa3d/docs/README.md": [
            f"Last updated: {last_updated}",
            patch["mainline_name"],
            patch["objective"],
            *common_metrics,
            patch["collaborator_entrypoint"],
            patch["local_execution_source"],
            patch["operations_entrypoint"],
            "nepa3d/docs/current_state.json",
        ],
        REPO_ROOT / "nepa3d/docs/llm_retrieval_index.md": [
            f"Last updated: {last_updated}",
            patch["mainline_name"],
            *common_metrics,
            patch["collaborator_entrypoint"],
            patch["local_execution_source"],
            "nepa3d/docs/current_state.json",
        ],
        REPO_ROOT / "nepa3d/docs/results_index.md": [
            f"Last updated: {last_updated}",
            patch["mainline_name"],
            *common_metrics,
            "patch_nepa/collaborator_reading_guide_active.md",
            "patch_nepa/execution_backlog_active.md",
            "operations/README.md",
            "nepa3d/docs/current_state.json",
        ],
    }

    errors: list[str] = []

    for ref_key in [
        "collaborator_entrypoint",
        "local_execution_source",
        "operations_entrypoint",
        "benchmark_doc",
        "storyline_doc",
        "hypothesis_doc",
    ]:
        ref = REPO_ROOT / patch[ref_key]
        if not ref.exists():
            errors.append(f"current_state.json points to missing file: {ref}")

    for path, substrings in checks.items():
        if not path.exists():
            errors.append(f"missing top-level doc: {path}")
            continue
        errors.extend(require_substrings(path, substrings))

    if errors:
        print("[docs-sync] FAILED")
        for err in errors:
            print(f"- {err}")
        return 1

    print("[docs-sync] OK")
    print(f"- state file: {STATE_PATH}")
    for path in checks:
        print(f"- checked: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
