#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "nepa3d/docs"
STATE_PATH = DOCS_ROOT / "current_state.json"
ALLOWED_ROOT_FILES = {
    "README.md",
    "current_state.json",
    "llm_retrieval_index.md",
    "results_index.md",
}
ALLOWED_ROOT_DIRS = {
    "_meta",
    "archive",
    "classification",
    "completion",
    "history",
    "operations",
    "patch_nepa",
    "query_nepa",
}
TOP_LEVEL_DOCS = [
    DOCS_ROOT / "README.md",
    DOCS_ROOT / "llm_retrieval_index.md",
    DOCS_ROOT / "results_index.md",
]
STALE_TOP_LEVEL_SNIPPETS = [
    "current PatchNEPA mainline:\n  - PatchNEPA v2 reconstruction `recong2` full300",
    "For most Patch-NEPA questions, read only:\n  - `nepa3d/docs/llm_retrieval_index.md`\n  - `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`",
]
LOCAL_REF_RE = re.compile(
    r"`((?:nepa3d|scripts|pointnepa)/[^`\s)]+)`"
    r"|\((?:\./)?((?:nepa3d|scripts|pointnepa)/[^)\s]+)\)"
)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def require_substrings(path: Path, substrings: list[str]) -> list[str]:
    text = load_text(path)
    missing = [s for s in substrings if s not in text]
    return [f"{path}: missing substring: {s}" for s in missing]


def normalize_local_ref(raw: str) -> str | None:
    ref = raw.rstrip(".,:;")
    if any(char in ref for char in "*{}<>"):
        return None
    ref = ref.split("#", 1)[0]
    ref = re.sub(r":\\d+$", "", ref)
    ref = ref.split("::", 1)[0]
    if not ref or ref in {"nepa3d/tracks/*", "nepa3d/tracks/*/configs/"}:
        return None
    return ref


def require_existing_local_refs(path: Path) -> list[str]:
    text = load_text(path)
    errors: list[str] = []
    seen: set[str] = set()
    for match in LOCAL_REF_RE.finditer(text):
        raw = match.group(1) or match.group(2)
        ref = normalize_local_ref(raw)
        if ref is None or ref in seen:
            continue
        seen.add(ref)
        if not (REPO_ROOT / ref).exists():
            errors.append(f"{path}: local reference does not exist: {ref}")
    return errors


def require_absent(path: Path, snippets: list[str]) -> list[str]:
    text = load_text(path)
    present = [s for s in snippets if s in text]
    return [f"{path}: stale top-level snippet still present: {s!r}" for s in present]


def require_docs_root_contract() -> list[str]:
    errors: list[str] = []
    for path in DOCS_ROOT.iterdir():
        if path.is_file() and path.name not in ALLOWED_ROOT_FILES:
            errors.append(
                f"{path}: unexpected root doc file; move it under a role folder "
                "or add it to the documented control-plane allowlist"
            )
        if path.is_dir() and path.name not in ALLOWED_ROOT_DIRS:
            errors.append(
                f"{path}: unexpected root doc directory; document its role before "
                "adding it beside the existing top-level folders"
            )
    for dirname in ALLOWED_ROOT_DIRS:
        readme = DOCS_ROOT / dirname / "README.md"
        if not readme.exists():
            errors.append(f"{readme}: missing README for top-level docs folder")
    return errors


def main() -> int:
    if not STATE_PATH.exists():
        print(f"[error] missing state file: {STATE_PATH}", file=sys.stderr)
        return 2

    state = json.loads(load_text(STATE_PATH))
    patch = state["patchnepa"]
    headline = patch["headline"]
    last_updated = state["last_updated"]
    historical_mainline = patch["historical_mainline_name"]
    historical_objective = patch["historical_objective"]

    common_metrics = [
        f"`obj_bg={headline['obj_bg']}`",
        f"`obj_only={headline['obj_only']}`",
        f"`pb_t50_rs={headline['pb_t50_rs']}`",
    ]

    checks: dict[Path, list[str]] = {
        REPO_ROOT / "nepa3d/docs/README.md": [
            f"Last updated: {last_updated}",
            patch["paper_direction"],
            historical_mainline,
            historical_objective,
            patch["active_route_decision"],
            patch["benchmark_status"],
            *common_metrics,
            patch["llm_brief_doc"],
            patch["paper_direction_doc"],
            patch["dataset_spec_doc"],
            patch["route_matrix_doc"],
            patch["collaborator_entrypoint"],
            patch["local_execution_source"],
            patch["operations_entrypoint"],
            patch["pointgpt_sidecar_doc"],
            "nepa3d/docs/current_state.json",
        ],
        REPO_ROOT / "nepa3d/docs/llm_retrieval_index.md": [
            f"Last updated: {last_updated}",
            patch["paper_direction"],
            historical_mainline,
            patch["active_route_decision"],
            patch["benchmark_status"],
            *common_metrics,
            patch["llm_brief_doc"],
            patch["paper_direction_doc"],
            patch["dataset_spec_doc"],
            patch["route_matrix_doc"],
            patch["collaborator_entrypoint"],
            patch["local_execution_source"],
            patch["pointgpt_sidecar_doc"],
            "nepa3d/docs/current_state.json",
        ],
        REPO_ROOT / "nepa3d/docs/results_index.md": [
            f"Last updated: {last_updated}",
            patch["paper_direction"],
            historical_mainline,
            patch["benchmark_status"],
            *common_metrics,
            "patch_nepa/current_llm_brief_active.md",
            "patch_nepa/paper_direction_geo_teacher_202604.md",
            "patch_nepa/dataset_geo_teacher_v1_spec.md",
            "patch_nepa/experiment_route_ab_matrix_202604.md",
            "patch_nepa/collaborator_reading_guide_active.md",
            "patch_nepa/execution_backlog_active.md",
            "operations/README.md",
            patch["pointgpt_sidecar_doc"],
            "nepa3d/docs/current_state.json",
        ],
    }

    errors: list[str] = []
    errors.extend(require_docs_root_contract())

    for ref_key in [
        "llm_brief_doc",
        "paper_direction_doc",
        "dataset_spec_doc",
        "route_matrix_doc",
        "geo_teacher_hypothesis_doc",
        "collaborator_entrypoint",
        "local_execution_source",
        "operations_entrypoint",
        "benchmark_doc",
        "storyline_doc",
        "hypothesis_doc",
        "itachi_results_doc",
        "pointgpt_sidecar_doc",
        "code_inventory_doc",
        "config_inventory_doc",
    ]:
        ref = REPO_ROOT / patch[ref_key]
        if not ref.exists():
            errors.append(f"current_state.json points to missing file: {ref}")

    for path, substrings in checks.items():
        if not path.exists():
            errors.append(f"missing top-level doc: {path}")
            continue
        errors.extend(require_substrings(path, substrings))
        errors.extend(require_absent(path, STALE_TOP_LEVEL_SNIPPETS))

    tier_docs = [
        REPO_ROOT / patch["llm_brief_doc"],
        REPO_ROOT / patch["paper_direction_doc"],
        REPO_ROOT / patch["dataset_spec_doc"],
        REPO_ROOT / patch["route_matrix_doc"],
        REPO_ROOT / patch["geo_teacher_hypothesis_doc"],
        REPO_ROOT / patch["collaborator_entrypoint"],
        REPO_ROOT / patch["benchmark_doc"],
        REPO_ROOT / patch["storyline_doc"],
        REPO_ROOT / patch["local_execution_source"],
        REPO_ROOT / patch["operations_entrypoint"],
        REPO_ROOT / patch["itachi_results_doc"],
        REPO_ROOT / patch["pointgpt_sidecar_doc"],
        REPO_ROOT / "pointnepa/docs/README.md",
        *TOP_LEVEL_DOCS,
    ]
    for path in tier_docs:
        if path.exists():
            errors.extend(require_existing_local_refs(path))

    if errors:
        print("[docs-sync] FAILED")
        for err in errors:
            print(f"- {err}")
        return 1

    print("[docs-sync] OK")
    print(f"- state file: {STATE_PATH}")
    for path in checks:
        print(f"- checked: {path}")
    print(f"- checked tier docs: {len(tier_docs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
