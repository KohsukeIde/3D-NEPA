import argparse
import json
import os
import shutil


def _materialize(src, dst, mode):
    if mode == "symlink":
        os.symlink(src, dst)
        return
    if mode == "hardlink":
        os.link(src, dst)
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    raise ValueError(f"unknown mode: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_cache_root", type=str, required=True)
    ap.add_argument("--split_json", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train_mesh", "train_pc", "train_udf", "eval"],
    )
    ap.add_argument(
        "--link_mode",
        type=str,
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src_root = os.path.abspath(args.src_cache_root)
    out_root = os.path.abspath(args.out_root)
    want_splits = set(args.splits)

    with open(args.split_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    records = payload.get("records", [])
    if not isinstance(records, list) or len(records) == 0:
        raise ValueError("split_json must contain non-empty 'records'")

    created = 0
    skipped = 0
    missing = 0

    for r in records:
        split = str(r["split"])
        if split not in want_splits:
            continue

        key = str(r["key"])
        synset, inst = key.split("/")
        src_rel = str(r["src_relpath"])
        src = os.path.join(src_root, src_rel)
        dst = os.path.join(out_root, split, synset, f"{inst}.npz")

        if not os.path.isfile(src):
            missing += 1
            continue

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.lexists(dst):
            if not args.overwrite:
                skipped += 1
                continue
            os.remove(dst)

        _materialize(src, dst, args.link_mode)
        created += 1

    meta_dir = os.path.join(out_root, "_meta")
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "split_source.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "src_cache_root": src_root,
                "split_json": os.path.abspath(args.split_json),
                "splits": sorted(list(want_splits)),
                "link_mode": args.link_mode,
                "created": created,
                "skipped": skipped,
                "missing": missing,
            },
            f,
            indent=2,
        )

    print(f"[materialize] out_root={out_root}")
    print(f"[materialize] created={created} skipped={skipped} missing={missing}")


if __name__ == "__main__":
    main()
