"""Compatibility shim for the CQA target-audit entrypoint."""

from nepa3d.tracks.patch_nepa.cqa.analysis.audit_cqa_targets import *  # noqa: F401,F403
from nepa3d.tracks.patch_nepa.cqa.analysis.audit_cqa_targets import main


if __name__ == "__main__":
    main()
