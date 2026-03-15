"""Compatibility shim for legacy CQA control-eval entrypoint."""

from nepa3d.tracks.patch_nepa.cqa.analysis.eval_primitive_answering_controls import *  # noqa: F401,F403
from nepa3d.tracks.patch_nepa.cqa.analysis.eval_primitive_answering_controls import main


if __name__ == "__main__":
    main()
