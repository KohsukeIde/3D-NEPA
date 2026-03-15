"""Compatibility shim for legacy CQA token-eval entrypoint."""

from nepa3d.tracks.patch_nepa.cqa.analysis.eval_primitive_answering_tokens import *  # noqa: F401,F403
from nepa3d.tracks.patch_nepa.cqa.analysis.eval_primitive_answering_tokens import main


if __name__ == "__main__":
    main()
