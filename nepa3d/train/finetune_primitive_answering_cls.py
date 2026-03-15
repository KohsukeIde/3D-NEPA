"""Compatibility shim for legacy CQA classifier finetune entrypoint."""

from nepa3d.tracks.patch_nepa.cqa.train.finetune_primitive_answering_cls import *  # noqa: F401,F403
from nepa3d.tracks.patch_nepa.cqa.train.finetune_primitive_answering_cls import main


if __name__ == "__main__":
    main()
