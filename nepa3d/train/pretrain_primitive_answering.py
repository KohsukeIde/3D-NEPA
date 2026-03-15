"""Compatibility shim for legacy CQA pretrain entrypoint."""

from nepa3d.tracks.patch_nepa.cqa.train.pretrain_primitive_answering import *  # noqa: F401,F403
from nepa3d.tracks.patch_nepa.cqa.train.pretrain_primitive_answering import main


if __name__ == "__main__":
    main()
