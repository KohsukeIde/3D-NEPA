"""Compatibility shim for legacy PatchNEPA pretrain entrypoint."""

from nepa3d.tracks.patch_nepa.mainline.train.pretrain_patch_nepa import *  # noqa: F401,F403
from nepa3d.tracks.patch_nepa.mainline.train.pretrain_patch_nepa import main


if __name__ == "__main__":
    main()
