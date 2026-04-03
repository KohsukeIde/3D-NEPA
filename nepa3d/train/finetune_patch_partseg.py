"""Compatibility shim for PatchNEPA part-seg fine-tune entrypoint."""

from nepa3d.tracks.patch_nepa.mainline.train.finetune_patch_partseg import *  # noqa: F401,F403
from nepa3d.tracks.patch_nepa.mainline.train.finetune_patch_partseg import main


if __name__ == "__main__":
    main()
