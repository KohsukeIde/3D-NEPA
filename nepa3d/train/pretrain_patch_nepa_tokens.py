"""Compatibility shim for legacy PatchNEPA token pretrain entrypoint."""

from nepa3d.tracks.patch_nepa.tokens.train.pretrain_patch_nepa_tokens import *  # noqa: F401,F403
from nepa3d.tracks.patch_nepa.tokens.train.pretrain_patch_nepa_tokens import main


if __name__ == "__main__":
    main()
