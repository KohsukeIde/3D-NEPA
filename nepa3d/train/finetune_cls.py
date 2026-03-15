"""Compatibility shim for legacy QueryNEPA finetune entrypoint."""

from nepa3d.tracks.query_nepa.train.finetune_cls import *  # noqa: F401,F403
from nepa3d.tracks.query_nepa.train.finetune_cls import main


if __name__ == "__main__":
    main()
