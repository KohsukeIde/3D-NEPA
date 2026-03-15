"""Compatibility shim for legacy QueryNEPA pretrain entrypoint."""

from nepa3d.tracks.query_nepa.train.pretrain import *  # noqa: F401,F403
from nepa3d.tracks.query_nepa.train.pretrain import main


if __name__ == "__main__":
    main()
