"""Compatibility shim for legacy KPlane pretrain entrypoint."""

from nepa3d.tracks.kplane.train.pretrain_kplane import *  # noqa: F401,F403
from nepa3d.tracks.kplane.train.pretrain_kplane import main


if __name__ == "__main__":
    main()
