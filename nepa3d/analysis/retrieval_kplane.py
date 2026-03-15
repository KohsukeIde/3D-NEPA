"""Compatibility shim for legacy KPlane retrieval analysis path."""

from nepa3d.tracks.kplane.analysis.retrieval_kplane import *  # noqa: F401,F403
from nepa3d.tracks.kplane.analysis.retrieval_kplane import main


if __name__ == "__main__":
    main()
