"""Compatibility shim for legacy QueryNEPA retrieval analysis path."""

from nepa3d.tracks.query_nepa.analysis.retrieval_ucpr import *  # noqa: F401,F403
from nepa3d.tracks.query_nepa.analysis.retrieval_ucpr import main


if __name__ == "__main__":
    main()
