"""Compatibility shim for legacy KPlane CPAC analysis path."""

from nepa3d.tracks.kplane.analysis.completion_cpac_kplane import *  # noqa: F401,F403
from nepa3d.tracks.kplane.analysis.completion_cpac_kplane import main


if __name__ == "__main__":
    main()
