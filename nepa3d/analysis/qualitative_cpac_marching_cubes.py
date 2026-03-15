"""Compatibility shim for legacy qualitative marching-cubes analysis path."""

from nepa3d.tracks.query_nepa.analysis.qualitative_cpac_marching_cubes import *  # noqa: F401,F403
from nepa3d.tracks.query_nepa.analysis.qualitative_cpac_marching_cubes import main


if __name__ == "__main__":
    main()
