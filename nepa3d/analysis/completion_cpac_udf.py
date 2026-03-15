"""Compatibility shim for legacy QueryNEPA CPAC analysis path."""

from nepa3d.tracks.query_nepa.analysis.completion_cpac_udf import *  # noqa: F401,F403
from nepa3d.tracks.query_nepa.analysis.completion_cpac_udf import main


if __name__ == "__main__":
    main()
