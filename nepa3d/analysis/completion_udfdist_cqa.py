"""Compatibility shim for CQA udf_distance dense-grid completion analysis."""

from nepa3d.tracks.patch_nepa.cqa.analysis.completion_udfdist_cqa import *  # noqa: F401,F403
from nepa3d.tracks.patch_nepa.cqa.analysis.completion_udfdist_cqa import main


if __name__ == "__main__":
    main()
