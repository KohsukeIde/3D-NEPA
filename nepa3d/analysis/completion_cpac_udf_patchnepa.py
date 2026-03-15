"""Compatibility shim for legacy PatchNEPA CPAC analysis path."""

from nepa3d.tracks.patch_nepa.mainline.analysis.completion_cpac_udf_patchnepa import *  # noqa: F401,F403
from nepa3d.tracks.patch_nepa.mainline.analysis.completion_cpac_udf_patchnepa import main


if __name__ == "__main__":
    main()
