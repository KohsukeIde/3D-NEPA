"""Compatibility shim for PatchNEPA classifier import path.

`PatchTransformerNepaClassifier` now lives in `patch_nepa.py` as a sibling
class to pretrain model (`PatchTransformerNepa`), while this module is kept to
avoid breaking existing imports.
"""

from .patch_nepa import PatchTransformerNepaClassifier

__all__ = ["PatchTransformerNepaClassifier"]

