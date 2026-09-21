"""Backward-compatibility shim for pontifex.estimators (migrated to pontifex.pz.estimators)."""

from .pz.estimators import (
    CommitteeOfExperts,
    get_bands_and_ref,
    Z_CENTERS,
    Z_GRID,
    extract_features,
)

__all__ = [
    "CommitteeOfExperts",
    "get_bands_and_ref",
    "Z_CENTERS",
    "Z_GRID",
    "extract_features",
]
