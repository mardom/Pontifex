"""Pontifex Core: Shared constants, guards, feature transformers, and metrics."""

from .constants import (
    LSST_BANDS,
    HSC_BANDS,
    ROMAN_BANDS,
    MAG_COL,
    MAG_ERR_COL,
    ROMAN_MAG_COL,
    ROMAN_ERR_COL,
    OBJECT_ID_COL,
    REDSHIFT_COL,
    MANYBAND_COL,
    TOMO_BIN_EDGES,
    Z_BIN_EDGES,
    Z_GRID_DEFAULT,
    Z_CENTERS_DEFAULT,
    PHYSICAL_LIMITS,
)
from .guard import sanitize_input_catalog
from .features import (
    mag_to_flux,
    flux_to_mag,
    extract_features,
    get_bands_and_ref,
)
from .metrics import (
    compute_distribution_moments,
    compute_moments_bias,
    compute_photoz_point_metrics,
)

__all__ = [
    "LSST_BANDS",
    "HSC_BANDS",
    "ROMAN_BANDS",
    "MAG_COL",
    "MAG_ERR_COL",
    "ROMAN_MAG_COL",
    "ROMAN_ERR_COL",
    "OBJECT_ID_COL",
    "REDSHIFT_COL",
    "MANYBAND_COL",
    "TOMO_BIN_EDGES",
    "Z_BIN_EDGES",
    "Z_GRID_DEFAULT",
    "Z_CENTERS_DEFAULT",
    "PHYSICAL_LIMITS",
    "sanitize_input_catalog",
    "mag_to_flux",
    "flux_to_mag",
    "extract_features",
    "get_bands_and_ref",
    "compute_distribution_moments",
    "compute_moments_bias",
    "compute_photoz_point_metrics",
]
