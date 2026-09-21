"""Pontifex: Dual-Path Gated Ensemble & Footprint-Corrected Photometric Redshift Pipeline.

Unified Architecture:
- pontifex.core: Shared feature transformers, resilient guards, filter definitions, and metrics.
- pontifex.pz: Individual galaxy photometric redshift PDF estimation & Mixture-of-Experts pipeline.
- pontifex.nz: Tomographic ensemble redshift distribution n(z) reconstruction & spatial sampling.
"""

__version__ = "2.0.0"

# Unified modular subpackages
from . import core
from . import pz
from . import nz

# Backwards compatibility top-level convenience exports
from .pz.pipeline import train_and_estimate, estimate_only
from .pz.estimators import CommitteeOfExperts, Z_CENTERS, Z_GRID
from .pz.em import PontifexEM
from .core.guard import sanitize_input_catalog
from .core.features import extract_features, mag_to_flux, flux_to_mag

__all__ = [
    "__version__",
    "core",
    "pz",
    "nz",
    "train_and_estimate",
    "estimate_only",
    "CommitteeOfExperts",
    "PontifexEM",
    "sanitize_input_catalog",
    "extract_features",
    "mag_to_flux",
    "flux_to_mag",
    "Z_CENTERS",
    "Z_GRID",
]
