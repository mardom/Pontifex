"""Pontifex Core Constants and Filter Definitions."""

from typing import Dict, List
import numpy as np

# Bandpass definitions
LSST_BANDS: List[str] = ["u", "g", "r", "i", "z", "y"]
HSC_BANDS: List[str] = ["g", "r", "i", "z", "y"]
ROMAN_BANDS: List[str] = ["Y", "J", "H"]

# Column naming conventions
MAG_COL = "mag_{band}_lsst"
MAG_ERR_COL = "mag_{band}_lsst_err"
ROMAN_MAG_COL = "mag_{band}_roman"
ROMAN_ERR_COL = "mag_{band}_roman_err"
OBJECT_ID_COL = "object_id"
REDSHIFT_COL = "redshift"
MANYBAND_COL = "redshift_manyband"

# Challenge Tomographic Bin Edges
TOMO_BIN_EDGES: Dict[str, np.ndarray] = {
    "taskset_1": np.array([0.0, 0.46, 0.74, 1.05, 1.48, 3.0]),
    "taskset_2": np.array([0.0, 0.52, 0.84, 1.20, 1.69, 3.0]),
    "taskset_3": np.array([0.0, 0.52, 0.84, 1.20, 1.69, 3.0]),
}

# Redshift Evaluation Grid Edges (300 bins, 301 edges)
Z_BIN_EDGES: Dict[str, np.ndarray] = {
    "taskset_1": np.linspace(0.0, 3.0, 301),
    "taskset_2": np.linspace(0.0, 3.0, 301),
    "taskset_3": np.linspace(0.0, 3.0, 301),
}

# Photo-z default evaluation grid
Z_GRID_DEFAULT: np.ndarray = np.linspace(0.0, 3.0, 301)
Z_CENTERS_DEFAULT: np.ndarray = 0.5 * (Z_GRID_DEFAULT[:-1] + Z_GRID_DEFAULT[1:])

# Default physical validation limits
PHYSICAL_LIMITS: Dict[str, float] = {
    "mag_min": 10.0,
    "mag_max": 38.0,
    "err_min": 1e-5,
    "err_max": 5.0,
    "ra_min": -360.0,
    "ra_max": 360.0,
    "dec_min": -90.0,
    "dec_max": 90.0,
    "z_min": 0.0,
    "z_max": 12.0,
}

