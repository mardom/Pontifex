"""Pontifex Feature Engineering and Photometric Transformations."""

from typing import Dict, List, Tuple
import numpy as np

from .constants import LSST_BANDS, ROMAN_BANDS


def mag_to_flux(mag: np.ndarray, zero_pt: float = 31.4) -> np.ndarray:
    """Convert AB magnitudes to linear flux using Pogson formulation.
    
    Safe for high magnitudes and NaNs (fills with 0.0).
    """
    valid = np.isfinite(mag) & (mag < 90.0)
    flux = np.zeros_like(mag, dtype=np.float32)
    flux[valid] = np.power(10.0, -0.4 * (mag[valid] - zero_pt))
    return flux


def flux_to_mag(flux: np.ndarray, zero_pt: float = 31.4, floor: float = 1e-5) -> np.ndarray:
    """Convert linear flux to AB magnitudes, clipping negative or sub-floor fluxes."""
    f_clipped = np.maximum(flux, floor)
    return -2.5 * np.log10(f_clipped) + zero_pt


def extract_features(data_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Extract magnitudes, errors, Pogson fluxes, and color indices from catalog dictionary.
    
    Automatically handles LSST bands (u, g, r, i, z, y) and optional Roman bands (Y, J, H).
    """
    lsst_bands = [f"mag_{b}_lsst" for b in LSST_BANDS]
    roman_bands = [f"mag_{b}_roman" for b in ROMAN_BANDS]
    all_mag_cols = [b for b in lsst_bands + roman_bands if b in data_dict]
    
    features = []
    
    # 1. Magnitudes and NaN indicator masks
    for col in all_mag_cols:
        m = np.asarray(data_dict[col], dtype=np.float32).copy()
        nan_mask = np.isnan(m) | (m > 90.0)
        m[nan_mask] = 30.0  # Safe non-detection replacement
        features.append(m)
        features.append(nan_mask.astype(np.float32))
        
        # Linear Pogson flux
        f = mag_to_flux(m)
        features.append(f)
        
    # 2. Measurement Errors
    for col in all_mag_cols:
        err_col = f"{col}_err"
        if err_col in data_dict:
            err = np.asarray(data_dict[err_col], dtype=np.float32).copy()
            err_mask = np.isnan(err) | (err <= 0)
            err[err_mask] = 1.0
            features.append(err)
            
    # 3. Consecutive and Cross-Survey Colors
    for i in range(len(all_mag_cols) - 1):
        col1 = all_mag_cols[i]
        col2 = all_mag_cols[i + 1]
        m1 = np.asarray(data_dict[col1], dtype=np.float32).copy()
        m2 = np.asarray(data_dict[col2], dtype=np.float32).copy()
        m1[np.isnan(m1) | (m1 > 90.0)] = 30.0
        m2[np.isnan(m2) | (m2 > 90.0)] = 30.0
        color = m1 - m2
        features.append(color)
        
    return np.column_stack(features)


def get_bands_and_ref(columns: List[str]) -> Tuple[List[str], str, bool]:
    """Identify available photometric bands, reference band, and Roman presence."""
    has_roman = any("roman" in c.lower() for c in columns)
    if has_roman:
        bands = ["u", "g", "r", "i", "z", "y", "Y", "J", "H"]
        ref_band = "H"
    else:
        bands = ["u", "g", "r", "i", "z", "y"]
        ref_band = "i"
    return bands, ref_band, has_roman
