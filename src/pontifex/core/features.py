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
    """Extract magnitudes, errors, Pogson fluxes, adjacent and wide-baseline colors.
    
    Automatically handles LSST bands (u, g, r, i, z, y) and optional Roman bands (Y, J, H).
    Extracts 49 features for LSST+Roman (or 44 for LSST only), matching Ascention challenge models.
    """
    lsst_bands = [f"mag_{b}_lsst" for b in LSST_BANDS]
    roman_bands = [f"mag_{b}_roman" for b in ROMAN_BANDS]
    all_mag_cols = [b for b in lsst_bands + roman_bands if b in data_dict]
    
    features = []
    
    # 1. Magnitudes and NaN indicator masks
    for col in all_mag_cols:
        m = np.asarray(data_dict[col], dtype=np.float32).copy()
        nan_mask = np.isnan(m)
        m[nan_mask] = 99.0
        features.append(m)
        features.append(nan_mask.astype(np.float32))
        
    # 2. Linear Pogson fluxes
    for col in all_mag_cols:
        m = np.asarray(data_dict[col], dtype=np.float32).copy()
        f = np.where(np.isnan(m), 0.0, 10.0 ** (-0.4 * (m - 24.0)))
        features.append(f)
        
    # 3. Measurement Errors
    for col in all_mag_cols:
        err_col = f"{col}_err"
        if err_col in data_dict:
            err = np.asarray(data_dict[err_col], dtype=np.float32).copy()
            err = np.where(np.isnan(err), 99.0, err)
            features.append(err)
            
    # 4. Adjacent band colors
    for i in range(len(all_mag_cols) - 1):
        b1, b2 = all_mag_cols[i], all_mag_cols[i + 1]
        m1 = np.where(np.isnan(data_dict[b1]), 99.0, data_dict[b1])
        m2 = np.where(np.isnan(data_dict[b2]), 99.0, data_dict[b2])
        features.append(m1 - m2)

    # 5. Wide-baseline colors (essential for photometric breaks & tomographic classification)
    if "mag_u_lsst" in data_dict and "mag_r_lsst" in data_dict:
        features.append(
            np.where(np.isnan(data_dict["mag_u_lsst"]), 99.0, data_dict["mag_u_lsst"])
            - np.where(np.isnan(data_dict["mag_r_lsst"]), 99.0, data_dict["mag_r_lsst"])
        )
    if "mag_g_lsst" in data_dict and "mag_i_lsst" in data_dict:
        features.append(
            np.where(np.isnan(data_dict["mag_g_lsst"]), 99.0, data_dict["mag_g_lsst"])
            - np.where(np.isnan(data_dict["mag_i_lsst"]), 99.0, data_dict["mag_i_lsst"])
        )
    if "mag_r_lsst" in data_dict and "mag_z_lsst" in data_dict:
        features.append(
            np.where(np.isnan(data_dict["mag_r_lsst"]), 99.0, data_dict["mag_r_lsst"])
            - np.where(np.isnan(data_dict["mag_z_lsst"]), 99.0, data_dict["mag_z_lsst"])
        )
    if "mag_i_lsst" in data_dict and "mag_y_lsst" in data_dict:
        features.append(
            np.where(np.isnan(data_dict["mag_i_lsst"]), 99.0, data_dict["mag_i_lsst"])
            - np.where(np.isnan(data_dict["mag_y_lsst"]), 99.0, data_dict["mag_y_lsst"])
        )
    if "mag_z_lsst" in data_dict and "mag_H_roman" in data_dict:
        features.append(
            np.where(np.isnan(data_dict["mag_z_lsst"]), 99.0, data_dict["mag_z_lsst"])
            - np.where(np.isnan(data_dict["mag_H_roman"]), 99.0, data_dict["mag_H_roman"])
        )
        
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

