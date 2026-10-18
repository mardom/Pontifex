"""Pontifex Feature Engineering and Photometric Transformations."""

from typing import Dict, List, Tuple
import numpy as np

from .constants import LSST_BANDS, ROMAN_BANDS

PHYSICAL_LIMITS: Dict[str, float] = {
    "mag_u_lsst": 26.1,
    "mag_g_lsst": 27.4,
    "mag_r_lsst": 27.5,
    "mag_i_lsst": 26.8,
    "mag_z_lsst": 26.1,
    "mag_y_lsst": 24.9,
    "mag_Y_roman": 26.7,
    "mag_J_roman": 26.9,
    "mag_H_roman": 26.7,
}


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

    Uses FeatureGuard physical 1-sigma limiting magnitudes for non-detections
    instead of unphysical 99.0 sentinels, preserving physical dropout colors
    and SOM standardization statistics across LSST (ugrizy) and Roman (YJH) bands.
    """
    lsst_bands = [f"mag_{b}_lsst" for b in LSST_BANDS]
    roman_bands = [f"mag_{b}_roman" for b in ROMAN_BANDS]
    all_mag_cols = [b for b in lsst_bands + roman_bands if b in data_dict]

    features = []
    clean_mags: Dict[str, np.ndarray] = {}

    # 1. Physical limiting-magnitude sanitized magnitudes and NaN indicator masks
    for col in all_mag_cols:
        m = np.asarray(data_dict[col], dtype=np.float32).copy()
        lim = float(PHYSICAL_LIMITS.get(col, 26.5))
        nan_mask = ~np.isfinite(m) | (m > 35.0) | (m < 10.0)
        m_clean = np.where(nan_mask, lim, m).astype(np.float32)
        clean_mags[col] = m_clean
        features.append(m_clean)
        features.append(nan_mask.astype(np.float32))

    # 2. Linear Pogson fluxes (zero for non-detections)
    for col in all_mag_cols:
        m = np.asarray(data_dict[col], dtype=np.float32)
        nan_mask = ~np.isfinite(m) | (m > 35.0) | (m < 10.0)
        f = np.where(nan_mask, 0.0, 10.0 ** (-0.4 * (np.clip(m, 10.0, 35.0) - 24.0))).astype(np.float32)
        features.append(f)

    # 3. Measurement Errors (1.0 mag conservative error for non-detections)
    for col in all_mag_cols:
        err_col = f"{col}_err"
        if err_col in data_dict:
            err = np.asarray(data_dict[err_col], dtype=np.float32).copy()
            bad_err = ~np.isfinite(err) | (err <= 0.0) | (err > 10.0)
            err = np.where(bad_err, 1.0, err).astype(np.float32)
            features.append(err)

    # 4. Adjacent band colors (physical dropout colors without 99.0 spikes)
    for i in range(len(all_mag_cols) - 1):
        b1, b2 = all_mag_cols[i], all_mag_cols[i + 1]
        c = np.clip(clean_mags[b1] - clean_mags[b2], -5.0, 10.0).astype(np.float32)
        features.append(c)

    # 5. Wide-baseline colors
    wide_pairs = [
        ("mag_u_lsst", "mag_r_lsst"),
        ("mag_g_lsst", "mag_i_lsst"),
        ("mag_r_lsst", "mag_z_lsst"),
        ("mag_i_lsst", "mag_y_lsst"),
        ("mag_z_lsst", "mag_H_roman"),
    ]
    for b1, b2 in wide_pairs:
        if b1 in clean_mags and b2 in clean_mags:
            c = np.clip(clean_mags[b1] - clean_mags[b2], -6.0, 12.0).astype(np.float32)
            features.append(c)

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
