"""Pontifex Feature Guard and Input Protection Module."""

import warnings
import logging
from typing import Dict, Any, Tuple
import numpy as np

from .constants import PHYSICAL_LIMITS

logger = logging.getLogger("pontifex.core.guard")

# Physical 1-sigma limiting magnitudes per band (LSST + Roman)
BAND_MAG_LIMITS: Dict[str, float] = {
    "mag_u_lsst": 26.3,
    "mag_g_lsst": 27.5,
    "mag_r_lsst": 27.7,
    "mag_i_lsst": 27.0,
    "mag_z_lsst": 26.2,
    "mag_y_lsst": 24.9,
    "mag_Y_roman": 26.7,
    "mag_J_roman": 26.9,
    "mag_H_roman": 26.7,
    "u": 26.3,
    "g": 27.5,
    "r": 27.7,
    "i": 27.0,
    "z": 26.2,
    "y": 24.9,
    "Y": 26.7,
    "J": 26.9,
    "H": 26.7,
}


def sanitize_input_catalog(
    catalog_dict: Dict[str, Any],
    nondetect_val: float = 99.0,
    raise_warnings: bool = True
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Sanitizes an input catalog dictionary for downstream expert model consumption.

    Rather than replacing non-detections (NaN/Inf/sentinel) with the bright sample median
    (which distorts SED dropout colors and fabricates high-SNR detections), non-detected
    magnitudes are mapped to their physical band-specific 1-sigma limiting magnitude
    (`BAND_MAG_LIMITS`) and their associated photometric uncertainties are set to 1.0 mag
    (SNR ~ 1), allowing heteroscedastic and template models (GPz, BPZ, MLP) to naturally
    downweight missing bands.
    """
    sanitized = {}
    guard_report = {"total_records": 0, "issues_by_column": {}}

    if not catalog_dict:
        raise ValueError("Input catalog_dict is empty!")

    first_key = list(catalog_dict.keys())[0]
    n_records = len(catalog_dict[first_key])
    guard_report["total_records"] = n_records

    if n_records == 0:
        raise ValueError("Input catalog dictionary contains zero records!")

    warnings_summary = []
    nondetect_masks: Dict[str, np.ndarray] = {}

    SKIP_COLS = {
        "object_id", "id", "ID", "objectId",
        "redshift", "redshift_manyband", "ztrue", "z_true", "zspec", "z_spec", "z", "Z",
        "ra", "dec", "RA", "DEC", "coord_ra", "coord_dec"
    }

    # First pass: sanitize magnitude and general feature columns, recording non-detection masks
    for col_name, raw_data in catalog_dict.items():
        arr = np.asarray(raw_data).copy()

        if not np.issubdtype(arr.dtype, np.number) and not np.issubdtype(arr.dtype, np.bool_):
            try:
                arr = arr.astype(np.float64)
            except Exception:
                sanitized[col_name] = arr
                continue

        if col_name in SKIP_COLS or np.issubdtype(arr.dtype, np.bool_) or np.issubdtype(arr.dtype, np.integer):
            sanitized[col_name] = arr
            continue

        is_err_col = col_name.endswith("_err") or "error" in col_name.lower() or "err" in col_name.lower()
        if is_err_col:
            continue

        arr = arr.astype(np.float64)
        col_issues = {
            "nan_count": 0,
            "inf_count": 0,
            "unphysical_err_count": 0,
            "unphysical_mag_count": 0,
            "outlier_count": 0
        }

        nan_mask = np.isnan(arr)
        inf_mask = np.isinf(arr)
        col_issues["nan_count"] = int(np.sum(nan_mask))
        col_issues["inf_count"] = int(np.sum(inf_mask))
        invalid_mask = nan_mask | inf_mask

        is_mag_col = col_name.startswith("mag_") or col_name.startswith("mag") or col_name in BAND_MAG_LIMITS
        if is_mag_col:
            sentinel_or_unphys = ((arr < PHYSICAL_LIMITS["mag_min"]) | (arr > PHYSICAL_LIMITS["mag_max"]) | (arr >= nondetect_val - 1.0)) & (~invalid_mask)
            col_issues["unphysical_mag_count"] = int(np.sum(sentinel_or_unphys))
            invalid_mask |= sentinel_or_unphys
            nondetect_masks[col_name] = invalid_mask
        else:
            valid_vals = arr[~invalid_mask]
            if len(valid_vals) > 10:
                med = np.median(valid_vals)
                q25, q75 = np.percentile(valid_vals, [25, 75])
                iqr = q75 - q25
                if iqr > 1e-6:
                    extreme_outliers = (np.abs(arr - med) > 10.0 * iqr) & (~invalid_mask)
                    col_issues["outlier_count"] = int(np.sum(extreme_outliers))
                    invalid_mask |= extreme_outliers

        total_corrupted = int(np.sum(invalid_mask))
        if total_corrupted > 0:
            freq_pct = (total_corrupted / n_records) * 100.0
            valid_vals = arr[~invalid_mask]
            if is_mag_col:
                if col_name in BAND_MAG_LIMITS:
                    fill_val = float(BAND_MAG_LIMITS[col_name])
                elif len(valid_vals) > 10:
                    fill_val = float(np.percentile(valid_vals, 99.5))
                else:
                    fill_val = 26.5
            else:
                fill_val = float(np.median(valid_vals)) if len(valid_vals) > 0 else 0.0

            arr[invalid_mask] = fill_val
            msg = (
                f"Feature Guard Alert: Column '{col_name}' contains {total_corrupted} / {n_records} "
                f"non-detected/invalid records ({freq_pct:.2f}% frequency). "
                f"Details: NaNs={col_issues['nan_count']}, Infs={col_issues['inf_count']}, "
                f"Unphysical Mags={col_issues['unphysical_mag_count']}, "
                f"Outliers={col_issues['outlier_count']}. Assigned limiting value={fill_val:.4f}."
            )
            warnings_summary.append(msg)
            logger.warning(msg)

        guard_report["issues_by_column"][col_name] = col_issues
        sanitized[col_name] = arr

    # Second pass: sanitize error columns, linking non-detections to 1.0 mag uncertainty
    for col_name, raw_data in catalog_dict.items():
        if col_name in sanitized:
            continue
        arr = np.asarray(raw_data, dtype=np.float64).copy()
        col_issues = {
            "nan_count": 0,
            "inf_count": 0,
            "unphysical_err_count": 0,
            "unphysical_mag_count": 0,
            "outlier_count": 0
        }

        nan_mask = np.isnan(arr)
        inf_mask = np.isinf(arr)
        col_issues["nan_count"] = int(np.sum(nan_mask))
        col_issues["inf_count"] = int(np.sum(inf_mask))
        invalid_mask = nan_mask | inf_mask

        unphys_err = ((arr <= 0.0) | (arr > PHYSICAL_LIMITS["err_max"])) & (~invalid_mask)
        col_issues["unphysical_err_count"] = int(np.sum(unphys_err))
        invalid_mask |= unphys_err

        # Match corresponding magnitude column to propagate non-detection mask
        paired_mag = col_name[:-4] if col_name.endswith("_err") else col_name.replace("mag_err_", "mag_")
        if paired_mag in nondetect_masks:
            invalid_mask |= nondetect_masks[paired_mag]

        total_corrupted = int(np.sum(invalid_mask))
        if total_corrupted > 0:
            freq_pct = (total_corrupted / n_records) * 100.0
            fill_val = 1.0  # 1-sigma limiting uncertainty for non-detected or unphysical error
            arr[invalid_mask] = fill_val
            msg = (
                f"Feature Guard Alert: Error column '{col_name}' contains {total_corrupted} / {n_records} "
                f"non-detected/invalid records ({freq_pct:.2f}% frequency). "
                f"Assigned 1-sigma limiting uncertainty fill_val={fill_val:.4f}."
            )
            warnings_summary.append(msg)
            logger.warning(msg)

        # Clip valid photometric errors to [0.003, 2.5] without overwriting faint errors with bright median
        arr = np.clip(arr, 0.003, 2.5)
        guard_report["issues_by_column"][col_name] = col_issues
        sanitized[col_name] = arr

    if raise_warnings and len(warnings_summary) > 0:
        full_warning = (
            f"\n[Pontifex Feature Guard Protection Triggered]\n"
            f"Detected {len(warnings_summary)} feature columns with non-detected or invalid records:\n"
            + "\n".join(f" - {w}" for w in warnings_summary)
        )
        warnings.warn(full_warning, UserWarning, stacklevel=2)

    return sanitized, guard_report


TARGET_SKIP_COLS = {
    "object_id", "id", "ID", "objectId",
    "redshift", "redshift_manyband", "ztrue", "z_true", "zspec", "z_spec", "z", "Z", "sz",
    "ra", "dec", "RA", "DEC", "coord_ra", "coord_dec",
}


class FeatureGuard:
    """Object-oriented interface to ``sanitize_input_catalog`` for defensive photometric sanitization."""

    def __init__(
        self,
        band_limits: Dict[str, float] = None,
        default_mag_err: float = 1.0,
        skip_cols: set = None,
        nondetect_val: float = 99.0,
        raise_warnings: bool = False,
    ):
        self.band_limits = dict(band_limits) if band_limits is not None else dict(BAND_MAG_LIMITS)
        self.default_mag_err = float(default_mag_err)
        self.skip_cols = set(skip_cols) if skip_cols is not None else set(TARGET_SKIP_COLS)
        self.nondetect_val = float(nondetect_val)
        self.raise_warnings = bool(raise_warnings)

    def sanitize_dict(self, catalog_dict: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Sanitize a catalog dictionary using physical 1-sigma limiting magnitudes."""
        return sanitize_input_catalog(
            catalog_dict,
            nondetect_val=self.nondetect_val,
            raise_warnings=self.raise_warnings,
        )

