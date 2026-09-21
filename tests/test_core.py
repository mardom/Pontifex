"""Test suite for pontifex.core: features, guards, constants, and metrics."""

import numpy as np
import pytest
from pontifex.core import (
    LSST_BANDS,
    ROMAN_BANDS,
    mag_to_flux,
    flux_to_mag,
    extract_features,
    get_bands_and_ref,
    sanitize_input_catalog,
    compute_distribution_moments,
    compute_moments_bias,
    compute_photoz_point_metrics,
)


def test_core_constants():
    assert len(LSST_BANDS) == 6
    assert len(ROMAN_BANDS) == 3


def test_flux_mag_conversions():
    mag = np.array([20.0, 25.0, 30.0, np.nan, 99.0])
    flux = mag_to_flux(mag)
    assert len(flux) == 5
    assert flux[0] > flux[1] > flux[2]
    assert flux[3] == 0.0  # NaN handled cleanly

    # Invert back
    mag_rec = flux_to_mag(flux[:3])
    np.testing.assert_allclose(mag_rec, mag[:3], atol=1e-4)


def test_extract_features():
    n_obj = 50
    dummy_data = {
        "mag_u_lsst": np.random.uniform(20.0, 26.0, n_obj),
        "mag_g_lsst": np.random.uniform(19.0, 25.0, n_obj),
        "mag_r_lsst": np.random.uniform(18.0, 24.0, n_obj),
        "mag_i_lsst": np.random.uniform(17.5, 23.5, n_obj),
        "mag_z_lsst": np.random.uniform(17.0, 23.0, n_obj),
        "mag_y_lsst": np.random.uniform(16.5, 22.5, n_obj),
        "mag_u_lsst_err": np.full(n_obj, 0.1),
        "mag_g_lsst_err": np.full(n_obj, 0.05),
        "mag_r_lsst_err": np.full(n_obj, 0.05),
        "mag_i_lsst_err": np.full(n_obj, 0.05),
        "mag_z_lsst_err": np.full(n_obj, 0.05),
        "mag_y_lsst_err": np.full(n_obj, 0.05),
    }
    feats = extract_features(dummy_data)
    assert feats.shape[0] == n_obj
    assert feats.shape[1] > 6  # includes mags, masks, fluxes, errors, and colors


def test_core_guard_sanitization():
    raw = {
        "mag_i_lsst": np.array([20.0, np.nan, np.inf, 50.0, 22.0]),
        "mag_i_lsst_err": np.array([0.05, -0.1, 0.05, 0.05, 0.05]),
        "object_id": np.array([1, 2, 3, 4, 5]),
    }
    sanitized, report = sanitize_input_catalog(raw, raise_warnings=False)
    assert np.all(np.isfinite(sanitized["mag_i_lsst"]))
    assert np.all(sanitized["mag_i_lsst_err"] > 0)
    assert np.array_equal(sanitized["object_id"], raw["object_id"])


def test_distribution_moments_and_biases():
    z_grid = np.linspace(0.0, 3.0, 301)
    # Gaussian centered at 1.0 with sigma 0.2
    nz_true = np.exp(-0.5 * ((z_grid - 1.0) / 0.2) ** 2)
    nz_true /= nz_true.sum()

    mu, sig = compute_distribution_moments(z_grid, nz_true)
    assert 0.98 < mu < 1.02
    assert 0.18 < sig < 0.22

    # Slight shift
    nz_est = np.exp(-0.5 * ((z_grid - 1.01) / 0.205) ** 2)
    nz_est /= nz_est.sum()

    res = compute_moments_bias(z_grid, nz_est, nz_true)
    assert abs(res["delta_mu"]) < 0.01
    assert abs(res["delta_sigma"]) < 0.01


def test_photoz_point_metrics():
    z_spec = np.array([0.5, 0.8, 1.2, 1.5, 2.0])
    z_phot = np.array([0.51, 0.82, 1.18, 1.52, 2.01])
    metrics = compute_photoz_point_metrics(z_phot, z_spec)
    assert abs(metrics["bias"]) < 0.05
    assert metrics["sigma_mad"] < 0.05
    assert metrics["outlier_rate"] == 0.0
