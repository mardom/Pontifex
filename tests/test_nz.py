"""Test suite for pontifex.nz: SOM transfer, classifier, calibration, and spatial sampler."""

import numpy as np
import pytest
from pontifex.nz import (
    compute_som_density_weights,
    train_tomographic_classifier,
    predict_tomographic_bins,
    build_calibration_histograms,
    generate_correlated_realizations,
    create_qp_samples_ensemble,
)


def test_som_density_weights():
    rng = np.random.default_rng(42)
    # DDF shifted faint, WFD broader
    X_ddf = rng.normal(loc=22.0, scale=1.0, size=(100, 6))
    X_wfd = rng.normal(loc=23.0, scale=1.5, size=(200, 6))

    weights = compute_som_density_weights(X_ddf, X_wfd, grid_size=(4, 4), num_iteration=100)
    assert len(weights) == 100
    assert np.all(weights > 0)
    assert np.isclose(np.mean(weights), 1.0, atol=1e-2)


def test_classifier_and_moe_regularization():
    rng = np.random.default_rng(42)
    n_samples = 120
    n_features = 10
    n_bins = 5

    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    y = rng.integers(0, n_bins, size=n_samples)

    # Use CPU for quick pytest
    clf = train_tomographic_classifier(
        X, y, n_estimators=10, max_depth=3, use_gpu=False
    )
    assert clf is not None

    # Predict with MoE entropy regularization
    X_test = rng.normal(size=(30, n_features)).astype(np.float32)
    bins, probs = predict_tomographic_bins(clf, X_test, n_tomo_bins=n_bins)

    assert len(bins) == 30
    assert probs.shape == (30, n_bins)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-4)


def test_calibration_histograms():
    z_grid = np.linspace(0.0, 3.0, 31)
    z_true = np.array([0.2, 0.3, 0.8, 0.9, 1.5, 1.6])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    n_tomo_bins = 3

    hists = build_calibration_histograms(z_true, y_pred, n_tomo_bins, z_grid)
    assert hists.shape == (3, 30)
    for k in range(3):
        assert np.isclose(hists[k].sum(), 1.0)


def test_spatial_sampler_and_qp():
    z_grid = np.linspace(0.0, 3.0, 21)
    n_tomo_bins = 2
    n_realizations = 10
    dummy_hists = np.ones((n_tomo_bins, 20)) / 20.0

    realization_matrix, bin_idx, i_real = generate_correlated_realizations(
        dummy_hists, z_grid, n_realizations=n_realizations
    )

    assert realization_matrix.shape == (n_tomo_bins * n_realizations, 20)
    assert len(bin_idx) == n_tomo_bins * n_realizations
    assert len(i_real) == n_tomo_bins * n_realizations

    # Verify qp ensemble creation
    ens = create_qp_samples_ensemble(dummy_hists, z_grid, n_realizations=n_realizations)
    assert ens is not None
    assert len(ens) == n_tomo_bins * n_realizations
