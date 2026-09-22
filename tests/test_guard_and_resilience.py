"""
Comprehensive Unit & Integration Test Suite for Pontifex Feature Guard & Resilience
-------------------------------------------------------------------------------------
Tests feature sanitization, UserWarning reporting, and end-to-end expert resilience
against NaN, Inf, negative errors, unphysical magnitudes, and extreme outliers.
"""

import os
import pytest
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import tables_io

# Ensure cache directory exists for HuggingFace models
os.makedirs(os.path.expanduser('~/.cache/huggingface/hub'), exist_ok=True)

from pontifex.guard import sanitize_input_catalog
from pontifex.estimators import CommitteeOfExperts, Z_CENTERS, Z_GRID, HAS_RAIL
from pontifex.em import PontifexEM
from pontifex.pipeline import train_and_estimate


def create_synthetic_corrupted_catalog(n_samples: int = 200, seed: int = 42) -> dict:
    """Helper to generate a mock dataset with intentional NaNs, Infs, negative errors, and outliers."""
    rng = np.random.default_rng(seed)
    
    bands = ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst']
    catalog = {
        "object_id": np.arange(1000, 1000 + n_samples, dtype=np.int64),
        "ra": rng.uniform(50.0, 52.0, n_samples),
        "dec": rng.uniform(-30.0, -28.0, n_samples),
        "redshift": rng.uniform(0.1, 2.5, n_samples)
    }
    
    for b in bands:
        mags = rng.uniform(20.0, 25.0, n_samples)
        errs = rng.uniform(0.01, 0.2, n_samples)
        
        # Inject intentional corruptions
        # 1. NaNs
        nan_idx = rng.choice(n_samples, size=int(0.05 * n_samples), replace=False)
        mags[nan_idx] = np.nan
        
        # 2. Infs
        inf_idx = rng.choice(n_samples, size=int(0.03 * n_samples), replace=False)
        mags[inf_idx] = np.inf
        
        # 3. Unphysical negative measurement errors
        neg_err_idx = rng.choice(n_samples, size=int(0.04 * n_samples), replace=False)
        errs[neg_err_idx] = -0.05
        
        # 4. Extreme outliers (100 sigma)
        outlier_idx = rng.choice(n_samples, size=int(0.02 * n_samples), replace=False)
        mags[outlier_idx] = 999.0
        
        catalog[b] = mags
        catalog[b + "_err"] = errs

    return catalog


def test_sanitize_input_catalog_nans_and_infs():
    """Test that sanitize_input_catalog cleans NaNs and Infs and raises UserWarning."""
    catalog = create_synthetic_corrupted_catalog(n_samples=100)
    
    with pytest.warns(UserWarning) as record:
        sanitized, report = sanitize_input_catalog(catalog, raise_warnings=True)
        
    assert len(record) > 0, "UserWarning should be emitted when corrupted features exist!"
    assert "Pontifex Feature Guard Protection Triggered" in str(record[0].message)
    
    for col, data in sanitized.items():
        if np.issubdtype(data.dtype, np.number):
            assert not np.any(np.isnan(data)), f"Column {col} still contains NaNs!"
            assert not np.any(np.isinf(data)), f"Column {col} still contains Infs!"


def test_sanitize_input_catalog_unphysical_errors():
    """Test that negative errors (err <= 0) are flagged and imputed with positive medians."""
    catalog = {
        "object_id": np.array([1, 2, 3, 4, 5]),
        "mag_i_lsst": np.array([21.0, 22.0, 23.0, 24.0, 25.0]),
        "mag_i_lsst_err": np.array([0.05, -0.10, 0.02, 0.0, 0.08])  # Contains -0.10 and 0.0
    }
    
    with pytest.warns(UserWarning):
        sanitized, report = sanitize_input_catalog(catalog, raise_warnings=True)
        
    cleaned_err = sanitized["mag_i_lsst_err"]
    assert np.all(cleaned_err > 0), "All measurement errors must be strictly positive after sanitization!"
    assert report["issues_by_column"]["mag_i_lsst_err"]["unphysical_err_count"] == 2


@pytest.mark.skipif(not HAS_RAIL, reason="RAIL is required for end-to-end committee tests")
def test_committee_resilience_to_corrupted_inputs():
    """Test that CommitteeOfExperts fits and predicts on corrupted input data without failing."""
    train_dict = create_synthetic_corrupted_catalog(n_samples=250, seed=123)
    test_dict = create_synthetic_corrupted_catalog(n_samples=50, seed=456)
    
    bands = ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst']
    ref_band = 'mag_i_lsst'
    
    committee = CommitteeOfExperts(is_ci=True)
    
    # Assert fit completes smoothly
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        model_dict = committee.fit(train_dict, bands=bands, ref_band=ref_band, is_roman=False)
        
    assert model_dict is not None
    assert "K_val" in model_dict
    
    # Assert predict completes smoothly and returns valid PDFs
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        predicted_pdfs = committee.predict(test_dict)
        
    assert predicted_pdfs.shape == (50, len(Z_CENTERS))
    assert not np.any(np.isnan(predicted_pdfs))
    assert not np.any(np.isinf(predicted_pdfs))
    
    # Probability density integral over z grid equals ~1.0
    dz = Z_CENTERS[1] - Z_CENTERS[0]
    integral = np.sum(predicted_pdfs, axis=1) * dz
    np.testing.assert_allclose(integral, 1.0, atol=1e-3)


def test_pontifex_em_resilience_to_corrupted_coordinates_and_pdfs():
    """Test that PontifexEM handles NaNs/Infs in spatial coordinates and initial PDFs."""
    rng = np.random.default_rng(789)
    n_unk = 60
    n_ref = 80
    
    unk_df = pd.DataFrame({
        'ra': [np.nan, 51.2, np.inf, 51.4] + list(rng.uniform(50.0, 52.0, n_unk - 4)),
        'dec': [-29.1, np.nan, -29.3, -29.4] + list(rng.uniform(-30.0, -28.0, n_unk - 4))
    })
    
    ref_df = pd.DataFrame({
        'ra': list(rng.uniform(50.0, 52.0, n_ref - 2)) + [np.nan, 51.5],
        'dec': list(rng.uniform(-30.0, -28.0, n_ref - 2)) + [-29.5, np.inf],
        'ztrue': rng.uniform(0.1, 2.5, n_ref)
    })
    
    # Initial PDFs with intentional NaN row
    initial_pdfs = rng.uniform(0.01, 1.0, size=(n_unk, len(Z_CENTERS)))
    initial_pdfs[5, :] = np.nan  # Corrupted PDF row
    
    em = PontifexEM(
        unk_df=unk_df,
        ref_df=ref_df,
        initial_pdfs=initial_pdfs,
        z_grid_edges=Z_GRID,
        is_ci=True
    )
    
    # Run EM optimization
    opt_pdfs = em.optimize(max_iter=2)
    assert opt_pdfs.shape == (n_unk, len(Z_CENTERS))
    assert not np.any(np.isnan(opt_pdfs))
    assert not np.any(np.isinf(opt_pdfs))


@pytest.mark.skipif(not HAS_RAIL, reason="RAIL is required for end-to-end train_and_estimate tests")
def test_train_and_estimate_end_to_end_resilience():
    """End-to-end test of train_and_estimate using HDF5 catalog files with intentional corruptions."""
    train_dict = create_synthetic_corrupted_catalog(n_samples=200, seed=111)
    test_dict = create_synthetic_corrupted_catalog(n_samples=40, seed=222)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = Path(tmpdir) / "corrupted_train.hdf5"
        test_path = Path(tmpdir) / "corrupted_test.hdf5"
        output_path = Path(tmpdir) / "output_predictions.hdf5"
        
        tables_io.write(train_dict, str(train_path))
        tables_io.write(test_dict, str(test_path))
        
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            train_and_estimate(
                train_file=train_path,
                test_file=test_path,
                output_file=output_path,
                optimize_hyperparams=False
            )
            
        assert output_path.exists(), "Final output predictions .hdf5 file must be created!"
