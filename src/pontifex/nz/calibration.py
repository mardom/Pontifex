"""Pontifex Calibration: Empirical n(z) Histograms & Transfer Calibration."""

from typing import Optional
import numpy as np


def build_calibration_histograms(
    z_true: np.ndarray,
    y_pred: np.ndarray,
    n_tomo_bins: int,
    grid_edges: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    smoothing: float = 0.03,
    probs_oof: Optional[np.ndarray] = None,
    hard_weight: float = 0.90,
) -> np.ndarray:
    """Construct smoothed empirical n_k(z) calibration histograms for each tomographic bin.

    When out-of-fold posterior probabilities `probs_oof` are provided, blends the
    SOM-weighted hard-assigned OOF histogram with the soft posterior-weighted OOF
    histogram (`hard_weight * hist_hard + (1 - hard_weight) * hist_soft`) to capture
    smooth boundary spillover while preserving sharp bin cores.

    Parameters
    ----------
    z_true : np.ndarray
        True/clean spectroscopic or many-band training redshifts.
    y_pred : np.ndarray
        Predicted out-of-fold tomographic bin index for each training galaxy.
    n_tomo_bins : int
        Number of tomographic bins.
    grid_edges : np.ndarray
        Redshift evaluation grid edges (e.g. 301 edges for 300 bins).
    sample_weights : Optional[np.ndarray]
        SOM transfer density ratio weights.
    smoothing : float
        Laplace smoothing term added to prevent log-loss divergence and zero-density artifacts.
    probs_oof : Optional[np.ndarray]
        Out-of-fold posterior probability matrix of shape (n_gal, n_tomo_bins).
    hard_weight : float
        Weight on the hard-assigned OOF histogram when `probs_oof` is provided.

    Returns
    -------
    calib_hists : np.ndarray
        Shape (n_tomo_bins, n_bins) array of normalized probability vectors.
    """
    calib_hists = []
    n_bins = len(grid_edges) - 1
    for k in range(n_tomo_bins):
        mask_k = (y_pred == k)
        if np.sum(mask_k) == 0 and probs_oof is None:
            hist_k = np.ones(n_bins, dtype=np.float64) / n_bins
        else:
            if sample_weights is not None:
                w_hard = sample_weights[mask_k]
                hist_hard = np.histogram(z_true[mask_k], grid_edges, weights=w_hard)[0].astype(np.float64)
            else:
                hist_hard = np.histogram(z_true[mask_k], grid_edges)[0].astype(np.float64)

            hist_hard = (hist_hard + smoothing) / np.maximum(
                hist_hard.sum() + smoothing * n_bins, 1e-12
            )

            if probs_oof is not None and probs_oof.shape[1] > k:
                if sample_weights is not None:
                    w_soft = sample_weights * probs_oof[:, k]
                else:
                    w_soft = probs_oof[:, k]
                hist_soft = np.histogram(z_true, grid_edges, weights=w_soft)[0].astype(np.float64)
                hist_soft = (hist_soft + smoothing) / np.maximum(
                    hist_soft.sum() + smoothing * n_bins, 1e-12
                )
                hist_k = hard_weight * hist_hard + (1.0 - hard_weight) * hist_soft
                hist_k /= hist_k.sum()
            else:
                hist_k = hist_hard / hist_hard.sum()

        calib_hists.append(hist_k)

    return np.array(calib_hists)
