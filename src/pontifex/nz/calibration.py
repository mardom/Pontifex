"""Pontifex Calibration: Empirical n(z) Histograms & Transfer Calibration."""

from typing import Optional
import numpy as np


def build_calibration_histograms(
    z_true: np.ndarray,
    y_pred: np.ndarray,
    n_tomo_bins: int,
    grid_edges: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    smoothing: float = 0.05,
) -> np.ndarray:
    """Construct smoothed empirical n_k(z) calibration histograms for each tomographic bin.
    
    Parameters
    ----------
    z_true : np.ndarray
        True/clean spectroscopic or many-band training redshifts.
    y_pred : np.ndarray
        Predicted tomographic bin index for each training galaxy.
    n_tomo_bins : int
        Number of tomographic bins.
    grid_edges : np.ndarray
        Redshift evaluation grid edges (e.g. 301 edges for 300 bins).
    sample_weights : Optional[np.ndarray]
        SOM transfer density ratio weights.
    smoothing : float
        Laplace smoothing term added to prevent log-loss divergence and zero-density artifacts.
        
    Returns
    -------
    calib_hists : np.ndarray
        Shape (n_tomo_bins, n_bins) array of normalized probability vectors.
    """
    calib_hists = []
    for k in range(n_tomo_bins):
        mask_k = (y_pred == k)
        if np.sum(mask_k) == 0:
            # Fallback to uniform if bin is empty
            n_bins = len(grid_edges) - 1
            hist_k = np.ones(n_bins, dtype=np.float64) / n_bins
        else:
            if sample_weights is not None:
                w_k = sample_weights[mask_k]
                hist_k = np.histogram(z_true[mask_k], grid_edges, weights=w_k)[0].astype(np.float64)
            else:
                hist_k = np.histogram(z_true[mask_k], grid_edges)[0].astype(np.float64)
                
            hist_k += smoothing
            hist_k /= hist_k.sum()
            
        calib_hists.append(hist_k)
        
    return np.array(calib_hists)
