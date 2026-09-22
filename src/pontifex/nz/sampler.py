"""Pontifex Spatial Sampler: Correlated Gaussian Process Realizations."""

from typing import Tuple
import numpy as np
from scipy.spatial.distance import cdist

try:
    import qp
except ImportError:
    qp = None


def generate_correlated_realizations(
    calib_hists: np.ndarray,
    grid_edges: np.ndarray,
    n_realizations: int = 100,
    length_scale: float = 0.15,
    gp_amplitude: float = 0.04,
    alpha_scale: float = 1000.0,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate correlated Gaussian Process spatial realizations for tomographic bins.
    
    Parameters
    ----------
    calib_hists : np.ndarray
        Shape (n_tomo_bins, n_bins) array of normalized calibration histograms.
    grid_edges : np.ndarray
        Redshift grid edges.
    n_realizations : int
        Number of realizations per tomographic bin (standard challenge is 100).
    length_scale : float
        RBF correlation length across redshift bins (default 0.15).
    gp_amplitude : float
        Variance scale of spatial mode modulation.
    alpha_scale : float
        Dirichlet concentration parameter scaling.
    random_seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    realization_matrix : np.ndarray
        Shape (n_tomo_bins * n_realizations, n_bins) of normalized realizations.
    bin_idx : np.ndarray
        Tomographic bin index associated with each realization row.
    i_real : np.ndarray
        Realization index (0 to n_realizations-1) for each realization row.
    """
    n_tomo_bins = len(calib_hists)
    n_grid = len(grid_edges) - 1
    z_mid = 0.5 * (grid_edges[:-1] + grid_edges[1:])

    # Spatial correlation RBF kernel across redshift bins
    dist_mat = cdist(z_mid[:, None], z_mid[:, None])
    cov_mat = gp_amplitude * np.exp(-0.5 * (dist_mat / length_scale) ** 2) + 1e-6 * np.eye(n_grid)
    gp_chol = np.linalg.cholesky(cov_mat)

    rng = np.random.default_rng(random_seed)
    realization_list = []

    for k in range(n_tomo_bins):
        alpha = calib_hists[k] * alpha_scale + 0.1
        dirichlet_draws = rng.dirichlet(alpha, size=n_realizations)

        # Correlated GP mode modulation
        gp_noise = rng.standard_normal(size=(n_realizations, n_grid))
        gp_modes = gp_noise @ gp_chol.T
        correlated_draws = dirichlet_draws * np.exp(gp_modes)
        correlated_draws /= np.sum(correlated_draws, axis=1, keepdims=True)

        for r in range(n_realizations):
            realization_list.append(correlated_draws[r])

    realization_matrix = np.array(realization_list)
    bin_idx = np.repeat(np.arange(n_tomo_bins), n_realizations)
    i_real = np.tile(np.arange(n_realizations), n_tomo_bins)

    return realization_matrix, bin_idx, i_real


def create_qp_samples_ensemble(
    calib_hists: np.ndarray,
    grid_edges: np.ndarray,
    n_realizations: int = 100,
    random_seed: int = 42,
):
    """Create a qp.hist Ensemble containing correlated realizations with metadata."""
    if qp is None:
        raise ImportError("qp-prob is required to create qp Ensembles.")
        
    realization_matrix, bin_idx, i_real = generate_correlated_realizations(
        calib_hists=calib_hists,
        grid_edges=grid_edges,
        n_realizations=n_realizations,
        random_seed=random_seed,
    )
    
    ens_samples = qp.hist.create_ensemble(grid_edges, realization_matrix)
    ens_samples.set_ancil(dict(bin_idx=bin_idx, i_realization=i_real))
    return ens_samples

