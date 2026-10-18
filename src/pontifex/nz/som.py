"""Pontifex Self-Organizing Map (SOM) Density Ratio Transfer Reweighting."""

import logging
from typing import Tuple
import numpy as np

try:
    from minisom import MiniSom
except ImportError:
    MiniSom = None

logger = logging.getLogger("pontifex.nz.som")


def compute_som_density_weights(
    X_ddf: np.ndarray,
    X_wfd: np.ndarray,
    grid_size: Tuple[int, int] = (16, 16),
    sigma: float = 1.5,
    learning_rate: float = 0.5,
    num_iteration: int = 2000,
    random_seed: int = 42,
) -> np.ndarray:
    """Compute transfer density ratio weights w(c) = P_WFD(c) / P_DDF(c) using MiniSom.
    
    Parameters
    ----------
    X_ddf : np.ndarray
        Feature matrix of deep field (DDF) spectroscopic training catalog.
    X_wfd : np.ndarray
        Feature matrix of wide-fast-deep (WFD) target survey catalog.
    grid_size : Tuple[int, int]
        SOM grid dimensions (default 16x16 = 256 cells).
    sigma : float
        Spread of neighborhood function.
    learning_rate : float
        Initial learning rate.
    num_iteration : int
        Number of training iterations.
    random_seed : int
        Seed for reproducibility.
        
    Returns
    -------
    weights : np.ndarray
        Density ratio weight for each galaxy in X_ddf, normalized so mean(w) = 1.0.
    """
    if MiniSom is None:
        logger.warning("MiniSom not installed. Falling back to uniform sample weights.")
        return np.ones(len(X_ddf), dtype=np.float32)

    # Use first 6 magnitude columns (or min dimension) for SOM spatial manifold
    n_dim = min(6, X_ddf.shape[1])
    feat_ddf = X_ddf[:, :n_dim].astype(np.float64)
    feat_wfd = X_wfd[:, :n_dim].astype(np.float64)

    # Normalize across joint dataset
    all_feat = np.vstack([feat_ddf, feat_wfd[: min(50000, len(feat_wfd))]])
    mean_val = np.nanmean(all_feat, axis=0)
    std_val = np.nanstd(all_feat, axis=0)
    std_val[std_val < 1e-6] = 1.0

    norm_ddf = np.clip((feat_ddf - mean_val) / std_val, -5.0, 5.0)
    norm_wfd = np.clip((feat_wfd - mean_val) / std_val, -5.0, 5.0)

    som = MiniSom(
        grid_size[0],
        grid_size[1],
        n_dim,
        sigma=sigma,
        learning_rate=learning_rate,
        random_seed=random_seed,
    )
    # Train SOM on representative subsample of WFD target space
    sub_wfd = norm_wfd[np.random.default_rng(random_seed).choice(len(norm_wfd), min(25000, len(norm_wfd)), replace=False)]
    som.train_batch(sub_wfd, num_iteration=num_iteration)

    # Map DDF and WFD samples to Best Matching Units (BMUs)
    ddf_bmus = [som.winner(x) for x in norm_ddf]
    wfd_bmus = [som.winner(x) for x in norm_wfd[: min(50000, len(norm_wfd))]]

    # Compute cell occupancy frequencies
    counts_ddf = np.zeros(grid_size, dtype=np.float64)
    for r, c in ddf_bmus:
        counts_ddf[r, c] += 1.0

    counts_wfd = np.zeros(grid_size, dtype=np.float64)
    for r, c in wfd_bmus:
        counts_wfd[r, c] += 1.0

    p_ddf = (counts_ddf + 1.0) / (np.sum(counts_ddf) + counts_ddf.size)
    p_wfd = (counts_wfd + 1.0) / (np.sum(counts_wfd) + counts_wfd.size)
    density_ratio = p_wfd / p_ddf

    # Assign weight to each DDF object based on its BMU cell
    weights = np.array([density_ratio[r, c] for r, c in ddf_bmus], dtype=np.float32)
    # Clip extreme outlier weights and re-normalize mean to 1.0
    weights = np.clip(weights, 0.05, 20.0)
    weights /= np.mean(weights)
    return weights

