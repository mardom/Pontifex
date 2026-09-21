"""Pontifex Metrics: Photo-z and Tomographic Moments Validation."""

from typing import Dict, Tuple
import numpy as np


def compute_distribution_moments(z_grid: np.ndarray, nz: np.ndarray) -> Tuple[float, float]:
    """Calculate mean redshift mu and width dispersion sigma from a normalized n(z) histogram.
    
    Parameters
    ----------
    z_grid : np.ndarray
        Bin centers or evaluation points.
    nz : np.ndarray
        Redshift probability density or normalized counts.
        
    Returns
    -------
    mu : float
        Mean redshift.
    sigma : float
        Redshift dispersion (standard deviation).
    """
    _trapz = getattr(np, "trapezoid", np.trapz)
    norm = _trapz(nz, z_grid) if len(z_grid) == len(nz) else np.sum(nz)
    if norm <= 0:
        return 0.0, 0.0
    p = nz / norm
    mu = np.sum(z_grid * p) if len(z_grid) != len(nz) else _trapz(z_grid * p, z_grid)
    var = np.sum((z_grid - mu) ** 2 * p) if len(z_grid) != len(nz) else _trapz((z_grid - mu) ** 2 * p, z_grid)
    sigma = np.sqrt(max(var, 0.0))
    return float(mu), float(sigma)


def compute_moments_bias(
    z_grid: np.ndarray,
    nz_est: np.ndarray,
    nz_true: np.ndarray
) -> Dict[str, float]:
    """Calculate DESC SRD Stage IV moment biases delta_mu and delta_sigma.
    
    delta_mu = (mu_est - mu_true) / (1 + mu_true)
    delta_sigma = (sigma_est - sigma_true) / (1 + mu_true)
    """
    mu_est, sig_est = compute_distribution_moments(z_grid, nz_est)
    mu_true, sig_true = compute_distribution_moments(z_grid, nz_true)
    
    denom = 1.0 + mu_true
    delta_mu = (mu_est - mu_true) / denom
    delta_sigma = (sig_est - sig_true) / denom
    
    return {
        "mu_est": mu_est,
        "mu_true": mu_true,
        "sigma_est": sig_est,
        "sigma_true": sig_true,
        "delta_mu": float(delta_mu),
        "delta_sigma": float(delta_sigma),
    }


def compute_photoz_point_metrics(z_phot: np.ndarray, z_spec: np.ndarray) -> Dict[str, float]:
    """Compute standard Rubin photo-z metrics: bias, sigma_MAD, and outlier fraction.
    
    Outliers are defined as |delta_z| / (1 + z_spec) > 0.15.
    """
    valid = np.isfinite(z_phot) & np.isfinite(z_spec)
    zp = z_phot[valid]
    zs = z_spec[valid]
    
    dz = (zp - zs) / (1.0 + zs)
    bias = float(np.median(dz))
    mad = float(1.4826 * np.median(np.abs(dz - bias)))
    outliers = float(np.mean(np.abs(dz) > 0.15))
    
    return {
        "bias": bias,
        "sigma_mad": mad,
        "outlier_rate": outliers,
        "n_samples": int(len(zp)),
    }
