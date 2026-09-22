Architecture Overview
=====================

**Pontifex** (v2.0.0) is organized into three unified, decoupled subpackages designed to tackle both individual photo-z probability density estimation (PZ) and tomographic redshift distribution reconstruction (NZ).

.. code-block:: text

   pontifex/
   ├── core/      # Shared transformations, feature guards, and cosmological metrics
   ├── pz/        # Individual photo-z estimation & Mixture-of-Experts pipeline
   └── nz/        # Tomographic binning, SOM density transfer, and correlated realizations

Core Module (``pontifex.core``)
--------------------------------

The ``pontifex.core`` subpackage provides shared primitives, feature transformations, resilient input validation, and metric evaluation used across both photometric redshift pipelines:

* **``pontifex.core.features``**:
  
  * ``mag_to_flux``: Converts astronomical AB magnitudes to Pogson linear fluxes with robust handling of non-detections and negative/infinite values.
  * ``flux_to_mag``: Inverts fluxes to magnitudes with sub-floor clipping.
  * ``extract_features``: Extracts joint multi-band magnitudes, error vectors, Pogson fluxes, and consecutive/cross-survey color indices (Rubin LSST $u, g, r, i, z, y$ + Roman Space Telescope $Y, J, H$).
  * ``get_bands_and_ref``: Auto-detects survey configuration and designates reference bandpasses ($i$-band for Rubin-only, $H$-band for Rubin+Roman).

* **``pontifex.core.guard``**:

  * ``sanitize_input_catalog``: Protects downstream estimators by intercepting and repairing corrupted inputs (NaNs, Infs, negative measurement errors, unphysical magnitudes, and extreme numerical outliers > 10 IQR). Emits structured diagnostics and summary reports.

* **``pontifex.core.metrics``**:

  * ``compute_distribution_moments``: Computes distribution mean redshift $\mu$ and dispersion width $\sigma$.
  * ``compute_moments_bias``: Evaluates DESC SRD Stage IV moment biases $\delta\mu = (\mu_{\text{est}} - \mu_{\text{true}}) / (1 + \mu_{\text{true}})$ and $\delta\sigma = (\sigma_{\text{est}} - \sigma_{\text{true}}) / (1 + \mu_{\text{true}})$.
  * ``compute_photoz_point_metrics``: Standard Rubin photo-z metrics ($\text{bias}$, $\sigma_{\text{MAD}}$, outlier fraction).

* **``pontifex.core.constants``**:

  * Challenge tomographic bin edges, evaluation grids (301 edges / 300 bins), and physical parameter limits.

Photo-z Engine (``pontifex.pz``)
---------------------------------

The ``pontifex.pz`` subpackage implements the **Dual-Path Gated Ensemble & Footprint-Corrected Photometric Redshift Pipeline**:

* **``pontifex.pz.estimators``**:
  
  * ``CommitteeOfExperts``: Gathers diverse predictive algorithms:
    
    * Deep Neural Networks (MLP with latent skip-connections)
    * Self-Organizing Maps (MiniSom)
    * Template fitting (LePhare, BPZ-lite)
    * Non-parametric algorithms (kNN with Optunity PSO tuning)
    * Density estimators (FlexZBoost, PZFlow Normalizing Flows, GPz)

* **``pontifex.pz.em``**:

  * ``PontifexEM``: Performs dynamic Expectation-Maximization spatial clustering recalibration using Nugundam 3D cross-correlations and SkyKatana boolean footprint masking.

* **``pontifex.pz.pipeline``**:

  * ``train_and_estimate``: High-level end-to-end training and test inference pipeline conforming to the LSST-DESC PZ Data Challenge submission specifications.

Tomographic Ensemble Engine (``pontifex.nz``)
----------------------------------------------

The ``pontifex.nz`` subpackage delivers optimal tomographic bin assignment, empirical density transfer calibration, and posterior spatial realizations:

* **``pontifex.nz.som``**:

  * ``compute_som_density_weights``: Calculates unsupervised Self-Organizing Map density ratios $w(c) = P_{\text{WFD}}(c) / P_{\text{DDF}}(c)$ to correct spectroscopic selection biases and faint-end completeness drops.

* **``pontifex.nz.classifier``**:

  * ``train_tomographic_classifier``: Fast GPU-accelerated XGBoost decision trees fitted with sample weights.
  * ``predict_tomographic_bins``: Predicts tomographic bins and enforces Hybrid Mixture-of-Experts entropy regularization for boundary galaxies ($H(p) > 1.25$).

* **``pontifex.nz.calibration``**:

  * ``build_calibration_histograms``: Reconstructs calibrated $n_k(z)$ distributions with Laplace smoothing.

* **``pontifex.nz.sampler``**:

  * ``generate_correlated_realizations``: Generates 100 correlated Gaussian Process spatial realizations per bin ($\ell_z = 0.15$) to accurately model cosmic sample variance across $20,000\text{ deg}^2$.
  * ``create_qp_samples_ensemble``: Packages realization curves into standardized `qp.hist` Ensembles.

* **``pontifex.nz.runner``**:

  * Standardized challenge interfaces for Taskset 1 (representative), Taskset 2 (non-representative), and Taskset 3 (realization ensembles).

