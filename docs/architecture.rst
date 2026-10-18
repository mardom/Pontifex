Architecture Overview
=====================

**Pontifex** (v2.2.0 - AionPlus) is organized into three unified, decoupled subpackages designed to tackle both individual photo-z probability density estimation (PZ) and tomographic redshift distribution reconstruction (NZ) with strict out-of-fold generalization guarantees.

.. code-block:: text

   pontifex/
   ├── core/      # Shared transformations, physical FeatureGuard, and cosmological metrics
   ├── pz/        # Individual photo-z estimation & Out-of-Fold Mixture-of-Experts pipeline
   └── nz/        # Tomographic binning, SOM density transfer, OOF calibration, and GP realizations

Core Module (``pontifex.core``)
--------------------------------

The ``pontifex.core`` subpackage provides shared primitives, feature transformations, resilient input validation, and metric evaluation used across both photometric redshift pipelines:

* **``pontifex.core.features``**:
  
  * ``mag_to_flux``: Converts astronomical AB magnitudes to Pogson linear fluxes with robust handling of non-detections and negative/infinite values.
  * ``flux_to_mag``: Inverts fluxes to magnitudes with sub-floor clipping.
  * ``extract_features``: Extracts joint multi-band magnitudes (imputing non-detections at physical :math:`1\sigma` limiting magnitudes), :math:`1.0\,\text{mag}` upper-limit uncertainties, explicit binary non-detection masks (`nan_mask`), Pogson fluxes, and consecutive/cross-survey color indices (Rubin LSST :math:`u, g, r, i, z, y` + Roman Space Telescope :math:`Y, J, H`).
  * ``get_bands_and_ref``: Auto-detects survey configuration and designates reference bandpasses (:math:`i`-band for Rubin or combined Rubin+Roman, :math:`J`-band for Roman-only).

* **``pontifex.core.guard``**:

  * ``sanitize_input_catalog``: Protects downstream estimators by intercepting non-detected or corrupted records (NaNs, Infs, sentinel :math:`99.0` values, negative measurement errors, and extreme numerical outliers > 10 IQR) and mapping non-detected magnitudes to their band-specific :math:`1\sigma` limiting magnitudes (`BAND_MAG_LIMITS`) with :math:`1.0\,\text{mag}` uncertainty (:math:`\text{SNR} \approx 1`).

* **``pontifex.core.metrics``**:

  * ``compute_distribution_moments``: Computes distribution mean redshift :math:`\mu` and dispersion width :math:`\sigma`.
  * ``compute_moments_bias``: Evaluates DESC SRD Stage IV moment biases :math:`\delta\mu = (\mu_{\text{est}} - \mu_{\text{true}}) / (1 + \mu_{\text{true}})` and :math:`\delta\sigma = (\sigma_{\text{est}} - \sigma_{\text{true}}) / (1 + \mu_{\text{true}})`.
  * ``compute_photoz_point_metrics``: Standard Rubin photo-z metrics (:math:`\text{bias}`, :math:`\sigma_{\text{MAD}}`, outlier fraction).

* **``pontifex.core.constants``**:

  * Challenge tomographic bin edges, evaluation grids (301 edges / 300 bins), and physical parameter limits.

Photo-z Engine (``pontifex.pz``)
---------------------------------

The ``pontifex.pz`` subpackage implements the **Out-of-Fold Gated Ensemble & Footprint-Corrected Photometric Redshift Pipeline**:

* **``pontifex.pz.estimators``**:
  
  * ``CommitteeOfExperts``: Splits training catalogs into an 80% base-expert fitting set (`fit_dict`) and a disjoint 20% calibration set (`cal_dict`), applies faint photometric dimming (`sim_faint`) and flux-space `C-Mixup` when bright spectroscopic selection is detected, and combines diverse predictive algorithms via out-of-fold precision-weighted local kNN gating (:math:`e_{j,k}^{-3}` on joint point + CRPS error with self-neighbor exclusion):
    
    * Deep Neural Networks (MLP-1, MLP-2, and AION-1 Multimodal Foundation Embeddings)
    * Self-Organizing Maps (MiniSom)
    * Template fitting (LePhare, BPZ-lite)
    * Non-parametric algorithms (kNN with Optunity PSO tuning)
    * Conditional density estimators (FlexZBoost, PZFlow Normalizing Flows, GPz)

* **``pontifex.pz.aion_pz``**:

  * Implements AION-1 foundation model tokenization, damped monotone empirical PIT inverse-CDF recalibration (`fit_pit_recalibration`), and continuous sub-grid parabolic mode extraction (`compute_subgrid_zmode`) from the uncalibrated sharpest peak.

* **``pontifex.pz.em``**:

  * ``PontifexEM``: Performs selective Expectation-Maximization spatial clustering recalibration on 3D PCA Mahalanobis outlier / blend candidates (:math:`D_M^2 > 11.34`) with :math:`0.85/0.15` prior damping.

* **``pontifex.pz.pipeline``**:

  * ``train_and_estimate`` & ``estimate_only``: High-level end-to-end training and test inference pipelines conforming to the LSST-DESC PZ Data Challenge submission specifications.

Tomographic Ensemble Engine (``pontifex.nz``)
----------------------------------------------

The ``pontifex.nz`` subpackage delivers optimal tomographic bin assignment, empirical density transfer calibration, and posterior spatial realizations:

* **``pontifex.nz.som``**:

  * ``compute_som_density_weights``: Calculates unsupervised Self-Organizing Map density ratios :math:`w(c) = P_{\text{WFD}}(c) / P_{\text{DDF}}(c)` to correct spectroscopic selection biases and faint-end completeness drops.

* **``pontifex.nz.classifier``**:

  * ``train_tomographic_classifier``: Fast GPU/CPU auto-detecting XGBoost decision trees fitted with SOM sample weights.
  * ``predict_tomographic_bins``: Predicts tomographic bins and enforces Hybrid Mixture-of-Experts entropy regularization for boundary galaxies (:math:`H(p) > 1.25`).

* **``pontifex.nz.calibration``**:

  * ``build_calibration_histograms``: Reconstructs calibrated :math:`n_k(z)` distributions strictly from 5-fold out-of-fold predictions by blending SOM-weighted hard-assigned and soft-posterior-weighted histograms (:math:`0.90\,h_k^{\text{hard}} + 0.10\,h_k^{\text{soft}}`).

* **``pontifex.nz.sampler``**:

  * ``generate_correlated_realizations``: Generates 100 correlated Gaussian Process spatial realizations per bin (:math:`\ell_z = 0.15`) to accurately model cosmic sample variance across :math:`20,000\text{ deg}^2`.
  * ``create_qp_samples_ensemble``: Packages realization curves into standardized ``qp.hist`` Ensembles.

* **``pontifex.nz.runner``**:

  * Standardized challenge interfaces for Taskset 1 (representative), Taskset 2 (non-representative), and Taskset 3 (realization ensembles) with WFD target-sample posterior transition adaptation.
