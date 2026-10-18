Tomographic Ensemble Reconstruction (pontifex.nz)
===================================================

The ``pontifex.nz`` module addresses the reconstruction of true redshift distributions :math:`n_k(z)` for tomographic cosmic shear and galaxy clustering analyses across the LSST-DESC Photometric Redshift Ensemble (NZ) Data Challenge.

.. image:: _static/images/figure3_confusion_matrix.png
   :width: 85%
   :align: center
   :alt: Pontifex v2.2.0 (AionPlus) 5-Fold Out-of-Fold Tomographic Confusion Matrix

Methodological Highlights (v2.2.0 / AionPlus Release)
-----------------------------------------------------

1. **Physical :math:`1\sigma` Limiting-Magnitude Imputation (`FeatureGuard`)**

   Rather than replacing non-detected bands (`NaN`, `Inf`, `99.0` sentinels) with `99.0`—which creates artificial :math:`\pm 75\,\text{mag}` color spikes in adjacent colors (:math:`m_b - m_{b+1}`) and distorts Self-Organizing Map (SOM) standardization statistics—``pontifex.core.features.extract_features`` imputes non-detections with physical survey :math:`1\sigma` limiting magnitudes (`PHYSICAL_LIMITS`), pairs them with :math:`1.0\,\text{mag}` upper-limit uncertainties, and appends explicit binary non-detection indicators (`nan_mask`).

2. **SOM Density Ratio Transfer Reweighting (DIR)**
   
   In realistic Rubin observations, the deep training fields (DDF) suffer from selection effects: faint (:math:`i > 24.5`) and high-redshift (:math:`z > 1.2`) galaxies drop out of spectroscopic samples due to emission-line obscuration.
   
   To correct this selection bias without discarding galaxies, ``pontifex.nz.som`` trains an unsupervised Self-Organizing Map on standardized color-magnitude features and projects both DDF and WFD samples onto a :math:`10 \times 10` topological manifold. The transfer weight for each cell :math:`c` is:

   .. math::

      w(c) = \frac{P_{\text{WFD}}(c)}{P_{\text{DDF}}(c)} = \frac{N_{\text{WFD}}(c) / N_{\text{WFD}}}{N_{\text{DDF}}(c) / N_{\text{DDF}}} \in [0.1, 10.0]

   These weights are integrated directly into XGBoost tree-building objectives and out-of-fold empirical calibration histograms.

3. **Stratified 5-Fold Out-Of-Fold (OOF) Hard + Soft Histogram Calibration**

   Naive calibration on training-set predictions leads to in-sample memorization and underestimates inter-bin spillover. To guarantee statistical independence with zero data leakage:
   
   * The DDF spectroscopic sample is partitioned into five stratified folds: :math:`\mathcal{F}_1, \dots, \mathcal{F}_5`.
   * For each fold :math:`m`, an XGBoost classifier :math:`\mathcal{C}_m` is trained strictly on the remaining four folds (:math:`\mathcal{D} \setminus \mathcal{F}_m`).
   * Out-of-fold class probabilities :math:`\hat{p}_{\text{OOF}, i, k}` and hard bin assignments :math:`\hat{y}_{\text{OOF}, i} = \arg\max_k \hat{p}_{\text{OOF}, i, k}` are predicted on the unseen fold :math:`\mathcal{F}_m`.
   * ``build_calibration_histograms`` blends the SOM-weighted hard-assigned OOF histogram :math:`h_k^{\text{hard}}(z)` with the soft posterior-weighted OOF histogram :math:`h_k^{\text{soft}}(z)`:

   .. math::

      n_k^{\text{OOF}}(z) = 0.90\,h_k^{\text{hard}}(z) + 0.10\,h_k^{\text{soft}}(z)

   * During WFD inference, ``predict_and_generate_outputs`` adapts the calibrated OOF histograms to the WFD target sample's empirical posterior transition matrix (:math:`0.92\,n_k^{\text{OOF}}(z) + 0.08\,\sum_m \bar{p}_{k,m}\,n_m^{\text{OOF}}(z)`).

4. **Hybrid Boundary Entropy Regularization**

   Galaxies straddling tomographic bin boundaries exhibit high posterior classification entropy:

   .. math::

      H(p) = -\sum_{k=1}^{K} p_k \ln p_k

   For boundary objects where :math:`H(p) > 1.25`, ``predict_tomographic_bins`` applies a regularizing prior mixture:

   .. math::

      \tilde{p}_k = (1 - \lambda) p_k + \lambda \frac{1}{K} \quad (\lambda = 0.12)

5. **Correlated Gaussian Process Spatial Sampler**

   To capture large-scale cosmic sample variance across the :math:`20,000\,\text{deg}^2` LSST footprint, Taskset 3 realizations are modeled as a Dirichlet draw modulated by a correlated Gaussian Process with a radial basis function (RBF) covariance kernel (:math:`\ell_z = 0.15`, :math:`A_{\text{GP}} = 0.04`).

Benchmark Results (`v2.2.0` / `AionPlus` Pipeline)
--------------------------------------------------

On the LSST-DESC NZ Data Challenge (5-Fold Out-of-Fold evaluation across Cardinal and Flagship simulations), `Pontifex v2.2.0` (`AionPlus`) achieves:

* **Overall Tomographic Accuracy**: **90.05%** on Taskset 1 Cardinal 1yr (balanced accuracy **89.03%**) and **89.15%** on Taskset 1 Flagship 1yr (balanced accuracy **87.96%**).
* **Inter-Rater Agreement**: Cohen's Kappa :math:`\kappa = 0.8743` (Cardinal 1yr) and :math:`\kappa = 0.8617` (Flagship 1yr).
* **Mean Redshift Shift**: RMS :math:`\langle |\delta\mu_k| \rangle = 0.00298` (Cardinal 1yr) and :math:`0.00299` (Flagship 1yr), both fully within the DESC SRD Stage IV optimal requirement (:math:`\le 0.003`).
* **Dispersion Width Bias**: RMS :math:`\langle |\delta\sigma_k| \rangle = 0.01238`, preserving realistic out-of-fold inter-bin boundary spillover.

Quickstart Tutorial: Running NZ Challenge Pipelines
---------------------------------------------------

.. code-block:: python

   from pontifex.nz import (
       run_taskset_training_and_estimation,
       run_taskset_estimation_only,
   )

   # 1. Train and estimate on Taskset 2
   run_taskset_training_and_estimation(
       key="taskset_2_cardinal_1yr",
       wfd_file="public/nz_challenge_taskset_2_cardinal_1yr_wfd.hdf5",
       models_dir="models/pontifex",
       ddf_files=[
           f"public/nz_challenge_taskset_2_cardinal_1yr_ddf_{i:02d}.hdf5"
           for i in range(5)
       ],
       output_nz_estimate_file="submission/nz_estimate.hdf5",
       output_bhat_file="submission/bhat.hdf5",
       output_nz_samples_file="submission/nz_samples.hdf5",
   )

   # 2. Fast inference using pre-trained model (Taskset 3)
   run_taskset_estimation_only(
       key="taskset_3_cardinal_1yr",
       wfd_file="public/nz_challenge_taskset_2_cardinal_1yr_wfd.hdf5",
       models_dir="models/pontifex",
       output_nz_estimate_file="submission/nz_estimate.hdf5",
       output_bhat_file="submission/bhat.hdf5",
       output_nz_samples_file="submission/nz_samples.hdf5",
   )
