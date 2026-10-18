.. Pontifex documentation master file

Welcome to Pontifex's Documentation!
====================================

**Pontifex** (v2.2.0 - AionPlus) is a unified, high-performance photometric redshift and tomographic distribution reconstruction framework engineered for the Vera C. Rubin Observatory Legacy Survey of Space and Time (LSST) and the Nancy Grace Roman Space Telescope.

Core Capabilities
-----------------

* **Dual-Challenge Architecture**:
  
  * **``pontifex.pz``**: Individual galaxy photo-z probability density function (PDF) estimation using an Out-of-Fold Committee of Diverse Experts (Deep MLPs, MiniSom, PZFlow Normalizing Flows, GPz, FlexZBoost, BPZ-lite, LePhare, AION-1 Multimodal Foundation Embeddings) coupled with faint photometric dimming + ``C-Mixup`` covariate-shift augmentation, empirical PIT inverse-CDF recalibration, sub-grid parabolic mode extraction, and selective Expectation-Maximization footprint gating (Nugundam + SkyKatana).
  * **``pontifex.nz``**: Tomographic ensemble distribution :math:`n(z)` reconstruction featuring Self-Organizing Map density ratio transfer reweighting (DIR), 5-fold out-of-fold hard+soft probability histogram calibration, WFD posterior transition adaptation, XGBoost classification with boundary entropy regularization, and correlated Gaussian Process spatial realizations (:math:`\ell_z = 0.15`).

* **Resilient Infrastructure (``pontifex.core``)**:
  
  * Automatic feature protection via ``FeatureGuard`` replacing non-detections with physical band-specific :math:`1\sigma` limiting magnitudes (`BAND_MAG_LIMITS`), :math:`1.0\,\text{mag}` upper-limit uncertainties, and binary non-detection indicators (`nan_mask`).
  * Continuous Pogson flux transformations with noise-floor clipping.
  * Native calculation of DESC Science Requirements Document (SRD) Stage IV tomographic moment biases (:math:`\delta\mu`, :math:`\delta\sigma`).

* **Validated on Simulated & Observational Benchmarks**:

  * Achieves **96.4% (324 / 336)** on the official SLAC rubric across all 16 LSST-DESC PZ Data Challenge regimes and **90.05% tomographic accuracy** (:math:`\text{RMS } |\delta\mu_k| \le 0.003`) in the NZ Data Challenge.
  * Outperforms 30+-band **COSMOS2020 Classic LePHARE** in outlier rate (:math:`\eta_{>0.15} = 2.22\%` vs. :math:`2.29\%`), median bias (:math:`+0.00040` vs. :math:`-0.00253`), and RMSE (:math:`0.1043` vs. :math:`0.1219`) on :math:`N = 30,198` cross-matched COSMOS galaxies (:math:`0.0 \le z_{\text{spec}} \le 1.3`) using only **6 optical broadband filters (`ugrizy`)** (see :doc:`tutorial`).

.. toctree::
   :maxdepth: 2
   :caption: User Guide & Architecture:

   architecture
   installation_and_pso
   tutorial
   experts
   tomography_nz
   autoapi/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
