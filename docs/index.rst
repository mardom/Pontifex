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
