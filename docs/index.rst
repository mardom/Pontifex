.. Pontifex documentation master file

Welcome to Pontifex's Documentation!
====================================

**Pontifex** (v2.1.0 - Ascention) is a unified, high-performance photometric redshift and tomographic distribution reconstruction framework engineered for Vera C. Rubin Observatory Legacy Survey of Space and Time (LSST) and the Nancy Grace Roman Space Telescope.

Core Capabilities
-----------------

* **Dual-Challenge Architecture**:
  
  * **``pontifex.pz``**: Individual galaxy photo-z probability density function (PDF) estimation using a Committee of Diverse Experts (Deep MLPs, MiniSom, PZFlow Normalizing Flows, GPz, FlexZBoost, BPZ-lite, LePhare) coupled with Expectation-Maximization footprint gating (Nugundam + SkyKatana).
  * **``pontifex.nz``**: Tomographic ensemble distribution :math:`n(z)` reconstruction featuring Self-Organizing Map density ratio transfer reweighting (DIR), XGBoost classification with boundary entropy regularization, and correlated Gaussian Process spatial realizations (:math:`\ell_z = 0.15`).

* **Resilient Infrastructure (``pontifex.core``)**:
  
  * Automatic feature protection against NaNs, infinite records, unphysical error limits, and photometric outliers.
  * Continuous Pogson flux transformations with noise-floor clipping.
  * Native calculation of DESC Science Requirements Document (SRD) Stage IV tomographic moment biases ($\delta\mu$, $\delta\sigma$).

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
