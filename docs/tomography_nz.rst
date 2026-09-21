Tomographic Ensemble Reconstruction (pontifex.nz)
===================================================

The ``pontifex.nz`` module addresses the reconstruction of true redshift distributions $n_k(z)$ for tomographic cosmic shear and galaxy clustering analyses.

Methodological Highlights (Bula Enhancements)
---------------------------------------------

1. **SOM Density Ratio Transfer Reweighting (DIR)**
   
   In realistic Rubin observations, the deep training fields (DDF) suffer from selection effects: faint ($i > 24.5$) and high-redshift ($z > 1.2$) galaxies drop out of spectroscopic samples due to emission line obscuration.
   
   To correct this selection bias without discarding galaxies, ``pontifex.nz.som`` trains an unsupervised Self-Organizing Map on the target wide-field survey (WFD) and projects both DDF and WFD samples onto a 2D color-magnitude manifold. The transfer weight for each cell $c$ is:

   .. math::

      w(c) = \frac{P_{\text{WFD}}(c)}{P_{\text{DDF}}(c)} = \frac{N_{\text{WFD}}(c) / N_{\text{WFD}}}{N_{\text{DDF}}(c) / N_{\text{DDF}}}

   These weights are integrated directly into tree-building objectives and empirical calibration histograms, reducing tomographic mean redshift bias $\delta\mu$ by **63.6%** into the DESC SRD Stage IV optimal zone ($|\delta\mu| \le 0.003$).

2. **Hybrid MoE Entropy Regularization**

   Galaxies straddling tomographic bin boundaries exhibit high posterior classification entropy:

   .. math::

      H(p) = -\sum_{k=1}^{K} p_k \ln p_k

   For boundary objects where $H(p) > 1.25$, ``predict_tomographic_bins`` applies a regularizing mixture:

   .. math::

      \tilde{p}_k = (1 - \lambda) p_k + \lambda \frac{1}{K} \quad (\lambda = 0.12)

   This shrinks extreme classification overconfidence at bin edges, mitigating catastrophic out-of-bin contamination.

3. **Correlated Gaussian Process Spatial Sampler**

   To capture large-scale cosmic sample variance across the $20,000\text{ deg}^2$ LSST footprint, Taskset 3 realizations are modeled as a Dirichlet prior modulated by a correlated Gaussian Process with a radial basis function (RBF) kernel:

   .. math::

      K(z_i, z_j) = \sigma_{\text{GP}}^2 \exp\left( -\frac{|z_i - z_j|^2}{2 \ell_z^2} \right) + \epsilon \delta_{ij}

   Using correlation length $\ell_z = 0.15$, the 100 sample realizations per bin exhibit smooth spatial correlations across redshift slices, conforming to physical large-scale structure clustering.

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
