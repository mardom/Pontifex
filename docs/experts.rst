The Committee of 10 Experts
===========================

**Pontifex** combines 10 distinct photometric redshift estimators into an adaptive, local feature-space ensemble using K-Nearest Neighbors (KNN) gating. This page provides a detailed reference for all 10 expert algorithms, their default calibration hyperparameter configurations, and their performance dominance across different data challenge task sets and galaxy populations.

Overview of Experts
-------------------

The committee integrates template-fitting, empirical regression, kernel density estimation, neural networks, normalizing flows, and topological clustering algorithms:

1. **BPZ (Bayesian Photometric Redshifts)**:
   * **Methodology**: Bayesian template-fitting using CWW/HDFN SED templates with luminosity function priors.
   * **Dominance**: Dominates in non-representative samples (**Taskset 3**) and high-redshift regimes ($z > 2.0$) where empirical training coverage is sparse.
   * **Default Calibration Parameters**: ``dz=0.01``, ``zmin=0.0``, ``zmax=3.0``, ``prior_name='hdfn_gen'``.

2. **LePhare**:
   * **Methodology**: Template optimization code computing synthetic magnitudes across COSMOS SED templates with extinction laws and emission lines.
   * **Dominance**: Dominates in faint, high-extinction galactic lines-of-sight and deep magnitude limit datasets (e.g., Roman High-z Taskset).
   * **Default Calibration Parameters**: ``GAL_SED='COSMOS_MOD.list'``, ``EXTINC_LAW='SMC_prevot.dat, SB_calzetti.dat'``, ``EB_V=[0.0, 0.5]``, ``Z_STEP=0.01``.

3. **FlexZBoost**:
   * **Methodology**: Non-parametric conditional density estimation utilizing gradient-boosted decision trees with B-spline basis expansion.
   * **Dominance**: Dominates in representative training catalogs (**Tasksets 1 & 2**) for bright, isolated galaxy populations ($i < 24.5$).
   * **Default Calibration Parameters**: ``max_depth=6``, ``nbump=35``, ``nsharp=15``, ``bump_thresh=0.02``, ``sharpen_alpha=0.7``.

4. **GPz (Gaussian Process Redshifts)**:
   * **Methodology**: Sparse Gaussian Process regression incorporating heteroscedastic, input-dependent photometric measurement noise.
   * **Dominance**: Dominates in low-SNR, noisy photometry regimes by dynamically incorporating flux uncertainty ($\sigma_m$).
   * **Default Calibration Parameters**: ``n_basis=50``, ``max_iter=100``, heteroscedastic variance weighting.

5. **PZFlow**:
   * **Methodology**: Differentiable Normalizing Flow architecture modeling joint continuous color-redshift probability density distributions.
   * **Dominance**: Dominates in blended and multi-component target catalogs (**Taskset 4**), effectively resolving multi-modal $p(z)$ distributions.
   * **Default Calibration Parameters**: ``bijector_layers=4``, ``hidden_units=32``, ``epochs=50``, ``batch_size=256``.

6. **NN1 (AION-MLP-1)**:
   * **Methodology**: Shallow Multi-Layer Perceptron trained on magnitude-color feature spaces.
   * **Dominance**: Provides fast, smooth baseline predictions with low variance for main-sequence galaxies ($0.2 < z < 1.2$).
   * **Default Calibration Parameters**: ``hidden_layer_sizes=(64, 32)``, ``activation='relu'``, ``max_iter=200``, ``alpha=1e-4``.

7. **NN2 (AION-MLP-2)**:
   * **Methodology**: Deep Multi-Layer Perceptron incorporating L2 regularization and dropout.
   * **Dominance**: Captures non-linear magnitude-color interactions in broad-band optical-NIR survey combinations (e.g., Rubin + Roman).
   * **Default Calibration Parameters**: ``hidden_layer_sizes=(128, 64, 32)``, ``max_iter=300``, ``alpha=1e-3``.

8. **miniSOM (Self-Organizing Map)**:
   * **Methodology**: Unsupervised topological SOM grid mapping galaxies to discrete color nodes with Gaussian kernel smoothing.
   * **Dominance**: Dominates in identifying anomalous feature space regions (e.g., color non-detections and blend contamination).
   * **Default Calibration Parameters**: ``x=12``, ``y=12``, ``sigma=1.5``, ``learning_rate=0.5``, ``topology='hexagonal'``.

9. **KNN (K-Nearest Neighbors KDE)**:
   * **Methodology**: Distance-weighted local kernel density estimator in standardized AION PCA latent space ($D_{\text{PCA}} = N_{95\%}$) with Mahalanobis eigenvalue weighting ($\lambda_d^{-1/4}$) and photometric noise floor masking ($S/N \ge 2.0$).
   * **Dominance**: Dominates in densely populated, highly representative feature space regions across all challenge catalogs.
   * **Default Calibration Parameters**: Optunity PSO tuned defaults: $k=17, \text{bw}=0.873$ (TS1), $k=17, \text{bw}=0.470$ (TS2), $k=35, \text{bw}=0.546$ (TS3), $k=29, \text{bw}=1.109$ (TS4).

10. **AION-Prior (AION Deep Mixture Prior)**:
    * **Methodology**: Hybrid neural estimator combining photometric color priors with spectroscopic redshift calibration.
    * **Dominance**: Dominates in mitigating catastrophic outliers by enforcing physical redshift boundary constraints.
    * **Default Calibration Parameters**: ``alpha=5e-4``, ``learning_rate_init=5e-3``, ``batch_size=128``.

---

Expert Dominance Summary Matrix
-------------------------------

.. list-table:: Performance Dominance Across DESC Challenge Tasksets
   :widths: 20 25 30 25
   :header-rows: 1

   * - Taskset / Regime
     - Primary Dominant Expert
     - Secondary Dominant Expert
     - Key Advantage
   * - **Taskset 1 (Representative Rubin)**
     - ``FlexZBoost``
     - ``GPz``
     - Minimal bias ($\sigma_{\rm MAD} < 0.015$)
   * - **Taskset 2 (Representative Roman)**
     - ``NN2 (AION-MLP-2)``
     - ``PZFlow``
     - Broad-band optical-NIR color coverage
   * - **Taskset 3 (Non-Representative)**
     - ``BPZ``
     - ``LePhare``
     - Extrapolation safety via SED templates
   * - **Taskset 4 (Blended Sources)**
     - ``PZFlow``
     - ``miniSOM``
     - Multi-modal PDF recovery rate (84.3%)
   * - **Low-SNR / High-Noise**
     - ``GPz``
     - ``KNN``
     - Heteroscedastic noise propagation
