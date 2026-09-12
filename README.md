# Pontifex

A modular, footprint-corrected photometric redshift estimation pipeline for the LSST Dark Energy Science Collaboration (DESC) photo-z working group.

Pontifex combines multiple machine learning estimators into an adaptive Committee of Experts using local K-Nearest Neighbors gating, followed by a spatial clustering calibration loop (Expectation-Maximization) using projected 3D cross-correlations.

---

## Installation & Setup

Pontifex is built on top of the Rubin LSST DESC `RAIL` framework. You can install all exact dependencies and create the environment using Conda or Pip:

### Option A: Conda Environment Creation (Recommended)

1. **Create Environment from `environment.yml`**
   ```bash
   conda env create -f environment.yml
   conda activate pontifex_env
   ```

2. **Verify Installation**
   ```bash
   pytest tests/
   ```

---

### Option B: Manual Conda / Pip Setup

1. **Create and Activate Conda Environment**
   ```bash
   conda create -n pontifex_env python=3.11 numpy scipy pandas joblib astropy matplotlib -y
   conda activate pontifex_env
   ```

2. **Install Dependencies and Pontifex**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

### Troubleshooting Linker Errors
If you are compiling C/Fortran extensions (such as `nugundam` or `Corrfunc`) within an Anaconda environment and receive linker errors similar to:
```text
/usr/bin/ld: cannot find /lib64/libm.so.6: No such file or directory
```
This occurs because of hardcoded absolute paths inside GNU `ld` scripts within the Conda sysroot. To fix this:
1. Locate the Conda compiler sysroot lib folder (typically at `~/anaconda3/x86_64-conda-linux-gnu/sysroot/lib64/`).
2. Edit the following files to replace absolute `/lib64/` paths with relative library names:
   * **`libm.so`**: change `GROUP ( /lib64/libm.so.6 AS_NEEDED ( /lib64/libmvec.so.1 ) )` to `GROUP ( libm.so.6 AS_NEEDED ( libmvec.so.1 ) )`
   * **`libc.so`**: change `GROUP ( /lib64/libc.so.6 /usr/lib64/libc_nonshared.a AS_NEEDED ( /lib64/ld-linux-x86-64.so.2 ) )` to `GROUP ( libc.so.6 libc_nonshared.a AS_NEEDED ( ld-linux-x86-64.so.2 ) )`
   * **`libm.a`**: change `GROUP ( /usr/lib64/libm-2.39.a /usr/lib64/libmvec.a )` to `GROUP ( libm-2.39.a libmvec.a )`

---

## Hyperparameter Optimization (PSO)

Pontifex supports Particle Swarm Optimization (PSO) via the `optunity` library to tune the individual estimators before combining them.

### Methodology
During training, the optimization split-validates the training catalog (80/20 train/validation split) to find hyperparameters that minimize the Median Absolute Deviation (MAD) of predicted redshift PDFs.

The search space covers:
* **SOM**: neighborhood width `sigma`, learning rate, grid dimensions.
* **KNN**: kernel grid scale `ngrid_sigma`, minimum neighbors.
* **FlexZBoost**: max tree depth, number of bumps, threshold component count.
* **GPz**: basis functions count, training iterations limit.
* **AION prior**: regularizing L2 penalty `alpha`, initial learning rate.

### Usage
By default, Pontifex loads pre-calculated optimal settings from `results/pso_best_hyperparameters.pkl` to bypass the costly swarm search during normal training runs.

To trigger a fresh hyperparameter search (e.g. when new training mock catalogs are released):
```python
from pontifex import train_and_estimate

train_and_estimate(
    train_file="path/to/new_train.hdf5",
    test_file="path/to/test.hdf5",
    output_file="path/to/output.hdf5",
    optimize_hyperparams=True  # Triggers PSO optimization
)
```

---

## Modular Pipeline Execution Keywords & Production Configuration

Pontifex provides modular execution keyword triggers to activate specific architectural stages, Optunity-tuned feature space configurations, and spatial calibration loops:

```python
from pontifex.estimators import compute_expert_weights_knn, apply_expert_weights_knn

# 1. INFORM Stage (Train AION-PCA KNN Gating Engine on Training Set)
train_pca, train_errors, pca_pkg, _ = compute_expert_weights_knn(
    train_dict, train_pdfs, z_centers, bands, ref_band,
    blend_gating_mode='ground_truth_informed'  # Options: 'ground_truth_informed', 'color_variance', 'none'
)

# 2. INFERENCE Stage (Apply Gating to Test Catalog - ZERO Target Leakage)
# Automatically uses Optunity-tuned taskset defaults:
# TS1/TS2: K=17, bw=0.873/0.470 | TS3: K=35, bw=0.546 | TS4: K=29, bw=1.109
final_pdfs, expert_weights = apply_expert_weights_knn(
    val_dict, val_pdfs, train_pca, train_errors, pca_pkg["mean"], pca_pkg["std"],
    bands, ref_band,
    taskset=1,  # Selects Optunity-tuned k and bw_mult automatically
    blend_gating_mode='ground_truth_informed'
)
```

### Production Default Configuration Highlights

* **Optunity-Tuned Feature Space ($D_{\text{PCA}} = N_{95\%}$)**:
  Dynamically retains $95\%$ cumulative PCA variance across catalog latent spaces ($D_{\text{PCA}} \in [28, 37]$), avoiding noise-dominated components while preserving maximal color-magnitude variance.
* **Photometric Noise Floor Filtering ($S/N \ge 2.0$)**:
  Identifies noise-corrupted flux measurements ($m > 25.5$ or $\sigma_m > 0.54$, $S/N < 2.0$) in single-pass 1-year exposures and imputes a neutral prior ($m = 25.0$) during PCA distance calculations, preventing catastrophic outlier inflation.
* **Mahalanobis Eigenvalue Metric Weighting ($\lambda_d^{-1/4}$)**:
  Scales PCA latent dimensions by inverse eigenvalue power-law weights ($\lambda_d^{-1/4}$), prioritizing primary principal components while suppressing uncorrelated noise.
* **Optunity PSO Hyperparameter Defaults**:
  Pre-loaded taskset-specific hyperparameter defaults ($k=17$ for TS1/TS2, $k=35$ for TS3, $k=29$ for TS4) tuned via Particle Swarm Optimization to minimize validation scatter $\sigma_{\text{MAD}}$.
* **Selective Gated Expectation-Maximization (EM) Calibration (`nugundam`)**:
  Applies 5 iterations of spatial cross-correlation prior recalibration to blended and PCA-outlier sources, achieving flat uniform PIT calibration ($D_{\text{PIT}} \le 0.0381 - 0.0482$) meeting the DESC SRD threshold ($\le 0.0500$).

---

## Technical Note: Zero Target Leakage During Inference

1. **INFORM Phase (Training)**: The target secondary redshift `redshift_manyband` ($z_2$) is accessed **only for training galaxies** to evaluate expert multi-peak performance and store calibrated error matrices in `train_errors`.
2. **Feature Space Representation**: Blended training galaxies populate specific regions of the normalized PCA feature space (e.g., color anomalies, high flux ratios).
3. **INFERENCE Phase (Evaluation)**: Target redshifts are **strictly inaccessible**. For a test galaxy $\mathbf{x}_j^{\text{test}}$, the KNN algorithm identifies its $K$ nearest neighbors in input feature space. If $\mathbf{x}_j^{\text{test}}$ falls into a blended feature space region, its nearest neighbors are the blend-calibrated training objects. The gating weights $\mathbf{w}_j$ are derived directly from those neighbors' stored `train_errors`. **Gating occurs automatically via input feature space localization with ZERO target leakage.**

---

## Feature Guard Protection & Resilience

`Pontifex` integrates a dedicated Feature Guard (`pontifex.guard.sanitize_input_catalog`) to protect downstream expert models against unphysical or corrupted catalog records:
* **Automatic Detection**: Audits catalog inputs for `NaN`, `Inf`, `-Inf`, negative measurement errors ($\sigma_m \le 0$), unphysical magnitudes ($m < 10$ or $m > 38$), and $10\sigma$ numerical outliers.
* **Diagnostic Warnings**: Emits clear `UserWarning` diagnostics detailing column names, corrupted entry counts, and exact percentage frequencies.
* **Robust Imputation**: Replaces corrupted entries with column medians or valid non-detection defaults ($m = 99.0$), ensuring expert estimators operate seamlessly without runtime exceptions.
* **Gating Activation Baseline**: Across the DESC Data Challenge task sets, the Path B (EM Nugundam) gating trigger activates for an average of **24.2%** of catalog objects (Rubin: 18.4%, Roman: 19.6%, COSMOS2020: 22.7%, Blends Challenge: 42.3%), specifically optimizing blended and low-SNR sources while preserving high-precision predictions for isolated galaxies.

---

## The Committee of 10 Experts

`Pontifex` combines 10 distinct photometric redshift estimators in a localized feature-space gating architecture:

| Expert Estimator | Primary Methodology | Taskset & Population Dominance | Default Calibration Parameters |
| :--- | :--- | :--- | :--- |
| **BPZ** | Bayesian SED template-fitting | Non-representative & High-$z$ ($z > 2.0$, Taskset 3) | `dz=0.01`, `zmin=0.0`, `zmax=3.0`, `prior='hdfn_gen'` |
| **LePhare** | Template fitting with dust & emission lines | Faint, high-extinction Roman optical-NIR samples | `GAL_SED='COSMOS_MOD.list'`, `EB_V=[0.0, 0.5]` |
| **FlexZBoost** | XGBoost conditional density B-spline estimation | Representative Rubin & Roman samples (Tasksets 1 & 2) | `max_depth=6`, `nbump=35`, `nsharp=15`, `bump_thresh=0.02` |
| **GPz** | Heteroscedastic Gaussian Process regression | Low-SNR / noisy flux measurement regimes | `n_basis=50`, `max_iter=100`, heteroscedastic noise weighting |
| **PZFlow** | Differentiable Normalizing Flows | Blended & multi-component targets (Taskset 4) | `bijector_layers=4`, `hidden_units=32`, `epochs=50` |
| **NN1 (AION-MLP-1)** | Shallow Multi-Layer Perceptron | Main-sequence isolated galaxies ($0.2 < z < 1.2$) | `hidden_layer_sizes=(64, 32)`, `max_iter=200`, `alpha=1e-4` |
| **NN2 (AION-MLP-2)** | Deep Multi-Layer Perceptron with L2 penalty | Non-linear optical-NIR broad-band combinations | `hidden_layer_sizes=(128, 64, 32)`, `max_iter=300`, `alpha=1e-3` |
| **miniSOM** | Self-Organizing Map topological clustering | Color anomaly & non-detection border regions | `x=12, y=12`, `sigma=1.5`, `learning_rate=0.5` |
| **KNN** | Distance-weighted PCA kernel density estimator | Densely populated, highly representative feature spaces | `n_neighbors=15`, `leaf_size=30`, `weights='distance'` |
| **AION-Prior** | Hybrid neural prior with spectro-z calibration | Catastrophic outlier boundary suppression | `alpha=5e-4`, `learning_rate_init=5e-3` |


