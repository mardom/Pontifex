Installation & Hyperparameter Optimization
===========================================

This section documents the environment generation, package installation, and automated hyperparameter optimization methodology for the **Pontifex** pipeline.

Installation and Environment Management
---------------------------------------

Pontifex is packaged as a standard pip-installable Python package. You can set up the complete environment using the provided ``environment.yml`` or ``requirements.txt``:

Option A: Automated Conda Environment Creation (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To automatically install Python 3.11, all scientific dependencies, RAIL estimators, and Pontifex:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate pontifex_env

Option B: Pip Requirements Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, install dependencies via ``requirements.txt``:

.. code-block:: bash

   conda create -n pontifex_env python=3.11 -y
   conda activate pontifex_env
   pip install -r requirements.txt
   pip install -e .

Troubleshooting Compiler / Linker Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are using Anaconda compilers and compile-time links fail with the following error:

.. code-block:: text

   /usr/bin/ld: cannot find /lib64/libm.so.6: No such file or directory
   /usr/bin/ld: cannot find /lib64/libmvec.so.1: No such file or directory

This is a known issue with absolute path lookup scripts in some Anaconda sysroot packages on Debian/Ubuntu systems. To resolve it, navigate to your Anaconda sysroot library folder (e.g., ``~/anaconda3/x86_64-conda-linux-gnu/sysroot/lib64/``) and modify the linker scripts (``libm.so``, ``libc.so``, and ``libm.a``) to use relative names:

* In ``libm.so``, change:
  ``GROUP ( /lib64/libm.so.6  AS_NEEDED ( /lib64/libmvec.so.1 ) )``
  to:
  ``GROUP ( libm.so.6  AS_NEEDED ( libmvec.so.1 ) )``
* In ``libc.so``, change:
  ``GROUP ( /lib64/libc.so.6 /usr/lib64/libc_nonshared.a  AS_NEEDED ( /lib64/ld-linux-x86-64.so.2 ) )``
  to:
  ``GROUP ( libc.so.6 libc_nonshared.a  AS_NEEDED ( ld-linux-x86-64.so.2 ) )``
* In ``libm.a``, change:
  ``GROUP ( /usr/lib64/libm-2.39.a /usr/lib64/libmvec.a )``
  to:
  ``GROUP ( libm-2.39.a libmvec.a )``

---

Hyperparameter Optimization (PSO)
---------------------------------

Pontifex implements an automated hyperparameter tuning workflow using Particle Swarm Optimization (PSO) via the ``optunity`` package. 

Methodology
~~~~~~~~~~~

Tuning is performed on-demand via the ``CommitteeOfExperts.fit()`` method when new training datasets are supplied. The optimization workflow splits the training data into a training subset (80%) and a validation subset (20%). It minimizes the Median Absolute Deviation (MAD) metric of the estimated redshift PDFs on the validation subset.

The following experts and hyperparameters are optimized in the loop:

* **Self-Organizing Maps (SOM)**:
  
  * ``sigma`` (neighborhood spread): range ``[0.5, 3.0]``
  * ``learning_rate``: range ``[0.1, 0.9]``
  * ``n_dim`` (SOM map width/height): range ``[8, 20]``

* **K-Nearest Neighbors (KNN & AION PCA Gating)**:
  
  * ``D_PCA`` (latent space dimension): dynamic 95% cumulative variance retention ($D_{\text{PCA}} \in [28, 37]$)
  * ``metric_weighting``: Mahalanobis eigenvalue power-law weights ($\lambda_d^{-1/4}$)
  * ``k_opt`` (neighbor count): range ``[10, 50]`` (tuned to $k=17$ for TS1/TS2, $k=35$ for TS3, $k=29$ for TS4)
  * ``bw_mult`` (bandwidth scale multiplier): range ``[0.3, 1.5]`` (tuned to $bw=0.873, 0.470, 0.546, 1.109$)

* **FlexZBoost**:
  
  * ``max_depth`` (tree depth): range ``[3, 10]``
  * ``nsharp`` (number of thresholding components): range ``[5, 30]``
  * ``nbump`` (number of boosting steps): range ``[10, 100]``

* **Gaussian Process Redshifts (GPz)**:
  
  * ``n_basis`` (number of basis functions): range ``[20, 100]``
  * ``max_iter`` (optimizing iterations): range ``[30, 200]``

* **AION (Classifying MLP Prior)**:
  
  * ``alpha`` (L2 regularization penalty): range ``[1e-5, 1e-3]``
  * ``learning_rate_init``: range ``[1e-3, 1e-2]``

Usage and On-Demand Triggering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, Pontifex loads the precomputed optimal configurations from ``results/pso_best_hyperparameters.pkl``. This ensures fast execution and consistent, optimal metrics without re-running the swarm search.

To perform a fresh optimization (for example, when a new training sample becomes available), set the ``optimize_hyperparams`` toggle to ``True``:

.. code-block:: python

   from pontifex import train_and_estimate

   train_and_estimate(
       train_file="path/to/new_train.hdf5",
       test_file="path/to/test.hdf5",
       output_file="path/to/output.hdf5",
       optimize_hyperparams=True
   )

This triggers the Particle Swarm Optimization run, updating the persistent config file ``results/pso_best_hyperparameters.pkl`` with the new optimal settings, which will then be used as the default for all future inference runs.
