Installation & Hyperparameter Optimization
===========================================

This section documents the environment generation, package installation, and automated out-of-fold hyperparameter optimization methodology for the **Pontifex** (`v2.2.0` / `AionPlus`) pipeline.

Installation and Environment Management
---------------------------------------

Pontifex is packaged as a standard pip-installable Python package (`version = "2.2.0"`). You can set up the complete environment using the provided ``environment.yml`` or ``requirements.txt``:

Option A: Automated Conda Environment Creation (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To automatically install Python, all scientific dependencies, RAIL estimators, and Pontifex:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate pontifex_env

Option B: Pip Requirements Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, install dependencies via ``requirements.txt`` and install ``pontifex`` in editable or standard mode:

.. code-block:: bash

   conda create -n pontifex_env python=3.13 -y
   conda activate pontifex_env
   pip install -r requirements.txt
   pip install -e .

Verify the installation by running the 21-test verification suite:

.. code-block:: bash

   pytest -v tests/

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

Out-of-Fold Calibration & Hyperparameter Optimization (PSO)
-----------------------------------------------------------

Pontifex implements an automated hyperparameter tuning workflow using Particle Swarm Optimization (PSO) via the ``optunity`` package, paired with strict 80/20 out-of-fold (`fit_dict` / `cal_dict`) gating and empirical PIT CDF recalibration.

Methodology
~~~~~~~~~~~

During ``CommitteeOfExperts.fit()``, the training catalog is partitioned into an 80% base-expert fitting subset (`fit_dict`) and a disjoint 20% calibration subset (`cal_dict`). Base experts are fitted exclusively on `fit_dict`, while local kNN expert gating weights (:math:`K=60`, inverse-cube precision weighting on :math:`0.5\,\lvert\Delta z\rvert/(1+z) + 0.5\,\text{CRPS}/(1+z)` with self-neighbor exclusion) and damped monotone PIT inverse-CDF recalibration are fitted strictly on `cal_dict`.

When ``optimize_hyperparams=True`` is enabled, Particle Swarm Optimization searches over:

* **Self-Organizing Maps (SOM)**:
  
  * ``sigma`` (neighborhood spread): range ``[0.8, 2.5]`` (default ``1.15``)
  * ``learning_rate``: range ``[0.2, 0.8]`` (default ``0.72``)
  * ``n_dim`` (SOM map width/height): range ``[8, 16]`` (default ``16``)

* **K-Nearest Neighbors (KNN & Local Precision Gating)**:
  
  * ``nneigh_min`` (base KNN neighbor count): default ``9``
  * ``ngrid_sigma`` (KDE bandwidth grid): default ``3``
  * ``K_val`` (gating neighborhood size on `cal_dict`): default ``60``

* **FlexZBoost**:
  
  * ``max_depth`` (tree depth): default ``8``
  * ``nsharp`` (sharpening grid steps): default ``8``
  * ``nbump`` (bump threshold grid steps): default ``13``

* **Gaussian Process Redshifts (GPz)**:
  
  * ``n_basis`` (number of basis functions): default ``59``
  * ``max_iter`` (optimizing iterations): default ``35`` (capped for generalization)

* **AION-PZ (MLP Classifier Head)**:
  
  * ``alpha`` (L2 regularization penalty): default ``5.89e-4``
  * ``learning_rate_init``: default ``2.26e-4``

Usage and On-Demand Triggering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, Pontifex uses precomputed optimal configurations (or loads ``results/pso_best_hyperparameters.pkl`` if present), ensuring fast, reproducible execution without nested cross-validation leakage.

To perform a fresh optimization on a new survey training catalog, set ``optimize_hyperparams=True``:

.. code-block:: python

   from pontifex import train_and_estimate

   train_and_estimate(
       train_file="path/to/new_train.hdf5",
       test_file="path/to/test.hdf5",
       output_file="path/to/output.hdf5",
       optimize_hyperparams=True
   )
