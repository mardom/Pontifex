Pontifex Hands-On Tutorial
==========================

This tutorial provides a complete walkthrough for setting up, running, and evaluating **Pontifex** (`v2.2.0` / `AionPlus`) on photometric data catalogs.

Notebook Repository Location
----------------------------

The interactive Jupyter notebook is located in the repository under:
``notebooks/pontifex_tutorial.ipynb``

Step 1: Setup & Import
----------------------

Import the core ``Pontifex`` modules:

.. code-block:: python

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt

   from pontifex.core.guard import sanitize_input_catalog
   from pontifex.pz.estimators import CommitteeOfExperts, Z_CENTERS, Z_GRID
   from pontifex.pz.em import PontifexEM
   from pontifex.pz.pipeline import train_and_estimate, estimate_only

Step 2: Feature Guard & Physical Upper-Limit Imputation
-------------------------------------------------------

Before feeding magnitudes and errors to machine learning experts, pass catalog dictionaries through ``sanitize_input_catalog``. Non-detected magnitudes (`NaN`, `Inf`, `99.0` sentinels) are mapped to their physical band-specific :math:`1\sigma` limiting magnitudes (`BAND_MAG_LIMITS`) and their associated uncertainties are set to :math:`1.0\,\text{mag}` (:math:`\text{SNR}\approx 1`):

.. code-block:: python

   # Sanitize and protect input catalog with physical 1-sigma limiting magnitudes
   sanitized_catalog, guard_report = sanitize_input_catalog(catalog_dict, raise_warnings=True)

   print(f"Total processed objects: {guard_report['total_records']}")

Step 3: Training the Out-of-Fold Committee of Experts
-----------------------------------------------------

Train the ensemble of experts (MLPs, KNN, SOM, FlexZBoost, GPz, BPZ-lite, AION-PZ) using the internal 80/20 `fit_dict`/`cal_dict` out-of-fold split, local kNN precision gating, and empirical PIT CDF recalibration:

.. code-block:: python

   committee = CommitteeOfExperts(is_ci=False)

   # Fit model on training catalog (supports 6-band Rubin or 9-band Rubin + Roman)
   model_dict = committee.fit(
       sanitized_catalog,
       bands=[
           'mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst',
           'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst',
           'mag_Y_roman', 'mag_J_roman', 'mag_H_roman',
       ],
       ref_band='mag_i_lsst',
       is_roman=False,
   )

   # Predict recalibrated probability density functions
   predicted_pdfs = committee.predict(sanitized_catalog)

Step 4: Selective Spatial Clustering Calibration (``PontifexEM``)
-----------------------------------------------------------------

For blended or PCA-outlier sources (:math:`D_M^2 > 11.34` in 3D PCA space), calibrate the predicted PDFs using projected 3D spatial cross-correlations with :math:`0.85/0.15` prior damping:

.. code-block:: python

   unk_df = pd.DataFrame({'ra': sanitized_catalog['ra'], 'dec': sanitized_catalog['dec']})
   ref_df = pd.DataFrame({
       'ra': sanitized_catalog['ra'],
       'dec': sanitized_catalog['dec'],
       'ztrue': sanitized_catalog['redshift'],
   })

   em = PontifexEM(
       unk_df=unk_df,
       ref_df=ref_df,
       initial_pdfs=predicted_pdfs,
       z_grid_edges=Z_GRID,
       is_ci=False,
   )

   calibrated_pdfs = em.optimize(max_iter=5)

Step 5: End-to-End Pipeline Execution
-------------------------------------

To run the complete training, sanitization, out-of-fold gating, PIT recalibration, sub-grid `zmode` extraction, and selective EM calibration pipeline from file inputs:

.. code-block:: python

   train_and_estimate(
       train_file="path/to/train_catalog.hdf5",
       test_file="path/to/test_catalog.hdf5",
       output_file="path/to/predictions.hdf5",
       save_model_to="path/to/pontifex_model.joblib",
       optimize_hyperparams=False,
   )
