Pontifex Hands-On Tutorial
==========================

This tutorial provides a complete walkthrough for setting up, running, and evaluating **Pontifex** on photometric data catalogs.

Notebook Repository Location
----------------------------

The interactive Jupyter notebook is located in the repository under:
``notebooks/pontifex_tutorial.ipynb``

Step 1: Setup & Import
----------------------

Import the core `Pontifex` modules:

.. code-block:: python

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt

   from pontifex.guard import sanitize_input_catalog
   from pontifex.estimators import CommitteeOfExperts, Z_CENTERS, Z_GRID
   from pontifex.em import PontifexEM
   from pontifex.pipeline import train_and_estimate

Step 2: Feature Guard & Input Protection
----------------------------------------

Before feeding magnitudes and errors to machine learning experts, pass catalog dictionaries through `sanitize_input_catalog`:

.. code-block:: python

   # Sanitize and protect input catalog from NaNs, Infs, negative errors, and extreme outliers
   sanitized_catalog, guard_report = sanitize_input_catalog(catalog_dict, raise_warnings=True)

   print(f"Total processed objects: {guard_report['total_records']}")

Step 3: Training the Committee of Experts
-----------------------------------------

Train the ensemble of experts (MLPs, KNN, FlexZBoost, GPz, BPZ, PZFlow) and predict baseline probability density functions:

.. code-block:: python

   committee = CommitteeOfExperts(is_ci=False)

   # Fit model on training catalog
   model_dict = committee.fit(
       sanitized_catalog,
       bands=['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst'],
       ref_band='mag_i_lsst',
       is_roman=False
   )

   # Predict probability density functions
   predicted_pdfs = committee.predict(sanitized_catalog)

Step 4: Spatial Clustering Calibration (`PontifexEM`)
-----------------------------------------------------

For blended or overlapping sources (activated automatically for ~24.2% of target objects), calibrate the predicted PDFs using projected 3D spatial cross-correlations:

.. code-block:: python

   unk_df = pd.DataFrame({'ra': sanitized_catalog['ra'], 'dec': sanitized_catalog['dec']})
   ref_df = pd.DataFrame({'ra': sanitized_catalog['ra'], 'dec': sanitized_catalog['dec'], 'ztrue': sanitized_catalog['redshift']})

   em = PontifexEM(
       unk_df=unk_df,
       ref_df=ref_df,
       initial_pdfs=predicted_pdfs,
       z_grid_edges=Z_GRID,
       is_ci=False
   )

   calibrated_pdfs = em.optimize(max_iter=5)

Step 5: End-to-End Pipeline Execution
-------------------------------------

To run the complete training, sanitization, estimation, and EM calibration pipeline from file inputs:

.. code-block:: python

   train_and_estimate(
       train_file="path/to/train_catalog.hdf5",
       test_file="path/to/test_catalog.hdf5",
       output_file="path/to/predictions.hdf5",
       optimize_hyperparams=False
   )
