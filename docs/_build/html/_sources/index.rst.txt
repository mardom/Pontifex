.. Pontifex documentation master file, created by
   sphinx-quickstart on Sat Jul 25 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Pontifex's documentation!
====================================

**Pontifex** is a modern, modular, footprint-corrected photometric redshift estimation pipeline. It builds on the LSST DESC photo-z WG's RAIL framework and integrates advanced spatial clustering calibration algorithms.

Key Features:
-------------

* **Committee of Experts**: Integrates multiple machine learning estimators (MLPs, miniSOM, normalizing flows, Gaussian processes, and templates) combined dynamically using local K-Nearest Neighbors gating.
* **Footprint-Correction**: Corrects correlation functions for spatial area loss using `SkyKatana` boolean masking.
* **Spatial Clustering Calibration**: Calibrates photo-z PDFs using dynamic EM optimization with `Nugundam`'s 3D spatial cross-correlations.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   autoapi/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
