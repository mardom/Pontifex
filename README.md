# Pontifex

Modular, footprint-corrected photometric redshift estimation pipeline for LSST DESC.

## Installation

```bash
pip install -e .
```

## Features

- **Committee of Experts**: Integrates multiple RAIL estimators combined via local KNN gating.
- **Footprint-Correction**: Using SkyKatana masks.
- **Spatial Clustering Calibration**: Dynamic EM optimization using Nugundam 3D spatial cross-correlations.
