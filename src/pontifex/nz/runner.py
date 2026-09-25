"""Pontifex Challenge Runners: Taskset 1, 2, and 3 Pipeline Automation."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import joblib
import tables_io

try:
    import qp
except ImportError:
    qp = None

from ..core.constants import TOMO_BIN_EDGES, Z_BIN_EDGES
from ..core.features import extract_features
from .som import compute_som_density_weights
from .classifier import train_tomographic_classifier, predict_tomographic_bins
from .calibration import build_calibration_histograms
from .sampler import generate_correlated_realizations

KEY_OFFSETS: Dict[str, int] = {
    "taskset_1_cardinal_1yr": 0,
    "taskset_1_cardinal_4yr": 1_000_000,
    "taskset_1_flagship_1yr": 2_000_000,
    "taskset_1_flagship_4yr": 3_000_000,
    "taskset_2_cardinal_1yr": 4_000_000,
    "taskset_2_cardinal_4yr": 5_000_000,
    "taskset_2_flagship_1yr": 6_000_000,
    "taskset_2_flagship_4yr": 7_000_000,
    "taskset_3_cardinal_1yr": 4_000_000,
    "taskset_3_cardinal_4yr": 5_000_000,
    "taskset_3_flagship_1yr": 6_000_000,
    "taskset_3_flagship_4yr": 7_000_000,
}


def train_and_calibrate(
    ddf_files: List[Union[str, Path]],
    key: str,
    models_dir: Union[str, Path],
    wfd_file: Optional[Union[str, Path]] = None,
) -> Tuple[Any, np.ndarray]:
    """Train XGBoost classifier with SOM transfer weights and build empirical calibration histograms."""
    taskset = key[0:9]
    tomo_edges = TOMO_BIN_EDGES[taskset]
    grid_edges = Z_BIN_EDGES[taskset]
    n_tomo_bins = len(tomo_edges) - 1

    # Load and concatenate DDF catalogs
    data_list = [tables_io.read(f) for f in ddf_files]
    common_keys = list(set.intersection(*[set(d.keys()) for d in data_list]))
    combined = {}
    for k in common_keys:
        arrays = [d[k] for d in data_list]
        combined[k] = np.concatenate(arrays)

    # Resolve true redshifts
    z = combined["redshift"].copy()
    if "redshift_manyband" in combined:
        fill = ~np.isfinite(z) & np.isfinite(combined["redshift_manyband"])
        z[fill] = combined["redshift_manyband"][fill]

    valid = np.isfinite(z)
    z_clean = z[valid]
    combined_clean = {k: v[valid] for k, v in combined.items()}

    y_true = np.digitize(z_clean, tomo_edges[1:-1])
    X = extract_features(combined_clean)

    sample_weights = None
    if wfd_file is not None and Path(wfd_file).exists():
        try:
            wfd_sample = tables_io.read(wfd_file)
            X_wfd = extract_features(wfd_sample)
            sample_weights = compute_som_density_weights(X, X_wfd)
        except Exception:
            sample_weights = None

    # 1. Stratified 5-Fold Cross-Validation for Out-Of-Fold (OOF) Calibration
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_oof = np.zeros(len(y_true), dtype=int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_true)):
        sw_fold = sample_weights[train_idx] if sample_weights is not None else None
        clf_fold = train_tomographic_classifier(
            X[train_idx],
            y_true[train_idx],
            sample_weights=sw_fold,
            random_state=42 + fold,
        )
        probs_val = clf_fold.predict_proba(X[val_idx])
        y_pred_oof[val_idx] = np.argmax(probs_val, axis=1)

    # 2. Build empirical calibration histograms from OUT-OF-FOLD predictions
    # This prevents in-sample calibration overfitting and preserves realistic dispersion
    calib_hists = build_calibration_histograms(
        z_true=z_clean,
        y_pred=y_pred_oof,
        n_tomo_bins=n_tomo_bins,
        grid_edges=grid_edges,
        sample_weights=sample_weights,
    )

    # 3. Train final production classifier on all training data
    clf = train_tomographic_classifier(
        X,
        y_true,
        sample_weights=sample_weights,
        random_state=42,
    )

    os.makedirs(models_dir, exist_ok=True)
    model_path = Path(models_dir) / f"{key}_model.joblib"
    joblib.dump({"clf": clf, "calib_hists": calib_hists}, model_path)

    return clf, calib_hists


def predict_and_generate_outputs(
    key: str,
    wfd_file: Union[str, Path],
    clf: Any,
    calib_hists: np.ndarray,
    output_nz_estimate_file: Union[str, Path],
    output_bhat_file: Union[str, Path],
    output_nz_samples_file: Optional[Union[str, Path]] = None,
) -> None:
    """Predict tomographic bins on WFD catalog and write qp Ensembles and bhat files."""
    taskset = key[0:9]
    tomo_edges = TOMO_BIN_EDGES[taskset]
    grid_edges = Z_BIN_EDGES[taskset]
    n_tomo_bins = len(tomo_edges) - 1

    wfd_data = tables_io.read(wfd_file)
    n_objects = len(wfd_data[list(wfd_data.keys())[0]])

    X_wfd = extract_features(wfd_data)
    bin_assignments, _ = predict_tomographic_bins(clf, X_wfd, n_tomo_bins=n_tomo_bins)

    offset = KEY_OFFSETS.get(key, 0)
    sequential_ids = np.arange(offset, offset + n_objects, dtype=int)

    # 1. Write bhat assignments
    bhat_dict = {
        "tomo_bin_index": bin_assignments,
        "object_id": sequential_ids,
    }
    Path(output_bhat_file).parent.mkdir(parents=True, exist_ok=True)
    tables_io.write(bhat_dict, output_bhat_file)

    # 2. Write central nz_estimate file
    bin_counts = np.bincount(bin_assignments, minlength=n_tomo_bins)
    ens_estimate = qp.hist.create_ensemble(grid_edges, calib_hists)
    ens_estimate.set_ancil(dict(n_objects=bin_counts))
    Path(output_nz_estimate_file).parent.mkdir(parents=True, exist_ok=True)
    ens_estimate.write_to(output_nz_estimate_file)

    # 3. Write correlated realizations nz_samples file
    if output_nz_samples_file is not None:
        n_realizations = 100
        realization_matrix, bin_idx, i_real = generate_correlated_realizations(
            calib_hists=calib_hists,
            grid_edges=grid_edges,
            n_realizations=n_realizations,
        )
        ens_samples = qp.hist.create_ensemble(grid_edges, realization_matrix)
        ens_samples.set_ancil(dict(bin_idx=bin_idx, i_realization=i_real))
        Path(output_nz_samples_file).parent.mkdir(parents=True, exist_ok=True)
        ens_samples.write_to(output_nz_samples_file)


def run_taskset_training_and_estimation(
    key: str,
    wfd_file: Union[str, Path],
    models_dir: Union[str, Path],
    ddf_files: List[Union[str, Path]],
    output_nz_estimate_file: Union[str, Path],
    output_bhat_file: Union[str, Path],
    output_nz_samples_file: Optional[Union[str, Path]] = None,
) -> None:
    clf, calib_hists = train_and_calibrate(ddf_files, key, models_dir, wfd_file=wfd_file)
    predict_and_generate_outputs(
        key,
        wfd_file,
        clf,
        calib_hists,
        output_nz_estimate_file,
        output_bhat_file,
        output_nz_samples_file,
    )


def run_taskset_estimation_only(
    key: str,
    wfd_file: Union[str, Path],
    models_dir: Union[str, Path],
    output_nz_estimate_file: Union[str, Path],
    output_bhat_file: Union[str, Path],
    output_nz_samples_file: Optional[Union[str, Path]] = None,
) -> None:
    task_key = key.replace("taskset_3", "taskset_2") if "taskset_3" in key else key
    model_path = Path(models_dir) / f"{task_key}_model.joblib"
    if not model_path.exists():
        model_path = Path(models_dir) / f"{key}_model.joblib"
    saved = joblib.load(model_path)
    predict_and_generate_outputs(
        key,
        wfd_file,
        saved["clf"],
        saved["calib_hists"],
        output_nz_estimate_file,
        output_bhat_file,
        output_nz_samples_file,
    )
