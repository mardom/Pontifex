#!/usr/bin/env python3
"""Run 5-fold cross-validation across all 16 LSST-DESC challenge datasets with AION+ configuration.

This script performs rigorous out-of-fold (OOF) evaluation of the updated Pontifex
Committee of Experts with the AION+ model (PIT-recalibrated neural density head + pure photometry / full feature modes):
1. 5-Fold cross-validation per configuration on representative samples (600 galaxies).
2. Evaluates full OOF probability density functions p_i(z) and z_mode.
3. Computes the complete set of challenge metrics:
   - Bias, Sigma_MAD, Catastrophic Outliers (>0.15)
   - Biweight statistics (Biweight Bias, Biweight Sigma_MAD, 3-Sigma Outliers)
   - PIT moments (Mean, Variance), KS statistic, PIT outliers
   - CRPS and Wasserstein-1 distance
4. Saves out-of-fold metrics to results/pontifex_aionplus_kfold_cv_metrics.csv.
"""

import os
import sys

# Pure CPU execution to prevent multi-process GPU contention / memory spikes
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["AION_PZ_DEVICE"] = "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["PONTIFEX_AION_FEATURES"] = "photo"

import time
import tempfile
import concurrent.futures
import numpy as np
import pandas as pd
from scipy.stats import kstest, wasserstein_distance
from astropy.stats import biweight_location, biweight_scale
from sklearn.model_selection import KFold
import tables_io
import qp

if not hasattr(np, 'trapezoid'):
    np.trapezoid = getattr(np, 'trapz', None)

# Paths
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_PZ_CHALLENGE = os.path.join(os.path.dirname(_REPO_ROOT), 'pz_challenge')
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, _PZ_CHALLENGE)

PUBLIC_DIR = os.path.join(_PZ_CHALLENGE, 'tests', 'public')
RESULTS_DIR = os.path.join(_REPO_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

import aion_pz
from pontifex.estimators import CommitteeOfExperts, get_bands_and_ref, Z_CENTERS, Z_GRID
from pontifex.guard import sanitize_input_catalog

TASKSETS = [1, 2, 3, 4]
SIMS = ['cardinal', 'flagship']
SCENARIOS = ['1yr', '10yr']


def evaluate_dataset_cv(t: int, sim: str, sc: str, n_samples: int = 600, n_splits: int = 5):
    """Execute 5-fold cross-validation on a single dataset with AION+ configuration."""
    prefix = f"taskset_{t}_{sim}_{sc}"
    train_file = os.path.join(PUBLIC_DIR, f"pz_challenge_taskset_{t}_{sim}_training_{sc}.hdf5")

    if not os.path.exists(train_file):
        print(f"[{prefix}] WARNING: File {train_file} not found!")
        return None

    print(f"[{prefix}] Starting 5-fold cross-validation ({n_samples} galaxies)...")
    t0 = time.time()

    # Load catalog
    d = tables_io.read(train_file)
    z_spec = np.asarray(d['redshift'], dtype='float64')

    if 'redshift_manyband' in d:
        z_mb = np.asarray(d['redshift_manyband'], dtype='float64')
        z_true_full = np.where(np.isfinite(z_spec), z_spec, z_mb)
    else:
        z_true_full = z_spec

    valid = np.isfinite(z_true_full)
    d_valid = {k: np.asarray(v)[valid] for k, v in d.items()}
    z_valid = z_true_full[valid]

    n_avail = len(z_valid)
    n_use = min(n_samples, n_avail)
    rng = np.random.default_rng(42 + t * 10 + (0 if sim == 'cardinal' else 1) * 2 + (0 if sc == '1yr' else 1))
    sub_idx = rng.choice(n_avail, n_use, replace=False)

    sub_dict = {k: d_valid[k][sub_idx] for k in d_valid}
    sub_dict['redshift'] = z_valid[sub_idx]
    sub_dict, _ = sanitize_input_catalog(sub_dict)

    z_sub = sub_dict['redshift']
    bands, ref_band, is_roman = get_bands_and_ref(list(sub_dict.keys()), taskset_id=t)
    sample_type = (
        'Roman 3-band' if is_roman
        else ('Combined 9-band' if 'mag_Y_roman' in bands
              else ('Rubin Blends' if t == 4 else 'Rubin 6-band'))
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_pdfs = np.zeros((n_use, len(Z_CENTERS)))

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            for fold, (tr_idx, te_idx) in enumerate(kf.split(z_sub)):
                tr_dict = {k: sub_dict[k][tr_idx] for k in sub_dict}
                te_dict = {k: sub_dict[k][te_idx] for k in sub_dict}

                committee = CommitteeOfExperts(is_ci=True)
                committee.fit(tr_dict, bands, ref_band, is_roman)
                preds = committee.predict(te_dict)
                oof_pdfs[te_idx] = preds
        finally:
            os.chdir(orig_cwd)

    # 1. Point metrics from out-of-fold predictions
    oof_zmode = Z_CENTERS[np.argmax(oof_pdfs, axis=1)]
    dz = (oof_zmode - z_sub) / (1.0 + z_sub)

    bias_median = float(np.median(dz))
    sigma_mad = float(1.4826 * np.median(np.abs(dz - bias_median)))
    outliers_catastrophic = float(np.mean(np.abs(dz) > 0.15) * 100.0)

    bw_loc = float(biweight_location(dz))
    bw_scale = float(biweight_scale(dz))
    clip_mask = np.abs(dz - bw_loc) <= 3.0 * bw_scale
    bw_outliers = float(np.mean(~clip_mask) * 100.0)
    bw_sigma_mad = float(1.4826 * np.median(np.abs(dz[clip_mask] - np.median(dz[clip_mask]))))
    bw_bias = float(np.median(dz[clip_mask]))

    # 2. Distribution metrics from out-of-fold PDFs
    dz_grid = Z_GRID[1] - Z_GRID[0]
    cdf_mat = np.cumsum(oof_pdfs, axis=1) * (Z_CENTERS[1] - Z_CENTERS[0])
    cdf_mat /= np.maximum(cdf_mat[:, -1:], 1e-12)

    # PIT values
    pit_vals = np.array([np.interp(z_sub[i], Z_CENTERS, cdf_mat[i]) for i in range(n_use)])
    pit_vals = np.clip(pit_vals, 0.0, 1.0)

    mean_pit = float(np.mean(pit_vals))
    var_pit = float(np.var(pit_vals))
    ks_stat = float(kstest(pit_vals, 'uniform').statistic)
    pit_outliers = float(np.mean((pit_vals < 1e-4) | (pit_vals > 1.0 - 1e-4)) * 100.0)

    # CRPS
    theta_mat = (Z_CENTERS[None, :] >= z_sub[:, None]).astype(float)
    crps_array = np.trapezoid((cdf_mat - theta_mat)**2, Z_CENTERS, axis=1)
    mean_crps = float(np.mean(crps_array))

    # Stacked N(z) Wasserstein distance
    stacked_pz = np.mean(oof_pdfs, axis=0)
    stacked_norm = stacked_pz / (np.sum(stacked_pz) * (Z_CENTERS[1] - Z_CENTERS[0]))
    hist_true, _ = np.histogram(z_sub, bins=np.linspace(0.03, 3.0, len(Z_CENTERS)+1), density=True)
    w_dist = float(wasserstein_distance(Z_CENTERS, Z_CENTERS, u_weights=stacked_norm, v_weights=hist_true))

    duration = time.time() - t0
    print(f"[{prefix}] COMPLETED in {duration:.1f}s | Bias={bias_median:+.4f} | Sigma_MAD={sigma_mad:.4f} | Outliers={outliers_catastrophic:.2f}% | PIT_KS={ks_stat:.4f} | CRPS={mean_crps:.4f}")

    metrics_row = {
        'Taskset': f'Taskset {t}',
        'Sample Type': sample_type,
        'Simulation': sim.capitalize(),
        'Scenario': sc,
        'N_CV': n_use,
        'CV_Folds': n_splits,
        'OOF Bias': bias_median,
        'OOF Sigma_MAD': sigma_mad,
        'OOF Outliers (>0.15)': outliers_catastrophic,
        'OOF Biweight Bias': bw_bias,
        'OOF Biweight Sigma_MAD': bw_sigma_mad,
        'OOF 3-Sigma Outliers': bw_outliers,
        'OOF Mean PIT': mean_pit,
        'OOF Var(PIT)': var_pit,
        'OOF PIT KS': ks_stat,
        'OOF PIT Outliers (%)': pit_outliers,
        'OOF Mean CRPS': mean_crps,
        'OOF Wasserstein W1': w_dist,
        'Runtime (s)': round(duration, 1)
    }

    return metrics_row


def main():
    print("=" * 80)
    print("PONTIFEX (AION+): EXECUTING 5-FOLD CV ACROSS ALL 16 CHALLENGE DATASETS")
    print("=" * 80)

    configs = []
    for t in TASKSETS:
        for sim in SIMS:
            for sc in SCENARIOS:
                configs.append((t, sim, sc))

    out_csv = os.path.join(RESULTS_DIR, 'pontifex_aionplus_kfold_cv_metrics.csv')
    all_metrics = []
    completed_keys = set()
    if os.path.exists(out_csv):
        try:
            existing_df = pd.read_csv(out_csv)
            for _, r in existing_df.iterrows():
                row_d = r.to_dict()
                all_metrics.append(row_d)
                completed_keys.add((str(row_d['Taskset']), str(row_d['Simulation']).capitalize(), str(row_d['Scenario'])))
            print(f"Loaded {len(all_metrics)} existing metrics from {out_csv}")
        except Exception as ex:
            print(f"Notice: Could not load existing CSV: {ex}")

    to_run = [
        (t, sim, sc) for (t, sim, sc) in configs
        if (f"Taskset {t}", sim.capitalize(), sc) not in completed_keys
    ]
    print(f"Configurations to evaluate: {len(to_run)} / {len(configs)}")

    if to_run:
        # Run across 4 concurrent workers
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(evaluate_dataset_cv, t, sim, sc): (t, sim, sc) for (t, sim, sc) in to_run}
            for future in concurrent.futures.as_completed(futures):
                t, sim, sc = futures[future]
                try:
                    row = future.result()
                    if row is not None:
                        all_metrics.append(row)
                        # Persist immediately to prevent loss
                        df = pd.DataFrame(all_metrics).drop_duplicates(subset=['Taskset', 'Simulation', 'Scenario']).sort_values(by=['Taskset', 'Simulation', 'Scenario'])
                        df.to_csv(out_csv, index=False)
                        print(f"[PROGRESS] Updated {out_csv} ({len(df)}/{len(configs)} completed)")
                except Exception as e:
                    print(f"ERROR processing Taskset {t} {sim} {sc}: {e}")

    if all_metrics:
        df = pd.DataFrame(all_metrics).drop_duplicates(subset=['Taskset', 'Simulation', 'Scenario']).sort_values(by=['Taskset', 'Simulation', 'Scenario'])
        df.to_csv(out_csv, index=False)
        print(f"\n[SUCCESS] Saved complete AION+ 5-fold cross-validation metrics to: {out_csv}")
        print("\n" + "=" * 80)
        print("FINAL 5-FOLD CROSS-VALIDATION METRICS (AION+)")
        print("=" * 80)
        print(df.to_string(index=False))
    else:
        print("\n[ERROR] No metrics collected!")


if __name__ == '__main__':
    main()
