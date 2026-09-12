import os
import sys
import pickle
import h5py
import numpy as np
import pandas as pd
from astropy.stats import biweight_location, biweight_scale
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA

# Add Pontifex and pz_challenge paths
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_PZ_CHALLENGE = os.path.join(os.path.dirname(_REPO_ROOT), 'pz_challenge')
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, _PZ_CHALLENGE)

import aion_pz
import qp
from pontifex.estimators import CommitteeOfExperts, Z_CENTERS, Z_GRID, get_bands_and_ref, extract_features

def main():
    public_dir = os.path.join(_PZ_CHALLENGE, 'tests', 'public')
    output_dir = os.path.join(_REPO_ROOT, 'results', 'moe_predictions')
    os.makedirs(output_dir, exist_ok=True)

    pso_file = os.path.join(_REPO_ROOT, 'results', 'pso_best_hyperparameters.pkl')
    if os.path.exists(pso_file):
        with open(pso_file, 'rb') as f:
            pso_params = pickle.load(f)
    else:
        pso_params = {}

    tasksets = [1, 2, 3, 4]
    sims = ['cardinal', 'flagship']
    scenarios = ['1yr', '10yr']

    pca_dims_info = []
    all_metrics = []

    print("=" * 80)
    print("STARTING PONTIFEX STAGE 1: MIXTURE OF EXPERTS (MoE) PIPELINE EVALUATION")
    print("=" * 80)

    for t in tasksets:
        for sim in sims:
            for sc in scenarios:
                train_file = os.path.join(public_dir, f'pz_challenge_taskset_{t}_{sim}_training_{sc}.hdf5')
                if not os.path.exists(train_file):
                    continue

                print(f"\n---> Processing Taskset {t} | Simulation: {sim.capitalize()} | Scenario: {sc}")
                
                # Load catalog
                train_dict = aion_pz.load_catalog(train_file)
                z_true = train_dict['redshift']
                valid = np.isfinite(z_true)

                clean_dict = {k: v[valid] for k, v in train_dict.items()}
                z_true_clean = z_true[valid]
                n_clean = len(z_true_clean)

                # Get band configuration and reference band
                bands, ref_band, is_roman = get_bands_and_ref(list(clean_dict.keys()), taskset_id=t)
                sample_type = 'Roman' if is_roman else ('Combined' if 'mag_Y_roman' in bands else 'Rubin')

                # 1. Complete feature space: Magnitudes + Colors + Mag Errors + Color Errors
                feats = extract_features(clean_dict, bands, ref_band)
                feats_mean = np.mean(feats, axis=0)
                feats_std = np.where(np.std(feats, axis=0) == 0, 1.0, np.std(feats, axis=0))
                feats_norm = (feats - feats_mean) / feats_std

                # 2. PCA feature reduction retaining >90% variance
                pca_90 = PCA(n_components=0.90, random_state=42).fit(feats_norm)
                n_pca_components = pca_90.n_components_
                var_explained = np.sum(pca_90.explained_variance_ratio_) * 100.0

                pca_dims_info.append({
                    'Taskset': f'Taskset {t}',
                    'Sample Type': sample_type,
                    'Simulation': sim.capitalize(),
                    'Scenario': sc,
                    'Bands Count': len(bands),
                    'Complete Feature Space': feats.shape[1],
                    'PCA Reduced Dim (>90% Var)': n_pca_components,
                    'Variance Retained (%)': f'{var_explained:.2f}%'
                })

                print(f"    Sample Type: {sample_type} ({len(bands)} bands)")
                print(f"    Full Feature Dimension: {feats.shape[1]} | PCA Reduced Dim (>90% Var): {n_pca_components} ({var_explained:.2f}% retained)")

                out_moe_file = os.path.join(output_dir, f'taskset_{t}_{sim}_{sc}_moe_prediction.hdf5')
                if os.path.exists(out_moe_file) and os.path.getsize(out_moe_file) > 10_000_000:
                    print(f"    Found existing valid MoE prediction ({os.path.getsize(out_moe_file)/(1024*1024):.1f} MB), computing metrics directly...")
                    ens = qp.read(out_moe_file)
                    moe_pdf_301 = ens.pdf(Z_GRID)
                    z_mode = Z_GRID[np.argmax(moe_pdf_301, axis=1)]
                else:
                    # 3. Fit Committee of Experts stage with PSO hyperparameters
                    committee = CommitteeOfExperts(is_ci=True)
                    committee.pso_params = pso_params
                    committee.fit(clean_dict, bands, ref_band, is_roman, optimize_hyperparams=False)

                    # 4. Predict Mixture of Experts (MoE) weighted ensemble PDF
                    moe_pdf = committee.predict(clean_dict)
                    f_interp = interp1d(Z_CENTERS, moe_pdf, axis=1, kind='linear', fill_value='extrapolate')
                    moe_pdf_301 = aion_pz._renorm(f_interp(Z_GRID), Z_GRID)
                    aion_pz.write_qp(moe_pdf_301, clean_dict['object_id'], out_moe_file, z_grid=Z_GRID)
                    z_mode = Z_CENTERS[np.argmax(moe_pdf, axis=1)]
                    print(f"    Saved MoE prediction to: {out_moe_file}")

                # 5. Compute challenge metrics for the final MoE ensemble
                dz = (z_mode - z_true_clean) / (1.0 + z_true_clean)

                bias_raw = np.median(dz)
                sigma_mad_raw = 1.4826 * np.median(np.abs(dz - bias_raw))
                raw_outliers = np.mean(np.abs(dz) > 0.15) * 100.0

                bw_loc = biweight_location(dz)
                bw_scale = biweight_scale(dz)
                clip_mask = np.abs(dz - bw_loc) <= 3.0 * bw_scale
                bw_outliers = np.mean(~clip_mask) * 100.0
                bw_sigma_mad = 1.4826 * np.median(np.abs(dz[clip_mask] - np.median(dz[clip_mask])))
                bw_bias = np.median(dz[clip_mask])

                all_metrics.append({
                    'Taskset': f'Taskset {t}',
                    'Sample Type': sample_type,
                    'Simulation': sim.capitalize(),
                    'Scenario': sc,
                    'Sample N': f'{n_clean:,}',
                    'PCA Dim (>90%)': n_pca_components,
                    'Median Bias (Raw)': f'{bias_raw:+.4f}',
                    'Sigma_MAD (Raw)': f'{sigma_mad_raw:.4f}',
                    'Unclipped Outliers (|dz|>0.15)': f'{raw_outliers:.2f}%',
                    'Median Bias (Biweight)': f'{bw_bias:+.4f}',
                    'Sigma_MAD (Biweight)': f'{bw_sigma_mad:.4f}',
                    'Biweight 3-Sigma Outliers': f'{bw_outliers:.2f}%'
                })
                # Incremental write
                pd.DataFrame(all_metrics).to_csv(os.path.join(_REPO_ROOT, 'results', 'pontifex_stage1_moe_metrics.csv'), index=False)

    # Save default hyperparameters for Stage 1 MoE
    default_params_path = os.path.join(_REPO_ROOT, 'results', 'pontifex_moe_defaults.pkl')
    with open(default_params_path, 'wb') as f:
        pickle.dump(pso_params, f)

    print("\n" + "=" * 80)
    print("PCA REDUCED FEATURE SPACE DIMENSIONALITY (>90% VARIANCE RETAINED)")
    print("=" * 80)
    df_pca = pd.DataFrame(pca_dims_info)
    print(df_pca.to_string(index=False))

    print("\n" + "=" * 80)
    print("PONTIFEX STAGE 1 (MoE) CHALLENGE METRICS REPORT ON COMPLETE DATASET")
    print("=" * 80)
    df_metrics = pd.DataFrame(all_metrics)
    print(df_metrics.to_string(index=False))

    # Save metrics to CSV
    metrics_csv = os.path.join(_REPO_ROOT, 'results', 'pontifex_stage1_moe_metrics.csv')
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"\nSaved metrics summary to: {metrics_csv}")

if __name__ == '__main__':
    main()
