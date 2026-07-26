import os
import logging
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import aion_pz
from .estimators import CommitteeOfExperts, get_bands_and_ref, Z_CENTERS, Z_GRID
from .em import PontifexEM

logger = logging.getLogger(__name__)

def train_and_estimate(
    train_file: Union[str, Path],
    test_file: Union[str, Path],
    output_file: Union[str, Path],
    save_model_to: Union[str, Path, None] = None,
    seed: int = 42,
) -> None:
    """
    Train the committee of experts, perform footprint-corrected EM calibration
    via Nugundam, and predict on the test catalog.
    """
    logger.info(f"Loading catalogs: train={train_file}, test={test_file}")
    train_dict = aion_pz.load_catalog(train_file)
    test_dict = aion_pz.load_catalog(test_file)

    # Resolve true redshifts
    z_true = np.asarray(train_dict[aion_pz.REDSHIFT_COL], dtype="float64")
    if aion_pz.MANYBAND_COL in train_dict:
        z_many = np.asarray(train_dict[aion_pz.MANYBAND_COL], dtype="float64")
        fill = ~np.isfinite(z_true) & np.isfinite(z_many)
        z_true[fill] = z_many[fill]
    good = np.isfinite(z_true)
    train_dict = {k: v[good] for k, v in train_dict.items()}
    train_dict['redshift'] = z_true[good]

    bands, ref_band, is_roman = get_bands_and_ref(list(train_dict.keys()))

    # Determine CI vs Production mode
    n_train = len(train_dict['redshift'])
    is_ci = n_train < 1500

    # 1. Fit committee of experts
    logger.info("Fitting Committee of Experts...")
    committee = CommitteeOfExperts(is_ci=is_ci)
    committee.fit(train_dict, bands, ref_band, is_roman)

    # 2. Predict initial PDFs on Z_CENTERS
    logger.info("Predicting initial PDFs...")
    initial_pdfs = committee.predict(test_dict)

    # 3. Perform Nugundam EM Calibration
    logger.info("Running Nugundam EM Calibration...")
    ref_df = pd.DataFrame({
        'ra': train_dict['ra'],
        'dec': train_dict['dec'],
        'ztrue': train_dict['redshift']
    })
    unk_df = pd.DataFrame({
        'ra': test_dict['ra'],
        'dec': test_dict['dec']
    })

    # Limit EM iterations in CI to speed up testing
    em_max_iter = 2 if is_ci else 5
    em = PontifexEM(
        unk_df=unk_df,
        ref_df=ref_df,
        initial_pdfs=initial_pdfs,
        z_grid_edges=Z_GRID,
        use_bias_correction=True,
        floor_val=1e-5,
        learning_rate=0.2,
        smoothing_sigma=1.0
    )
    
    calibrated_pdfs = em.optimize(max_iter=em_max_iter)

    # 4. Interpolate and normalize onto Z_GRID
    f_interp = interp1d(Z_CENTERS, calibrated_pdfs, axis=1, kind='linear', fill_value='extrapolate')
    calibrated_pdfs_301 = f_interp(Z_GRID)
    calibrated_pdfs_301 = aion_pz._renorm(calibrated_pdfs_301, Z_GRID)

    # 5. Write outputs and save model weights
    logger.info(f"Writing final calibrated PDFs to {output_file}")
    aion_pz.write_qp(calibrated_pdfs_301, test_dict[aion_pz.OBJECT_ID_COL], output_file, z_grid=Z_GRID)

    if save_model_to is not None:
        logger.info(f"Saving trained model weights to {save_model_to}")
        committee.save(str(save_model_to))


def estimate_only(
    model_file: Union[str, Path],
    test_file: Union[str, Path],
    output_file: Union[str, Path],
) -> None:
    """
    Run committee prediction and Nugundam EM calibration using saved model weights.
    """
    logger.info(f"Loading model weights from {model_file}...")
    committee = CommitteeOfExperts()
    committee.load(str(model_file))
    
    logger.info(f"Loading test catalog: {test_file}")
    test_dict = aion_pz.load_catalog(test_file)

    # 1. Predict initial PDFs on Z_CENTERS
    logger.info("Predicting initial PDFs...")
    initial_pdfs = committee.predict(test_dict)

    # 2. Run Nugundam EM Calibration using stored training coordinates as reference
    logger.info("Running Nugundam EM Calibration...")
    train_som = committee.model_dict["train_dict_som"]
    ref_df = pd.DataFrame({
        'ra': train_som['ra'],
        'dec': train_som['dec'],
        'ztrue': train_som['redshift']
    })
    unk_df = pd.DataFrame({
        'ra': test_dict['ra'],
        'dec': test_dict['dec']
    })

    n_test = len(test_dict[list(test_dict.keys())[0]])
    is_ci = n_test < 1500
    em_max_iter = 2 if is_ci else 5

    em = PontifexEM(
        unk_df=unk_df,
        ref_df=ref_df,
        initial_pdfs=initial_pdfs,
        z_grid_edges=Z_GRID,
        use_bias_correction=True,
        floor_val=1e-5,
        learning_rate=0.2,
        smoothing_sigma=1.0
    )
    
    calibrated_pdfs = em.optimize(max_iter=em_max_iter)

    # 3. Interpolate and normalize onto Z_GRID
    f_interp = interp1d(Z_CENTERS, calibrated_pdfs, axis=1, kind='linear', fill_value='extrapolate')
    calibrated_pdfs_301 = f_interp(Z_GRID)
    calibrated_pdfs_301 = aion_pz._renorm(calibrated_pdfs_301, Z_GRID)

    # 4. Write outputs
    logger.info(f"Writing final calibrated PDFs to {output_file}")
    aion_pz.write_qp(calibrated_pdfs_301, test_dict[aion_pz.OBJECT_ID_COL], output_file, z_grid=Z_GRID)
