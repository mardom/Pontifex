import logging
import numpy as np
import pandas as pd
import astropy.table as at
from scipy.ndimage import gaussian_filter1d
import nugundam as ng

logger = logging.getLogger(__name__)

def generate_footprint_randoms_combined(ref_df: pd.DataFrame, unk_df: pd.DataFrame, columns=('ra', 'dec'), order_sparse=11):
    """
    Generate footprint-corrected random catalogs using SkyKatana.
    Falls back to bounding box randoms if the catalogs are too small (e.g. in CI)
    to prevent empty array failures in isolated pixel removal.
    """
    combined = pd.concat([ref_df[['ra', 'dec']], unk_df[['ra', 'dec']]], ignore_index=True)
    
    if len(combined) < 1500:
        logger.info("Catalog size too small for sparse footprint masking. Falling back to uniform bounding box randoms.")
        rng = np.random.default_rng(42)
        ra_min, ra_max = combined['ra'].min(), combined['ra'].max()
        dec_min, dec_max = combined['dec'].min(), combined['dec'].max()
        
        ref_rands = pd.DataFrame({
            'ra': rng.uniform(ra_min, ra_max, len(ref_df) * 3),
            'dec': rng.uniform(dec_min, dec_max, len(ref_df) * 3)
        })
        unk_rands = pd.DataFrame({
            'ra': rng.uniform(ra_min, ra_max, len(unk_df) * 3),
            'dec': rng.uniform(dec_min, dec_max, len(unk_df) * 3)
        })
        return ref_rands, unk_rands
        
    try:
        from skykatana import SkyMaskPipe
        logger.info("Building SkyKatana footprint mask on combined sample...")
        pipe = SkyMaskPipe(order_out=13)
        pipe.build_foot_mask(combined, columns=columns, order_sparse=order_sparse, remove_isopixels=True, erode_borders=True)
        
        logger.info("Generating footprint-corrected random catalogs...")
        ref_rands = pipe.makerans(stage='footmask', nr=len(ref_df) * 3)
        unk_rands = pipe.makerans(stage='footmask', nr=len(unk_df) * 3)
        return ref_rands, unk_rands
    except Exception as e:
        logger.warning(f"SkyKatana footprint masking failed with: {e}. Falling back to uniform bounding box randoms.")
        rng = np.random.default_rng(42)
        ra_min, ra_max = combined['ra'].min(), combined['ra'].max()
        dec_min, dec_max = combined['dec'].min(), combined['dec'].max()
        
        ref_rands = pd.DataFrame({
            'ra': rng.uniform(ra_min, ra_max, len(ref_df) * 3),
            'dec': rng.uniform(dec_min, dec_max, len(ref_df) * 3)
        })
        unk_rands = pd.DataFrame({
            'ra': rng.uniform(ra_min, ra_max, len(unk_df) * 3),
            'dec': rng.uniform(dec_min, dec_max, len(unk_df) * 3)
        })
        return ref_rands, unk_rands


class PontifexEM:
    """
    Expectation-Maximization (EM) optimization of photometric redshift PDFs
    using Nugundam clustering cross-correlations and SkyKatana masks.
    """
    def __init__(
        self,
        unk_df,
        ref_df,
        initial_pdfs,
        z_grid_edges,
        cosmo=None,
        pimax=100.0,
        unk_rands_df=None,
        ref_rands_df=None,
        mc_nreal=5,
        nthreads=4,
        use_bias_correction=True,
        floor_val=1e-5,
        learning_rate=0.2,
        smoothing_sigma=1.0
    ):
        self.unk_df = unk_df.copy()
        self.ref_df = ref_df.copy()
        self.initial_pdfs = np.array(initial_pdfs, dtype=float)
        self.z_grid_edges = np.array(z_grid_edges, dtype=float)
        self.z_bin_centers = 0.5 * (self.z_grid_edges[:-1] + self.z_grid_edges[1:])
        
        if cosmo is None:
            # Default to LSST DESC standard cosmology
            self.cosmo = ng.DistanceSpec(calcdist=True, h0=67.27, omegam=0.3121, omegal=0.6879)
        else:
            self.cosmo = cosmo
            
        self.pimax = pimax
        self.mc_nreal = mc_nreal
        self.nthreads = nthreads
        self.use_bias_correction = use_bias_correction
        self.floor_val = floor_val
        self.learning_rate = learning_rate
        self.smoothing_sigma = smoothing_sigma
        
        # Verify sizes
        if len(self.unk_df) != self.initial_pdfs.shape[0]:
            raise ValueError(f"Rows in unk_df ({len(self.unk_df)}) != initial_pdfs rows ({self.initial_pdfs.shape[0]})")
        if len(self.z_bin_centers) != self.initial_pdfs.shape[1]:
            raise ValueError(f"Z bins ({len(self.z_bin_centers)}) != initial_pdfs cols ({self.initial_pdfs.shape[1]})")
            
        # Build randoms using SkyKatana footprint-correction if not provided
        if ref_rands_df is None or unk_rands_df is None:
            self.ref_rands_df, self.unk_rands_df = generate_footprint_randoms_combined(self.ref_df, self.unk_df)
        else:
            self.ref_rands_df = ref_rands_df.copy()
            self.unk_rands_df = unk_rands_df.copy()
            
        rng = np.random.default_rng(1234)
        if 'ztrue' not in self.ref_rands_df.columns:
            self.ref_rands_df['ztrue'] = rng.choice(self.ref_df['ztrue'], len(self.ref_rands_df))
        if 'ztrue' not in self.unk_rands_df.columns:
            self.unk_rands_df['ztrue'] = rng.choice(self.ref_df['ztrue'], len(self.unk_rands_df))
            
        self.target_nz = None
        self.optimized_pdfs = None

    def compute_target_nz_nugundam(self, pdfs, seppmin=0.1, dsepp=0.2, nsepp=10):
        unk = at.Table.from_pandas(self.unk_df)
        ref = at.Table.from_pandas(self.ref_df)
        unk_rand = at.Table.from_pandas(self.unk_rands_df)
        ref_rand = at.Table.from_pandas(self.ref_rands_df)
        
        n_bins = len(self.z_grid_edges) - 1
        n_clust = np.zeros(n_bins)
        
        binning = ng.ProjectedBinning.from_binsize(
            nsepp=nsepp, seppmin=seppmin, dsepp=dsepp, logsepp=True,
            nsepv=1, dsepv=self.pimax
        )
        
        cfg_cross = ng.ProjectedCrossConfig(
            estimator="LS",
            columns_data1=ng.ProjectedCatalogColumns(ra="ra", dec="dec", redshift=None),
            columns_random1=ng.ProjectedCatalogColumns(ra="ra", dec="dec", redshift="ztrue"),
            columns_data2=ng.ProjectedCatalogColumns(ra="ra", dec="dec", redshift="ztrue"),
            columns_random2=ng.ProjectedCatalogColumns(ra="ra", dec="dec", redshift="ztrue"),
            nthreads=self.nthreads,
            binning=binning,
            grid=ng.ProjectedGridSpec(autogrid=True, pxorder="natural"),
            distance=self.cosmo
        )
        cfg_cross.mc_pdf.enabled = True
        cfg_cross.mc_pdf.sample_within_bin = True
        cfg_cross.mc_pdf.nreal = self.mc_nreal
        cfg_cross.mc_pdf.z_grid = self.z_grid_edges
        cfg_cross.mc_pdf.grid_kind = 'edges'
        cfg_cross.mc_pdf.pdf_data1 = ng.PDFSourceSpec(kind="external_matrix", matrix=pdfs)
        cfg_cross.mc_pdf.store_realizations = False
        cfg_cross.mc_pdf.resampling_backend = "fast"
        cfg_cross.mc_pdf.resampling_random_policy = "fixed"
        cfg_cross.progress.enabled = False
        
        cfg_auto = ng.ProjectedAutoConfig(
            estimator="LS",
            columns_data=ng.ProjectedCatalogColumns(ra="ra", dec="dec", redshift="ztrue"),
            columns_random=ng.ProjectedCatalogColumns(ra="ra", dec="dec", redshift="ztrue"),
            nthreads=self.nthreads,
            binning=binning,
            grid=ng.ProjectedGridSpec(autogrid=True, pxorder="natural"),
            distance=self.cosmo
        )
        cfg_auto.progress.enabled = False
        
        for k in range(n_bins):
            z_low = self.z_grid_edges[k]
            z_high = self.z_grid_edges[k+1]
            
            mask_ref = (ref['ztrue'] >= z_low) & (ref['ztrue'] < z_high)
            mask_ref_rand = (ref_rand['ztrue'] >= z_low) & (ref_rand['ztrue'] < z_high)
            
            ref_bin = ref[mask_ref]
            ref_rand_bin = ref_rand[mask_ref_rand]
            
            if len(ref_bin) < 5 or len(ref_rand_bin) < 10:
                n_clust[k] = self.floor_val
                continue
                
            try:
                res_cross = ng.pccf(unk, ref_bin, cfg_cross, random1=unk_rand, random2=ref_rand_bin)
                w_cross = np.array(res_cross.wp)
                w_cross_sum = np.sum(w_cross)
                
                if self.use_bias_correction:
                    res_auto = ng.pcf(ref_bin, ref_rand_bin, cfg_auto)
                    w_auto = np.array(res_auto.wp)
                    w_auto_sum = np.sum(w_auto)
                    
                    denom = np.sqrt(np.max([w_auto_sum, 1e-5]))
                    n_clust[k] = w_cross_sum / denom
                else:
                    n_clust[k] = w_cross_sum
            except Exception as e:
                logger.warning(f"Error computing correlation in bin {k} ([{z_low:.3f}, {z_high:.3f}]): {e}")
                n_clust[k] = self.floor_val
                
        nans = np.isnan(n_clust)
        if np.any(nans):
            x = np.arange(len(n_clust))
            n_clust[nans] = np.interp(x[nans], x[~nans], n_clust[~nans])
            
        n_clust = np.maximum(n_clust, self.floor_val)
        return n_clust / np.sum(n_clust)

    def optimize(self, max_iter=5, tol=1e-5, seppmin=0.1, dsepp=0.2, nsepp=10):
        current_pdfs = self.initial_pdfs.copy()
        eps = 1e-15
        
        logger.info(f"Starting EM optimization using nugundam (max_iter={max_iter}, tol={tol:.2e})...")
        
        for i in range(max_iter):
            rng = np.random.default_rng(1234 + i)
            # Sampling redshift based on current PDF
            z_samples = []
            for pdf in current_pdfs:
                p = np.asarray(pdf, dtype=np.float64).copy()
                p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
                p = np.maximum(p, 0.0)
                p_sum = p.sum()
                if p_sum > 0:
                    p /= p_sum
                else:
                    p = np.ones_like(p) / len(p)
                p = p / p.sum()
                p[-1] = 1.0 - p[:-1].sum()
                z_samples.append(rng.choice(self.z_bin_centers, p=p))
                
            self.unk_rands_df['ztrue'] = rng.choice(z_samples, size=len(self.unk_rands_df))
            
            target_nz = self.compute_target_nz_nugundam(
                pdfs=current_pdfs,
                seppmin=seppmin,
                dsepp=dsepp,
                nsepp=nsepp
            )
            
            if self.smoothing_sigma > 0:
                target_nz = gaussian_filter1d(target_nz, sigma=self.smoothing_sigma)
                target_nz = np.maximum(target_nz, self.floor_val)
                target_nz = target_nz / np.sum(target_nz)
                
            current_nz = np.sum(current_pdfs, axis=0)
            current_nz = np.where(current_nz == 0, eps, current_nz)
            current_nz = current_nz / np.sum(current_nz)
            
            ratio = (target_nz / current_nz) ** self.learning_rate
            new_pdfs = current_pdfs * ratio
            
            new_pdfs = np.nan_to_num(new_pdfs, nan=0.0, posinf=0.0, neginf=0.0)
            new_pdfs = np.maximum(new_pdfs, 0.0)
            row_sums = np.sum(new_pdfs, axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, eps, row_sums)
            new_pdfs = new_pdfs / row_sums
            
            diff = np.sqrt(np.mean((new_pdfs - current_pdfs) ** 2))
            current_pdfs = new_pdfs
            
            logger.info(f"[Iteration {i+1}] PDF RMS difference: {diff:.2e}")
            if diff < tol:
                logger.info(f"EM converged at iteration {i+1} (diff={diff:.2e})")
                break
        else:
            logger.warning(f"EM reached max_iter={max_iter} without converging to tol={tol:.2e}")
            
        self.optimized_pdfs = current_pdfs
        return self.optimized_pdfs
