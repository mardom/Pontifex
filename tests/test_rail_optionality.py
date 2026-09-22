"""Test suite verifying that Pontifex imports and core/nz functions work when RAIL is absent."""

import sys
import subprocess
import pytest


def test_import_without_rail():
    """Verify that importing pontifex without rail succeeds and flags HAS_RAIL=False."""
    code = """
import sys
sys.modules['rail'] = None
sys.modules['rail.core'] = None
sys.modules['lephare'] = None
sys.modules['aion_pz'] = None
import pontifex
assert pontifex.__version__ == '2.0.1'
assert pontifex.HAS_RAIL is False
from pontifex import CommitteeOfExperts, PontifexEM, extract_features
assert callable(CommitteeOfExperts)
assert callable(PontifexEM)
assert callable(extract_features)
"""
    cmd = [sys.executable, "-c", code]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Import failed without rail: {res.stderr}"


def test_core_and_nz_without_rail():
    """Verify that pontifex.core and pontifex.nz are completely usable without rail."""
    code = """
import sys
sys.modules['rail'] = None
import numpy as np
import pontifex.core as core
import pontifex.nz as nz

mags = {
    'mag_u_lsst': np.array([22.0, 23.0]),
    'mag_g_lsst': np.array([21.0, 22.0]),
    'mag_r_lsst': np.array([20.0, 21.0]),
    'mag_i_lsst': np.array([19.5, 20.5]),
    'mag_z_lsst': np.array([19.0, 20.0]),
    'mag_y_lsst': np.array([18.8, 19.8]),
}
feats = core.extract_features(mags)
assert feats.shape[0] == 2
"""
    cmd = [sys.executable, "-c", code]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Core/NZ usage failed without rail: {res.stderr}"


def test_committee_informative_error_without_rail():
    """Verify that calling CommitteeOfExperts.fit raises a descriptive ImportError without rail."""
    code = """
import sys
sys.modules['rail'] = None
from pontifex import CommitteeOfExperts
c = CommitteeOfExperts()
try:
    c.fit({}, [], '', False)
    sys.exit(1)
except ImportError as e:
    assert 'RAIL' in str(e)
    sys.exit(0)
"""
    cmd = [sys.executable, "-c", code]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Informative error check failed: {res.stderr}"

