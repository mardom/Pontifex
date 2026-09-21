"""Test suite for unified architecture exports and backwards-compatibility shims."""

import pytest


def test_top_level_submodules():
    import pontifex
    assert hasattr(pontifex, "core")
    assert hasattr(pontifex, "pz")
    assert hasattr(pontifex, "nz")
    assert pontifex.__version__ == "2.0.0"


def test_backwards_compatibility_shims():
    # Test root-level convenience imports
    from pontifex import (
        train_and_estimate,
        estimate_only,
        CommitteeOfExperts,
        PontifexEM,
        sanitize_input_catalog,
        extract_features,
    )
    assert callable(train_and_estimate)
    assert callable(estimate_only)
    assert callable(CommitteeOfExperts)
    assert callable(PontifexEM)
    assert callable(sanitize_input_catalog)
    assert callable(extract_features)

    # Test legacy module path imports
    from pontifex.guard import sanitize_input_catalog as legacy_guard
    from pontifex.em import PontifexEM as legacy_em
    from pontifex.pipeline import train_and_estimate as legacy_pipe
    from pontifex.estimators import CommitteeOfExperts as legacy_est

    assert callable(legacy_guard)
    assert callable(legacy_em)
    assert callable(legacy_pipe)
    assert callable(legacy_est)
