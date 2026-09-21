"""Backward-compatibility shim for pontifex.em (migrated to pontifex.pz.em)."""

from .pz.em import PontifexEM, generate_footprint_randoms_combined

__all__ = ["PontifexEM", "generate_footprint_randoms_combined"]
