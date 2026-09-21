"""Backward-compatibility shim for pontifex.pipeline (migrated to pontifex.pz.pipeline)."""

from .pz.pipeline import train_and_estimate, estimate_only

__all__ = ["train_and_estimate", "estimate_only"]
