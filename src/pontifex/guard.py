"""Backward-compatibility shim for pontifex.guard (migrated to pontifex.core.guard)."""

from .core.guard import sanitize_input_catalog, PHYSICAL_LIMITS

__all__ = ["sanitize_input_catalog", "PHYSICAL_LIMITS"]
