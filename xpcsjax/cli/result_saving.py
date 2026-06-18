"""Compatibility shim -- result persistence moved to xpcsjax.service.persist.

Kept so existing CLI imports (``from xpcsjax.cli.result_saving import save_results``)
continue to resolve. New code should import from ``xpcsjax.service.persist``.
"""

from __future__ import annotations

from xpcsjax.service.persist import (
    _config_summary,  # noqa: F401 -- re-exported for tests/cli/test_debug_audit_2026_06_17.py
    _extract_parameters,  # noqa: F401 -- re-exported for tests/test_debug_audit_2026_06_18.py
    save_results,
    save_results_json,
    save_results_npz,
)

# Note: _config_summary and _extract_parameters are intentionally re-exported (tests
# import them) but are NOT part of the public surface, so they stay out of __all__.
__all__ = ["save_results", "save_results_json", "save_results_npz"]
