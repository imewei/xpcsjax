"""parameter_manager must derive all bounds from parameter_registry.

This guards against the historical bug where parameter_manager.py declared
its own contrast bounds that disagreed with the registry."""

from __future__ import annotations

import inspect
import re

from xpcsjax.config.parameter_manager import ParameterManager
from xpcsjax.config.parameter_registry import get_bounds, get_param_names

ANALYSIS_MODES = ("static", "static_isotropic", "laminar_flow")


def test_no_inline_bound_constants_in_manager() -> None:
    """parameter_manager.py source must not redeclare bounds — all bounds come from the registry.

    The predicate flags lines that pair a numeric literal with a ``"min"`` or
    ``"max"`` key (the historical bug pattern), while ignoring docstring
    examples and registry-derived lookups. Bounds-as-arguments / cache copies
    that operate on dicts (no inline numeric literal) are also ignored.
    """
    src = inspect.getsource(ParameterManager)

    # A literal bound assignment looks like ``"min": 0.0`` or ``"max": -1e5``;
    # match a quoted min/max key followed by a numeric literal.
    literal_bound_re = re.compile(
        r"""["']             # opening quote
            (?:min|max)      # key name
            ["']             # closing quote
            \s*:\s*          # colon separator
            -?               # optional sign
            \d+              # digits
            (?:\.\d*)?       # optional fraction
            (?:[eE][+-]?\d+)?  # optional exponent
        """,
        re.VERBOSE,
    )

    suspicious = []
    for line in src.splitlines():
        stripped = line.strip()
        # Skip docstring example lines.
        if stripped.startswith(">>>") or stripped.startswith("..."):
            continue
        # Skip lines that already defer to the registry.
        if "registry" in line.lower() or "get_bounds" in line:
            continue
        if literal_bound_re.search(line):
            suspicious.append(line)

    assert not suspicious, (
        "parameter_manager.py contains literal bound numerics — "
        "must derive bounds from the registry:\n" + "\n".join(suspicious)
    )


def test_manager_bounds_match_registry_for_all_modes() -> None:
    """For every (analysis_mode, param), manager bounds == registry bounds."""
    for mode in ANALYSIS_MODES:
        pm = ParameterManager(analysis_mode=mode)
        for name in get_param_names(mode):
            # Public API: get_bounds_as_tuples returns [(min, max), ...]
            mgr_bounds = pm.get_bounds_as_tuples([name])[0]
            reg_bounds = get_bounds(name)
            assert mgr_bounds == reg_bounds, (
                f"mismatch for {mode}.{name}: "
                f"manager={mgr_bounds}, registry={reg_bounds}"
            )
