"""Guard test: no direct ``scipy.optimize.least_squares`` in the NLSQ path.

Architectural boundary: xpcsjax v0.1 is JAX-native NLSQ — the optimizer of
record is ``nlsq.CurveFit`` (JAX-accelerated trust-region) reached via
``xpcsjax.optimization.nlsq.heterodyne_adapter.NLSQAdapter``.  Direct calls
to ``scipy.optimize.least_squares`` are scheduled-for-removal in the
v0.1 cleanup and must not re-enter the optimizer path.

This guard mirrors heterodyne's ``tests/unit/optimization/nlsq/test_no_scipy.py``
adapted for xpcsjax's module layout.  Identified as a parity gap by the
2026-05-22 Codex/Gemini test-suite audit.

Allowed:
- ``from scipy.optimize import OptimizeResult`` (type / return-shape import,
  used by ``heterodyne_result_builder.py``).
- ``scipy.optimize.curve_fit`` references in docstrings (xpcsjax delegates
  to nlsq.CurveFit which has a curve_fit-shaped API, so prose mentions are
  expected).

Forbidden:
- ``from scipy.optimize import least_squares``
- ``scipy.optimize.least_squares(...)`` call
- ``least_squares(...)`` call (bare-name)
- ``class ScipyNLSQAdapter`` (the homodyne-era fallback that v0.1 retired)
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Files that must NOT call or import scipy.optimize.least_squares.  This is
# the v0.1 NLSQ surface — any file that owns optimizer entry/dispatch.
NLSQ_FILES = [
    "xpcsjax/optimization/nlsq/heterodyne_adapter.py",
    "xpcsjax/optimization/nlsq/heterodyne_core.py",
    "xpcsjax/optimization/nlsq/heterodyne_constant_mode.py",
    "xpcsjax/optimization/nlsq/adapter.py",
    "xpcsjax/optimization/nlsq/core.py",
    "xpcsjax/optimization/nlsq/wrapper.py",
    "xpcsjax/optimization/nlsq/fallback_chain.py",
    "xpcsjax/optimization/nlsq/cmaes_wrapper.py",
    "xpcsjax/optimization/nlsq/hierarchical.py",
    "xpcsjax/optimization/nlsq/strategies/residual.py",
    "xpcsjax/optimization/nlsq/strategies/residual_jit.py",
    "xpcsjax/optimization/nlsq/strategies/stratified_ls.py",
    "xpcsjax/optimization/nlsq/strategies/hybrid_streaming.py",
    "xpcsjax/optimization/nlsq/strategies/out_of_core.py",
]


class TestNoScipyLeastSquares:
    """Verify scipy.optimize.least_squares is absent from the NLSQ path."""

    @pytest.mark.parametrize("filepath", NLSQ_FILES)
    def test_no_scipy_least_squares_import(self, filepath: str) -> None:
        """No NLSQ-path file imports ``scipy.optimize.least_squares``."""
        path = PROJECT_ROOT / filepath
        if not path.exists():
            pytest.skip(f"{filepath} not present (optional / future file)")
        source = path.read_text(encoding="utf-8")
        # Both styles are forbidden:
        #   from scipy.optimize import least_squares
        #   from scipy.optimize import least_squares as anything_else
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "scipy.optimize":
                names = {alias.name for alias in node.names}
                assert "least_squares" not in names, (
                    f"{filepath} imports scipy.optimize.least_squares — "
                    f"xpcsjax v0.1 is JAX-native, use NLSQAdapter / nlsq.CurveFit"
                )

    @pytest.mark.parametrize("filepath", NLSQ_FILES)
    def test_no_scipy_least_squares_call(self, filepath: str) -> None:
        """No NLSQ-path file calls scipy's ``least_squares(...)``.

        Distinguishes ``scipy.optimize.least_squares`` (banned) from
        ``ls.least_squares(...)`` where ``ls`` is an instance of
        ``nlsq.LeastSquares`` (xpcsjax's JAX-native class, allowed).  The
        attribute-name match alone is too coarse — we walk the chain.
        """
        path = PROJECT_ROOT / filepath
        if not path.exists():
            pytest.skip(f"{filepath} not present (optional / future file)")
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Collect bare-name aliases pointing at scipy.optimize.least_squares.
        # Only these aliases turn a bare ``least_squares(...)`` call into a
        # scipy call; with ``from nlsq import LeastSquares`` etc. the bare
        # name doesn't reach scipy.
        scipy_aliases: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "scipy.optimize":
                for alias in node.names:
                    if alias.name == "least_squares":
                        scipy_aliases.add(alias.asname or alias.name)

        def _attr_chain(node: ast.AST) -> list[str]:
            """Return the dotted-name chain rooted at the leftmost Name.

            ``scipy.optimize.least_squares`` -> ['scipy', 'optimize',
            'least_squares']; ``ls.least_squares`` -> ['ls',
            'least_squares']; anything else -> [].
            """
            parts: list[str] = []
            cur = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
                return list(reversed(parts))
            return []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # Bare-name call: least_squares(...) — only flag if the bare
            # name was aliased from scipy in this file.
            if isinstance(func, ast.Name) and func.id in scipy_aliases:
                pytest.fail(
                    f"{filepath}:{getattr(node, 'lineno', '?')} "
                    f"calls scipy-aliased {func.id}() directly — route via NLSQAdapter"
                )
            # Attribute call: only flag chains that resolve to scipy.optimize.
            if isinstance(func, ast.Attribute) and func.attr == "least_squares":
                chain = _attr_chain(func)
                # scipy.optimize.least_squares  OR  optimize.least_squares
                # (where 'optimize' was imported via ``from scipy import
                # optimize``).  Anything else (e.g. ``ls.least_squares``)
                # is allowed.
                is_scipy_chain = (
                    chain[:2] == ["scipy", "optimize"]
                    or chain[:1] == ["optimize"]  # less common; conservative
                )
                if is_scipy_chain:
                    pytest.fail(
                        f"{filepath}:{getattr(node, 'lineno', '?')} "
                        f"calls scipy.optimize.least_squares() — route via NLSQAdapter"
                    )

    def test_no_scipy_nlsq_adapter_class(self) -> None:
        """ScipyNLSQAdapter (the retired fallback) must not reappear in adapter.py."""
        candidate = PROJECT_ROOT / "xpcsjax/optimization/nlsq/heterodyne_adapter.py"
        assert candidate.exists()
        source = candidate.read_text(encoding="utf-8")
        assert "class ScipyNLSQAdapter" not in source, (
            "ScipyNLSQAdapter is the homodyne-era scipy fallback retired in "
            "v0.1; do not reintroduce."
        )
