"""Regression tests for ``ResultBuilder`` (homodyne result_builder.py).

The builder's ``iterations`` field documents "Number of optimizer iterations"
and every sibling builder reads ``nit`` for it. A prior bug read ``nfev``
(function-evaluation count) instead, over-reporting iterations by a large
factor for trust-region solves (nfev >> nit).
"""

import numpy as np

from xpcsjax.optimization.nlsq.result_builder import ResultBuilder


def _builder(info: dict) -> ResultBuilder:
    return (
        ResultBuilder()
        .with_parameters(np.array([1.0, 2.0]))
        .with_covariance(np.eye(2))
        .with_data_size(100)
        .with_info(info)
    )


def test_iterations_uses_nit_not_nfev():
    """The iterations field must reflect optimizer iterations (nit), not nfev."""
    out = _builder({"nit": 5, "nfev": 137, "cost": 0.5}).build()
    assert out["iterations"] == 5, (
        f"iterations must come from nit (5), not nfev (137); got {out['iterations']}"
    )


def test_iterations_prefers_explicit_iterations_key_over_default():
    """With no nit, an explicit 'iterations' key is honored before defaulting."""
    out = _builder({"iterations": 9, "nfev": 200, "cost": 0.5}).build()
    assert out["iterations"] == 9


def test_iterations_defaults_to_zero_without_nfev_fallback():
    """With neither nit nor iterations, default to 0 — never fall back to nfev."""
    out = _builder({"nfev": 321, "cost": 0.5}).build()
    assert out["iterations"] == 0, (
        f"missing nit/iterations must default to 0, not nfev (321); got {out['iterations']}"
    )
