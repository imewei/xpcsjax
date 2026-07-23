"""Regression: explicit ``per_angle_mode="averaged"`` expands DOF like ``"auto"``.

``_effective_param_count_for_ooc`` gates the DOF-expansion branch (which feeds
the out-of-core covariance-scaling denominator) on the resolved token. It used to
test only the literal ``"auto"``, so an explicit ``"averaged"`` config fell
through to the compressed ``n_params`` count — under-expanding DOF and skewing
reported uncertainties. Elsewhere (``anti_degeneracy_controller`` /
``hybrid_streaming``) explicit ``"averaged"`` is already a first-class token
equivalent to resolved-auto-averaged; this pins that ``out_of_core`` agrees.
"""

from __future__ import annotations

from xpcsjax.optimization.nlsq.strategies.out_of_core import (
    _effective_param_count_for_ooc,
)


def test_explicit_averaged_matches_auto_expansion() -> None:
    n_phi, n_physical = 5, 7
    kw = dict(per_angle_scaling=True, n_params=n_physical + 2, n_phi=n_phi, n_physical=n_physical)
    expected = 2 * n_phi + n_physical  # expanded per-angle scaling count

    auto = _effective_param_count_for_ooc(anti_degeneracy_config={"per_angle_mode": "auto"}, **kw)
    averaged = _effective_param_count_for_ooc(
        anti_degeneracy_config={"per_angle_mode": "averaged"}, **kw
    )

    assert auto == expected
    assert averaged == expected, (
        "explicit per_angle_mode='averaged' must expand DOF identically to "
        f"'auto' (got {averaged}, expected {expected}); the compressed "
        f"n_params={n_physical + 2} would skew covariance scaling."
    )
