"""Structural parity tests: same mode taxonomy in homodyne and heterodyne.

These tests assert *shape* parity, not numerical parity. Different physics
=> different fit values; what must match is the parameter-vector structure
for the same (mode, n_physics, n_phi, K) tuple.

Heterodyne mode -> dim formula matches homodyne per
https://homodyne.readthedocs.io/en/latest/theory/anti_degeneracy.html
Parameter Count Summary table:

* ``constant``   : ``n_physics``                       (scaling frozen pre-fit)
* ``individual`` : ``n_physics + 2 * n_phi``           (free per-angle scaling)
* ``fourier``    : ``n_physics + 2 * (2K + 1)``        (truncated basis)

All three parametrizations pass. Explicit multi-angle ``individual`` mode is a
JOINT fit: the ``n_phi`` independent per-angle ``(contrast, offset)`` are packed
as the ``2 * n_phi`` scaling tail of the joint vector and optimized jointly with
physics via :func:`_fit_joint_multi_phi` (``FourierReparameterizer`` in
``"independent"`` mode), matching ``laminar_flow`` and upstream heterodyne. The
parameter dim is therefore ``n_physics + 2 * n_phi`` — the same as the old
sequential aggregate, but now from one consistent joint optimum (its parameters
reproduce its reported chi-squared). The sequential per-angle aggregate
(:func:`_aggregate_individual_results`, ``covariance_structure=
"block_diagonal_sequential"``) survives only as the ``config is None`` /
single-angle fallback.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

# The fixture builders live alongside the C-series return-shape tests. They are
# imported here (rather than copied) so the parity test stays in lock-step with
# the canonical model/c2-stack construction used by the rest of Sub-PR C.
from tests.optimization.test_heterodyne_return_shape import (
    _C2_N_TIMES,
    _build_minimal_heterodyne_model_for_fourier,
    _build_synthetic_c2_stack_for_fourier,
)
from xpcsjax.optimization.nlsq.results import OptimizationResult


@pytest.mark.parametrize(
    "mode,scaling_dim_for",
    [
        ("constant", lambda n_phys, n_phi, K: 0),
        ("individual", lambda n_phys, n_phi, K: 2 * n_phi),
        ("fourier", lambda n_phys, n_phi, K: 2 * (2 * K + 1)),
    ],
)
def test_heterodyne_param_dim_matches_homodyne_formula(
    mode: str,
    scaling_dim_for: Callable[[int, int, int], int],
) -> None:
    """Heterodyne parameter dim = ``n_physics + (mode-specific scaling dim)``.

    Locks in the parity contract per homodyne anti-degeneracy docs Parameter
    Count Summary. If a future change shifts the parameter packing layout,
    this test catches it.
    """
    pytest.importorskip("xpcsjax.core.heterodyne_model_stateful")
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

    model = _build_minimal_heterodyne_model_for_fourier()
    n_physics = model.param_manager.n_varying
    n_phi = 6  # full angle set — large enough to trigger the fourier window
    K = 2

    config = NLSQConfig(per_angle_mode=mode, fourier_order=K, max_nfev=30)
    c2 = _build_synthetic_c2_stack_for_fourier(
        n_phi=n_phi, n_t=_C2_N_TIMES, model=model
    )
    phi = np.linspace(0.0, 150.0, n_phi, dtype=np.float64)

    result = fit_nlsq_multi_phi(model, c2, phi, config, weights=None)
    assert isinstance(result, OptimizationResult), (
        f"mode={mode!r}: expected OptimizationResult, got {type(result).__name__}"
    )

    scaling_dim = scaling_dim_for(n_physics, n_phi, K)
    expected_dim = n_physics + scaling_dim
    assert result.parameters.shape == (expected_dim,), (
        f"mode={mode!r}: expected dim {expected_dim} "
        f"(n_physics={n_physics} + scaling_dim={scaling_dim}), "
        f"got {result.parameters.shape[0]}"
    )
