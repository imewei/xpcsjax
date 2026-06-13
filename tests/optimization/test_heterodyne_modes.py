"""Tests for heterodyne per-angle mode vocabulary parity with homodyne."""

from __future__ import annotations

import pytest

from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig


def test_individual_mode_accepted() -> None:
    """`individual` is the canonical name (matches homodyne docs)."""
    cfg = NLSQConfig(per_angle_mode="individual")
    assert cfg.per_angle_mode == "individual"
    errors = cfg.validate()
    assert errors == [], f"expected no validation errors, got {errors}"


def test_independent_deprecation_alias() -> None:
    """`independent` maps to `individual` with a DeprecationWarning that points at the user's call site."""
    with pytest.warns(DeprecationWarning, match=r"'independent' is deprecated") as records:
        cfg = NLSQConfig(per_angle_mode="independent")  # type: ignore[arg-type]
    assert cfg.per_angle_mode == "individual"
    assert len(records) == 1
    # stacklevel should point at this test file, not dataclass-synthesized <string> code
    assert records[0].filename.endswith("test_heterodyne_modes.py"), (
        f"DeprecationWarning fired at {records[0].filename}:{records[0].lineno} — "
        "expected to point at user call site (stacklevel issue?)"
    )


def test_averaged_function_renamed() -> None:
    """The averaged-scaling joint solver uses the corrected name."""
    from xpcsjax.optimization.nlsq import heterodyne_core

    assert hasattr(heterodyne_core, "_fit_joint_averaged_multi_phi"), "expected renamed function"
    assert not hasattr(heterodyne_core, "_fit_joint_constant_multi_phi"), (
        "old mislabeled name must be removed — "
        "true 'constant' mode lands in Sub-PR B with its own dedicated function"
    )


def test_constant_mode_dispatches_to_constant_fit() -> None:
    """`per_angle_mode='constant'` reaches `_fit_joint_constant_multi_phi`."""
    from unittest.mock import patch

    import numpy as np

    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

    config = NLSQConfig(per_angle_mode="constant")

    # Use a stub model — dispatch test doesn't run the fit body, just verifies
    # the dispatch reaches the right function.
    class _StubModel:
        pass

    model = _StubModel()
    c2 = np.zeros((2, 8, 8))
    phi = np.array([0.0, 45.0])

    with patch(
        "xpcsjax.optimization.nlsq.heterodyne_constant_mode._fit_joint_constant_multi_phi"
    ) as mock_fit:
        sentinel = object()
        mock_fit.return_value = sentinel
        result = fit_nlsq_multi_phi(model, c2, phi, config, weights=None)  # type: ignore[arg-type]

    assert result is sentinel, "dispatch did not reach constant-mode fit"
    mock_fit.assert_called_once()


def test_auto_with_small_n_phi_uses_individual() -> None:
    """`auto` with n_phi < constant_scaling_threshold resolves to individual.

    Unified rule (matches the homodyne AntiDegeneracyController): auto never
    selects constant — few angles get per-angle individual scaling.
    """
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import _resolve_effective_mode

    config = NLSQConfig(per_angle_mode="auto", constant_scaling_threshold=3)
    assert _resolve_effective_mode(config, 1) == "individual"
    assert _resolve_effective_mode(config, 2) == "individual"


def test_auto_with_mid_n_phi_uses_averaged() -> None:
    """`auto` mode with constant_threshold <= n_phi < fourier_auto_threshold dispatches averaged."""
    from unittest.mock import patch

    import numpy as np

    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

    config = NLSQConfig(
        per_angle_mode="auto",
        constant_scaling_threshold=3,
        fourier_auto_threshold=6,
    )

    class _StubModel:
        pass

    model = _StubModel()
    c2 = np.zeros((4, 8, 8))  # n_phi = 4, in averaged window
    phi = np.linspace(0, 135, 4)

    with patch(
        "xpcsjax.optimization.nlsq.heterodyne_core._fit_joint_averaged_multi_phi"
    ) as mock_avg:
        mock_avg.return_value = "sentinel"
        fit_nlsq_multi_phi(model, c2, phi, config, weights=None)  # type: ignore[arg-type]

    mock_avg.assert_called_once()


def test_auto_with_large_n_phi_uses_averaged() -> None:
    """`auto` with large n_phi dispatches averaged, NOT fourier.

    Unified rule: auto never auto-selects fourier (``fourier_auto_threshold``
    has no effect under auto). Even at n_phi=8 (>= the old fourier threshold),
    auto resolves to averaged; fourier must be requested explicitly.
    """
    from unittest.mock import patch

    import numpy as np

    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

    config = NLSQConfig(
        per_angle_mode="auto",
        constant_scaling_threshold=3,
        fourier_auto_threshold=6,
    )

    class _StubModel:
        pass

    model = _StubModel()
    c2 = np.zeros((8, 8, 8))  # n_phi = 8, at/above the OLD fourier threshold
    phi = np.linspace(0, 157.5, 8)

    with patch(
        "xpcsjax.optimization.nlsq.heterodyne_core._fit_joint_averaged_multi_phi"
    ) as mock_avg:
        mock_avg.return_value = "sentinel"
        fit_nlsq_multi_phi(model, c2, phi, config, weights=None)  # type: ignore[arg-type]

    mock_avg.assert_called_once()


def test_auto_never_selects_constant_or_fourier() -> None:
    """Unified rule: auto ∈ {individual, averaged} for ALL n_phi; explicit
    constant/fourier/individual pass through unchanged."""
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import _resolve_effective_mode

    auto = NLSQConfig(per_angle_mode="auto", constant_scaling_threshold=3, fourier_auto_threshold=6)
    for n_phi in (1, 2, 3, 4, 5, 6, 10, 23, 100):
        mode = _resolve_effective_mode(auto, n_phi)
        assert mode in ("individual", "averaged"), (
            f"auto must never select {mode!r} (n_phi={n_phi})"
        )
        assert mode == ("individual" if n_phi < 3 else "averaged")

    # Explicit modes are honored regardless of n_phi.
    for explicit in ("constant", "fourier", "individual"):
        cfg = NLSQConfig(per_angle_mode=explicit)
        assert _resolve_effective_mode(cfg, 8) == explicit


def test_no_build_joint_fourier_symbol() -> None:
    """The legacy in-memory Fourier joint builders are deleted (Task 10).

    ``_build_joint_fourier`` (reparameterizer factory) and
    ``_build_joint_problem_fourier`` (physics-first problem builder) are the two
    in-memory Fourier-arm symbols torn down here. Both must be gone from
    ``heterodyne_core`` after the teardown.
    """
    import xpcsjax.optimization.nlsq.heterodyne_core as hc

    assert not hasattr(hc, "_build_joint_fourier")
    assert not hasattr(hc, "_build_joint_problem_fourier")


def test_resolver_rejects_fourier_in_dispatch() -> None:
    """In-memory ``fit_nlsq_multi_phi`` rejects a fourier-resolved mode (Task 10).

    The in-memory joint path no longer builds a Fourier problem; a fourier
    resolved mode raises ``ValueError`` at the dispatch site (the shared
    ``_resolve_effective_mode`` still RETURNS ``"fourier"`` for the
    stratified-LS / streaming paths, which is out of scope here).
    """
    import numpy as np
    import pytest

    from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

    # n_phi=4 (>= constant_scaling_threshold) so an explicit ``fourier`` request
    # resolves to ``"fourier"`` and reaches the in-memory dispatch's reject path.
    model, c2, phi = make_synthetic_two_component(n_phi=4, n_t=10)
    cfg = NLSQConfig(per_angle_mode="fourier")
    with pytest.raises(ValueError, match="unknown per_angle_mode"):
        fit_nlsq_multi_phi(model, np.asarray(c2), np.asarray(phi), cfg, None)
