"""Live A/B parity gate: xpcsjax vs the upstream homodyne NLSQ fit, rtol=1e-10.

Unlike ``test_homodyne_equivalence`` (which compares xpcsjax against a *frozen
JSON baseline*), this runs the upstream ``homodyne`` fit **live in the same
process** on the same config, then asserts the two fits agree on fitted
parameters and chi-squared to rtol=1e-10. Running both sides live:

* eliminates baseline staleness as a variable (no regeneration step), and
* catches drift on *either* side — an upstream change is caught too.

Constraints honored:

* ``homodyne`` is NOT an xpcsjax dependency (CLAUDE.md). The test
  ``importorskip``s it, so normal installs skip cleanly.
* The fits are slow (the laminar config runs CMA-ES, ~6 min) and need the
  registered datasets on disk, so the suite is gated behind
  ``XPCSJAX_RUN_AB_PARITY=1`` and skips configs whose path is absent.

Run manually::

    XPCSJAX_RUN_AB_PARITY=1 uv run pytest tests/characterization/test_homodyne_nlsq_ab_parity.py -v

The config registry and result-extraction helpers are shared with
``test_homodyne_equivalence`` so the two parity gates cannot drift apart.
"""

from __future__ import annotations

import gc
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

# Reuse the single registry of (label -> config path) and the result-shape
# adapters so this gate and the frozen-baseline gate stay in lockstep.
from tests.characterization.test_homodyne_equivalence import (
    CONFIGS,
    _extract_chi_squared,
    _extract_params,
)

_NO_WORSE_REL = 1e-3  # ~1e-3 no-worse band (CLAUDE.md dual-gate / engine-unification)


@pytest.fixture(autouse=True)
def _release_jax_memory_between_fits():
    """Release JAX caches + fit buffers after each test so the module's multiple
    heavy ~23M-point C020 fits (strict A/B + the DG-D no-worse sibling, each
    running upstream + xpcsjax) do not accumulate RAM and OOM the run."""
    yield
    try:
        import jax

        jax.clear_caches()
    except Exception:  # pragma: no cover - best-effort hygiene only
        pass
    gc.collect()


GATE_OPT_IN = os.environ.get("XPCSJAX_RUN_AB_PARITY") == "1"
_SKIP_REASON = (
    "Live A/B parity gate is slow (laminar runs CMA-ES, ~6 min) and requires "
    "the upstream `homodyne` package plus the registered datasets on disk. "
    "Set XPCSJAX_RUN_AB_PARITY=1 to run."
)


def _available_labels() -> list[str]:
    """Registered labels whose config path currently exists on disk."""
    return sorted(label for label, path in CONFIGS.items() if Path(path).exists())


@pytest.mark.skipif(not GATE_OPT_IN, reason=_SKIP_REASON)
def test_homodyne_importable_when_ab_gate_active() -> None:
    """With the gate on, upstream ``homodyne`` MUST be importable.

    Mirrors the env-guard in ``test_homodyne_equivalence``: closes the
    "parametrize over empty list silently passes" trap, since a live A/B with
    no upstream package would otherwise collect zero cases and report green
    without ever comparing anything.
    """
    if importlib.util.find_spec("homodyne") is None:
        pytest.fail(
            "XPCSJAX_RUN_AB_PARITY=1 but upstream `homodyne` is not importable. "
            "Install it into the dev env (`uv pip install -e /path/to/homodyne`) "
            "— the A/B gate compares against homodyne's own live fit."
        )


@pytest.mark.skipif(not GATE_OPT_IN, reason=_SKIP_REASON)
@pytest.mark.parametrize("label", _available_labels())
def test_homodyne_nlsq_live_ab_parity(label: str) -> None:
    """xpcsjax and upstream homodyne must produce identical fits (rtol=1e-10)."""
    pytest.importorskip("homodyne")

    config_path = CONFIGS[label]
    assert Path(config_path).exists(), (
        f"{label}: registered config path is dead — {config_path} not found."
    )

    # Lazy imports — keep JAX/homodyne off the collection path so a skipped run
    # pays nothing. Both packages expose the same trio of entry points; calling
    # the matching one on each side is the whole point of the A/B.
    # homodyne is intentionally not a declared dependency (CLAUDE.md), so static
    # type checkers can't resolve it; the importorskip above guards runtime.
    import homodyne.config  # type: ignore[import-not-found]
    import homodyne.data  # type: ignore[import-not-found]
    import homodyne.optimization  # type: ignore[import-not-found]

    from xpcsjax.data import load_xpcs_data as x_load_xpcs_data
    from xpcsjax.optimization.nlsq import fit_nlsq_jax as x_fit_nlsq_jax

    # --- upstream homodyne (the oracle) ---
    h_cfg = homodyne.config.ConfigManager(str(config_path))
    h_data = homodyne.data.load_xpcs_data(str(config_path))
    h_result = homodyne.optimization.fit_nlsq_jax(h_data, h_cfg)

    # --- xpcsjax (must mirror it) ---
    from xpcsjax.config import ConfigManager as XpcsConfigManager

    x_cfg = XpcsConfigManager(str(config_path))
    # DG-D strict half: pin EXPLICIT per_angle_mode='individual' so xpcsjax stays
    # bit-identical to upstream homodyne's individual fit after the Phase-5 default
    # flip (auto->averaged@n_phi>=3); upstream has no such flip, so the explicit
    # token keeps both sides on the same scaling-first individual layout. The
    # default change is gated by the no-worse sibling below. Static labels keep
    # individual either way (Finding 8).
    x_cfg.config.setdefault("optimization", {}).setdefault("nlsq", {}).setdefault(
        "anti_degeneracy", {}
    ).update(
        {
            "enable": True,
            "per_angle_mode": "individual",
            "hierarchical": {"enable": False},
            "regularization": {"enable": False},
        }
    )
    x_data = x_load_xpcs_data(str(config_path))
    x_result = x_fit_nlsq_jax(x_data, x_cfg)

    x_params = np.asarray(_extract_params(x_result), dtype=np.float64)
    h_params = np.asarray(_extract_params(h_result), dtype=np.float64)

    assert x_params.shape == h_params.shape, (
        f"{label}: parameter-count mismatch (xpcsjax={x_params.shape}, homodyne={h_params.shape})"
    )

    np.testing.assert_allclose(
        x_params,
        h_params,
        rtol=1e-10,
        err_msg=f"{label}: fitted parameters diverged between xpcsjax and upstream homodyne",
    )

    np.testing.assert_allclose(
        float(_extract_chi_squared(x_result)),
        float(_extract_chi_squared(h_result)),
        rtol=1e-10,
        err_msg=f"{label}: chi_squared diverged between xpcsjax and upstream homodyne",
    )


@pytest.mark.skipif(not GATE_OPT_IN, reason=_SKIP_REASON)
@pytest.mark.parametrize("label", _available_labels())
def test_homodyne_nlsq_default_no_worse(label: str) -> None:
    """DG-D no-worse half: xpcsjax's Phase-5 DEFAULT (`auto` -> averaged@n_phi>=3)
    fit must be no-worse-SSR vs upstream homodyne's individual fit within ~1e-3.

    Upstream homodyne (individual) is the reference. averaged is MORE constrained,
    so xpcsjax's default chi2 can only degrade-or-equal; the degradation is the
    INTENDED Phase-5 default change, not a divergence (spec Risk 2). Static labels
    are gated OUT of Phase 5 (auto->averaged is a no-op there; Finding 8) and skipped.
    """
    pytest.importorskip("homodyne")

    config_path = CONFIGS[label]
    assert Path(config_path).exists(), (
        f"{label}: registered config path is dead — {config_path} not found."
    )

    from xpcsjax.config import ConfigManager as XpcsConfigManager

    probe = XpcsConfigManager(str(config_path))
    if str(probe.config.get("analysis_mode", "")).startswith("static"):
        pytest.skip(f"{label}: static mode is deferred (auto->averaged is a no-op)")

    import homodyne.config  # type: ignore[import-not-found]
    import homodyne.data  # type: ignore[import-not-found]
    import homodyne.optimization  # type: ignore[import-not-found]

    from xpcsjax.data import load_xpcs_data as x_load_xpcs_data
    from xpcsjax.optimization.nlsq import fit_nlsq_jax as x_fit_nlsq_jax

    # upstream homodyne individual (the oracle / reference SSR)
    h_cfg = homodyne.config.ConfigManager(str(config_path))
    h_data = homodyne.data.load_xpcs_data(str(config_path))
    h_result = homodyne.optimization.fit_nlsq_jax(h_data, h_cfg)
    chi2_homodyne = float(_extract_chi_squared(h_result))

    # xpcsjax DEFAULT auto (-> averaged for n_phi >= 3)
    x_cfg = XpcsConfigManager(str(config_path))
    x_cfg.config.setdefault("optimization", {}).setdefault("nlsq", {}).setdefault(
        "anti_degeneracy", {}
    ).update({"enable": True, "per_angle_mode": "auto"})
    x_data = x_load_xpcs_data(str(config_path))
    x_result = x_fit_nlsq_jax(x_data, x_cfg)
    chi2_xpcsjax_default = float(_extract_chi_squared(x_result))

    assert chi2_xpcsjax_default <= chi2_homodyne * (1.0 + _NO_WORSE_REL) + 1e-9, (
        f"{label}: xpcsjax default auto/averaged chi2 {chi2_xpcsjax_default} worse "
        f"than upstream homodyne individual {chi2_homodyne} beyond no-worse band"
    )
