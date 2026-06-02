"""Regression tests for per-angle CMA-ES diagnostics + warm-start auto-skip.

Two parity items closed alongside this file:

1. **``global_escape`` diagnostics symmetry.** The joint escapes tag
   ``nlsq_diagnostics["global_escape"]``; the per-angle ``_fit_cmaes`` path only
   set ``metadata["optimizer"]`` / ``["cmaes_winner"]``. The per-angle path now
   also tags ``metadata["global_escape"]`` (the only field aggregated into
   ``nlsq_diagnostics["per_angle_metadata"]``), mirroring the joint values:
   ``"cmaes"`` when CMA-ES won, ``"cmaes_warmstart_kept"`` when it ran but the
   NLSQ warm-start was kept, ``"cmaes_warmstart_auto_skip"`` when it was gated
   off.

2. **Warm-start auto-skip parity.** ``cmaes_warmstart_auto_skip`` /
   ``cmaes_warmstart_skip_threshold`` were honored only on laminar_flow's
   ``core.py`` path; the heterodyne per-angle path ignored them and always paid
   for the full global search. ``_fit_cmaes`` now skips CMA-ES when the NLSQ
   warm-start's reduced χ² is below threshold, mirroring homodyne
   ``core.py:2296-2362``.
"""

from __future__ import annotations

import pytest

from xpcsjax.optimization.nlsq import heterodyne_core
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_core import _fit_cmaes

from ._heterodyne_fixtures import make_synthetic_two_component


def test_warmstart_auto_skip_gates_off_cmaes() -> None:
    """A good NLSQ warm-start (reduced χ² < threshold) skips CMA-ES entirely.

    With a deliberately huge threshold (1e9), any finite warm-start reduced χ²
    qualifies, so CMA-ES must not run at all. We assert that by monkeypatching
    ``fit_with_cmaes`` to explode if it is ever reached — which also means this
    test needs no evosax backend.
    """
    import pytest as _pytest

    def _explode(**_kwargs):
        raise AssertionError(
            "fit_with_cmaes must NOT be called when warm-start auto-skip fires"
        )

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(heterodyne_core, "fit_with_cmaes", _explode)

        model, c2, phi = make_synthetic_two_component(n_phi=1, n_t=10)
        config = NLSQConfig(
            enable_cmaes=True,
            cmaes_warmstart_auto_skip=True,
            cmaes_warmstart_skip_threshold=1e9,  # any finite reduced χ² qualifies
        )
        result = _fit_cmaes(
            model, c2[0], float(phi[0]), config, weights=None, angle_idx=0
        )

    assert result.success, "warm-start must succeed for auto-skip to be meaningful"
    assert result.metadata.get("cmaes_skipped") is True
    assert result.metadata.get("cmaes_winner") == "nlsq_warmstart_auto_skip"
    assert result.metadata.get("optimizer") == "cmaes"
    assert result.metadata.get("global_escape") == "cmaes_warmstart_auto_skip", (
        "auto-skip must surface the global_escape diagnostics tag for symmetry "
        f"with the joint escapes; got {result.metadata.get('global_escape')!r}"
    )


def test_warmstart_auto_skip_disabled_runs_cmaes() -> None:
    """``cmaes_warmstart_auto_skip=False`` must NOT gate CMA-ES off.

    Even with a huge threshold, disabling auto-skip means the global search
    runs. We prove the gate is bypassed by asserting ``fit_with_cmaes`` IS
    reached (the spy raises a sentinel we catch), without needing evosax.
    """

    class _ReachedError(Exception):
        pass

    def _spy(**_kwargs):
        raise _ReachedError

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(heterodyne_core, "fit_with_cmaes", _spy)

        model, c2, phi = make_synthetic_two_component(n_phi=1, n_t=10)
        config = NLSQConfig(
            enable_cmaes=True,
            cmaes_warmstart_auto_skip=False,
            cmaes_warmstart_skip_threshold=1e9,
        )
        with pytest.raises(_ReachedError):
            _fit_cmaes(model, c2[0], float(phi[0]), config, weights=None, angle_idx=0)


@pytest.mark.skipif(
    not getattr(heterodyne_core, "HAS_CMAES", False),
    reason="CMA-ES backend (evosax) not installed; full search path not testable",
)
def test_global_escape_tag_mirrors_winner_when_search_runs() -> None:
    """When CMA-ES actually runs, the per-angle result carries a ``global_escape``
    tag whose value is consistent with the Phase-3 winner.

    auto-skip is disabled so Phase 2/3 run; the tag is "cmaes" iff CMA-ES won,
    else "cmaes_warmstart_kept" — the per-angle mirror of the joint escape's
    "<kind>" / "<kind>_warmstart_kept" convention.
    """
    model, c2, phi = make_synthetic_two_component(n_phi=1, n_t=10)
    config = NLSQConfig(
        enable_cmaes=True,
        cmaes_warmstart_auto_skip=False,
        cmaes_max_iterations=15,
        cmaes_population_size=6,
        cmaes_restart_strategy="none",
        cmaes_max_restarts=0,
    )
    result = _fit_cmaes(model, c2[0], float(phi[0]), config, weights=None, angle_idx=0)

    assert "cmaes_skipped" not in result.metadata, (
        "auto-skip was disabled; the result must not be tagged as skipped"
    )
    tag = result.metadata.get("global_escape")
    assert tag in {"cmaes", "cmaes_warmstart_kept"}, (
        f"per-angle CMA-ES result must carry a global_escape tag; got {tag!r}"
    )
    winner = result.metadata.get("cmaes_winner")
    expected = "cmaes" if winner == "cmaes" else "cmaes_warmstart_kept"
    assert tag == expected, (
        f"global_escape ({tag!r}) must mirror the Phase-3 winner "
        f"({winner!r} → expected {expected!r})"
    )


@pytest.mark.skipif(
    not getattr(heterodyne_core, "HAS_CMAES", False),
    reason="entering _fit_cmaes via the public path is gated on HAS_CMAES",
)
def test_global_escape_surfaces_in_per_angle_metadata() -> None:
    """The ``global_escape`` tag must survive aggregation into
    ``nlsq_diagnostics["per_angle_metadata"]`` (which copies each result's
    ``.metadata``).

    Routed through the public ``fit_nlsq_multi_phi`` with ``n_phi=1`` — the
    only layout that reaches the *per-angle* ``_fit_cmaes`` (multi-angle +
    ``enable_cmaes`` goes to the JOINT escape instead). Entering ``_fit_cmaes``
    requires ``HAS_CMAES`` even though auto-skip (threshold 1e9) then gates off
    the actual evosax search — so ``fit_with_cmaes`` is patched to explode as a
    belt-and-suspenders check that the search never runs.
    """
    from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

    model, c2, phi = make_synthetic_two_component(n_phi=1, n_t=10)
    config = NLSQConfig(
        enable_cmaes=True,
        cmaes_warmstart_auto_skip=True,
        cmaes_warmstart_skip_threshold=1e9,
    )

    def _explode(**_kwargs):
        raise AssertionError("CMA-ES must be auto-skipped in this fixture")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(heterodyne_core, "fit_with_cmaes", _explode)
        result = fit_nlsq_multi_phi(model, c2, phi, config, weights=None)

    diag = result.nlsq_diagnostics or {}
    per_angle = diag.get("per_angle_metadata")
    assert isinstance(per_angle, list) and len(per_angle) == 1
    assert per_angle[0].get("global_escape") == "cmaes_warmstart_auto_skip", (
        "per-angle global_escape tag must propagate into "
        f"nlsq_diagnostics['per_angle_metadata']; got {per_angle[0].get('global_escape')!r}"
    )
