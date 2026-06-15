"""Regression: explicit ``per_angle_mode="averaged"`` is accepted on every
laminar internal path, symmetric with the standard wrapper path.

Before this fix the standard wrapper path resolved per-angle modes through the
shared ``resolve_per_angle_mode`` seam (which returns the explicit ``"averaged"``
token verbatim), while two laminar internal resolvers — the
``AntiDegeneracyController`` (also on the laminar CMA-ES path) and the
hybrid-streaming driver — used inline ``if/elif/else`` blocks that *raised*
``ValueError("unknown per_angle_mode 'averaged'")`` for the very token the
resolver emits. That was a genuine intra-laminar inconsistency: an
``anti_degeneracy.per_angle_mode="averaged"`` config that the standard path
accepts crashed the controller / streaming paths.

Both internal resolvers now route through ``resolve_per_angle_mode``, so the
canonical token set (auto / constant / individual / averaged) is handled
uniformly everywhere. ``"averaged"`` stays an internal/config-resolved variant
(the config dataclass user-token surface remains ``individual``/``constant``/
``auto`` by design — users write ``auto``); this test only pins that the
internal resolvers no longer diverge on it.
"""

import ast
import inspect

import numpy as np

from xpcsjax.optimization.nlsq.anti_degeneracy_controller import (
    AntiDegeneracyController,
)
from xpcsjax.optimization.nlsq.per_angle_mode import resolve_per_angle_mode


def test_resolver_passes_through_explicit_averaged():
    """The shared seam returns ``"averaged"`` verbatim (the contract the inline
    paths previously violated)."""
    assert resolve_per_angle_mode("averaged", n_phi=5) == "averaged"


def test_controller_accepts_explicit_averaged():
    """The anti-degeneracy controller (standard + CMA-ES laminar paths) must
    accept an explicit ``"averaged"`` token without raising, resolving it to the
    ``averaged`` actual mode."""
    n_phi, n_physical = 5, 7
    phi_angles = np.linspace(0.0, 90.0, n_phi)
    controller = AntiDegeneracyController.from_config(
        {"enable": True, "per_angle_mode": "averaged"},
        n_phi=n_phi,
        phi_angles=phi_angles,
        n_physical=n_physical,
        per_angle_scaling=True,
        is_laminar_flow=True,
    )
    assert controller.per_angle_mode_actual == "averaged"


def test_controller_existing_tokens_unchanged():
    """The seam refactor must not change the resolved actual mode for the
    pre-existing tokens (auto/constant/individual)."""
    n_phi, n_physical = 5, 7
    phi_angles = np.linspace(0.0, 90.0, n_phi)

    def resolved(mode: str) -> str:
        c = AntiDegeneracyController.from_config(
            {"enable": True, "per_angle_mode": mode},
            n_phi=n_phi,
            phi_angles=phi_angles,
            n_physical=n_physical,
            per_angle_scaling=True,
            is_laminar_flow=True,
        )
        return c.per_angle_mode_actual

    # auto at n_phi (5) >= threshold (3) -> averaged
    assert resolved("auto") == "averaged"
    assert resolved("constant") == "constant"
    assert resolved("individual") == "individual"


def test_controller_rejects_unknown_token():
    """A truly unknown / removed-legacy token must still raise (via the shared
    resolver's ``else`` branch)."""
    import pytest

    with pytest.raises(ValueError, match="unknown per_angle_mode"):
        AntiDegeneracyController.from_config(
            {"enable": True, "per_angle_mode": "definitely_not_a_mode"},
            n_phi=5,
            phi_angles=np.linspace(0.0, 90.0, 5),
            n_physical=7,
            per_angle_scaling=True,
            is_laminar_flow=True,
        )


def _source_of(func) -> str:
    return inspect.getsource(func)


def test_hybrid_streaming_routes_through_shared_resolver():
    """The laminar hybrid-streaming driver must route per-angle mode resolution
    through ``resolve_per_angle_mode`` and must NOT contain its own
    ``unknown per_angle_mode`` reject block (which used to crash on an explicit
    ``"averaged"`` token the standard path accepts)."""
    from xpcsjax.optimization.nlsq.strategies import hybrid_streaming as hs

    src = _source_of(hs.fit_with_stratified_hybrid_streaming)

    # Must call the shared static-pin seam (``_resolve_streaming_per_angle_mode``
    # wraps ``resolve_per_angle_mode`` and enforces the static-individual pin).
    assert "_resolve_streaming_per_angle_mode(" in src, (
        "hybrid-streaming driver no longer routes through the shared "
        "_resolve_streaming_per_angle_mode seam"
    )

    # Must not re-raise the inline 'unknown per_angle_mode' that rejected
    # 'averaged'. Parse the call's body and assert no ValueError raise mentions
    # 'unknown per_angle_mode' as a string constant.
    tree = ast.parse(inspect.getsource(hs))
    offending = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "unknown per_angle_mode" in node.value:
                offending.append(node.value)
    assert not offending, (
        "hybrid-streaming module still contains an inline "
        f"'unknown per_angle_mode' reject path: {offending}"
    )
