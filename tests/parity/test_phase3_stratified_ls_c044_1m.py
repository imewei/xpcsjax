"""Phase 3 — C044 >=1M stratified-LS scaling-first no-worse-SSR oracle.

Maintainer-local LIVE data oracle, OFF by default. The empirical basin gate for
the Phase-3 scaling-first re-order of the heterodyne >=1M stratified-LS path
(Tasks 2-8). Mirrors the gating of
``tests/parity/test_engine_heterodyne_inmemory_scaling_first_c044.py`` (the
in-memory sibling): reads the real C044 dataset under ``${XPCSJAX_DATA_ROOT}/C044``
and SKIPS unless the config + cache are present, so CI / fresh clones never run it.

What it asserts
---------------
On REAL noisy C044 data cropped to a >=1M-point window (23 angles x 220 time
indices = 1,113,200 points), the CURRENT scaling-first stratified-LS path
(``fit_heterodyne_stratified_least_squares``, called directly) is no-worse-SSR vs
the captured pre-re-order baseline:

* ``averaged`` / ``individual`` — no-worse within 1e-3 (the same keep-better band
  the production global escapes use). Both modes ran on the stratified-LS path
  pre-Phase-3, so a captured baseline exists. A materially worse SSR is a REAL
  basin regression from the scaling-first re-order — diagnose it; do NOT loosen.

* ``constant`` — NO pre-Phase-3 stratified baseline exists: at ``3538de8`` the
  stratified path RAISED ``NotImplementedError`` for ``constant`` and routed to
  the in-memory joint fit. Phase 3 added ``constant`` to the stratified path. So
  this oracle asserts the durable, achievable contract for ``constant``: it now
  RUNS on stratified-LS, produces a FINITE physics-only SSR, and packs a
  physics-only vector (``n_scaling == 0`` -> ``len(parameters) == n_varying``).

The committed baseline (``fixtures/c044_stratified_ssr_baseline.json``) was
captured ONCE at the pre-re-order commit ``3538de8`` (the last commit before the
scaling-first re-order ``e44ec65``). It must NOT be overwritten from a post-re-order
checkout.

The resolved per-angle modes are forced via the production ``per_angle_mode``
token that ``_resolve_effective_mode`` maps at C044's ``n_phi == 23 >= 3``:
``constant -> "constant"``, ``individual -> "individual"``, ``averaged -> "auto"``
(``auto`` resolves to ``averaged`` for ``n_phi >= 3``; the bare ``"averaged"``
string is not a user-facing token).

Run locally::

    XPCSJAX_DATA_ROOT=/home/wei/Documents/Projects/data \
        uv run pytest tests/parity/test_phase3_stratified_ls_c044_1m.py -v
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

_DATA_ROOT = os.environ.get("XPCSJAX_DATA_ROOT", "/home/wei/Documents/Projects/data")
_C044_DIR = Path(_DATA_ROOT) / "C044"
_C044_CONFIG = _C044_DIR / "xpcsjax_config.yaml"
_C044_CACHE = _C044_DIR / "cached_frames_1000_2000.npz"

_GATE_OK = _C044_CONFIG.is_file() and _C044_CACHE.is_file()
_SKIP_REASON = (
    "Phase-3 C044 >=1M stratified-LS oracle is maintainer-local; set "
    "XPCSJAX_DATA_ROOT to a tree containing C044/xpcsjax_config.yaml + its cache "
    f"(looked in {_C044_DIR}). Never enabled in CI / fresh clones."
)

# Pre-re-order (physics-first) stratified-LS SSR, captured ONCE at commit 3538de8.
_BASELINE = Path(__file__).parent / "fixtures" / "c044_stratified_ssr_baseline.json"

# 23 angles x 220^2 = 1,113,200 points -> the >=1M stratified-LS regime.
_N_T = 220
_NO_WORSE_RTOL = 1e-3  # same keep-better band as the production global escapes

# Capture-label -> production per_angle_mode token (same mapping the baseline used).
_MODE_TO_PRODUCTION = {
    "constant": "constant",
    "individual": "individual",
    "averaged": "auto",
}


def _load_helpers():
    """Load the in-repo C044 loader by path (it lives under scripts/, not a package)."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "realdata_engine_fit_parity_c044.py"
    spec = importlib.util.spec_from_file_location("_phase3_c044_helpers", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_c044():
    helpers = _load_helpers()
    model, c2, phi = helpers.load_c044(str(_C044_CONFIG), n_t=_N_T)
    return model, np.asarray(c2, dtype=np.float64), np.asarray(phi, dtype=np.float64)


def _run_stratified(model, c2, phi, label):
    """Run the >=1M stratified-LS driver for an explicit mode; return (chi2, result)."""
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import (
        fit_heterodyne_stratified_least_squares,
    )

    cfg = NLSQConfig(per_angle_mode=_MODE_TO_PRODUCTION[label])
    res = fit_heterodyne_stratified_least_squares(
        model=model,
        c2=c2,
        phi=phi,
        config=cfg,
        weights=None,
        shuffle=False,
    )
    return float(res.chi_squared), res


@pytest.mark.skipif(not _GATE_OK, reason=_SKIP_REASON)
def test_c044_1m_constant_runs_and_is_finite():
    """``constant`` now RUNS on the >=1M stratified-LS path (it raised pre-Phase-3).

    No pre-re-order stratified baseline exists for ``constant`` (it routed to the
    in-memory joint fit at ``3538de8``), so the durable contract is: it runs on
    stratified-LS, the SSR is finite, and the packed vector is physics-only
    (``n_scaling == 0`` -> ``len(parameters) == n_varying``).
    """
    model, c2, phi = _load_c044()
    assert c2.size >= 1_000_000, f"C044 window must be >=1M points, got {c2.size}"

    chi2_constant, res_c = _run_stratified(model, c2, phi, "constant")
    assert np.isfinite(chi2_constant) and chi2_constant > 0.0

    n_physics = int(model.param_manager.n_varying)
    assert len(res_c.parameters) == n_physics, (
        f"constant must pack a physics-only vector (n_scaling=0); got "
        f"{len(res_c.parameters)} params vs n_varying={n_physics}"
    )
    assert (res_c.nlsq_diagnostics or {}).get("per_angle_mode") == "constant"


@pytest.mark.skipif(not _GATE_OK, reason=_SKIP_REASON)
@pytest.mark.skipif(not _BASELINE.is_file(), reason="pre-re-order SSR baseline not captured")
@pytest.mark.parametrize("mode", ["averaged", "individual"])
def test_c044_1m_stratified_no_worse_than_pre_reorder(mode):
    """Scaling-first stratified-LS SSR no-worse than the pre-re-order baseline.

    The empirical basin-neutrality gate for the Phase-3 scaling-first re-order: a
    materially worse SSR (beyond the 1e-3 keep-better band) is a REAL basin
    regression from the re-order — diagnose it; do NOT loosen the tolerance.
    """
    baseline = json.loads(_BASELINE.read_text())[mode]
    assert baseline.get("supported_pre_phase3") is True
    ssr_base = float(baseline["chi_squared"])
    assert np.isfinite(ssr_base) and ssr_base > 0.0

    model, c2, phi = _load_c044()
    assert c2.size >= 1_000_000, f"C044 window must be >=1M points, got {c2.size}"

    ssr_now, _res = _run_stratified(model, c2, phi, mode)
    assert np.isfinite(ssr_now)

    rel_excess = (ssr_now - ssr_base) / max(abs(ssr_base), 1e-300)
    assert ssr_now <= ssr_base * (1.0 + _NO_WORSE_RTOL), (
        f"{mode}: scaling-first stratified-LS SSR {ssr_now:.6e} worse than the "
        f"pre-re-order baseline {ssr_base:.6e} beyond rtol={_NO_WORSE_RTOL:.0e} "
        f"(rel_excess={rel_excess:.3e}). This is a REAL basin regression from the "
        "scaling-first re-order — diagnose it; do NOT loosen the tolerance."
    )
